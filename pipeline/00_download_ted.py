"""
Step 0: Download TED CSV bulk files from the EU Open Data Portal.

Two file types per year:
  CAN = Contract Award Notices  (outcomes: n_offers, award value, winner country)
  CN  = Contract Notices        (inputs: procedure type, CPV, criteria, prep time)

Confirmed download URLs (EU Data Portal API):
  https://data.europa.eu/api/hub/store/data/ted-contract-award-notices-{year}.zip
  https://data.europa.eu/api/hub/store/data/ted-contract-notices-{year}.zip

Combined 2018-2023 bundles also available:
  https://data.europa.eu/api/hub/store/data/ted-contract-award-notices-2018-2023.zip
  https://data.europa.eu/api/hub/store/data/ted-contract-notices-2018-2023.zip

Usage:
  python pipeline/00_download_ted.py                      # combined 2018-2023
  python pipeline/00_download_ted.py --years 2022 2023    # specific years
  python pipeline/00_download_ted.py --combined           # explicit combined bundle
"""

import argparse
import csv
import glob
import io
import logging
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "Procurement data"
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "https://data.europa.eu/api/hub/store/data"

# Final merged filenames expected by src/pipeline/01_linkage.py
OUT_CAN = DATA_DIR / "export_CAN_2023_2018.csv"
OUT_CFC = DATA_DIR / "export_CFC_2018_2023.csv"


def _progress_reporthook(label: str):
    """urllib reporthook that logs download progress every ~10%."""
    state = {"last_pct": -1, "start": time.time()}

    def hook(count, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(100, int(count * block_size * 100 / total_size))
        if pct >= state["last_pct"] + 10:
            elapsed = time.time() - state["start"]
            mb = count * block_size / 1_048_576
            logger.info("  %s  %d%%  (%.1f MB, %.0fs)", label, pct, mb, elapsed)
            state["last_pct"] = pct

    return hook


def download_zip(url: str, dest: Path, label: str) -> Path:
    """Download a ZIP file with progress logging."""
    logger.info("Downloading %s → %s", label, dest.name)
    tmp = dest.with_suffix(".zip.tmp")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress_reporthook(label))
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1_048_576
        logger.info("  ✓ Saved %.1f MB → %s", size_mb, dest)
        return dest
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def extract_csv_from_zip(zip_path: Path, out_csv: Path, notice_type: str) -> None:
    """
    Extract and concatenate all CSV files from a ZIP into a single output CSV.
    notice_type is 'CAN' or 'CFC' — used only for logging.
    """
    logger.info("Extracting %s from %s ...", notice_type, zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        if not csv_names:
            raise RuntimeError(f"No CSV files found inside {zip_path}")
        logger.info("  Found %d CSV file(s): %s", len(csv_names), ", ".join(csv_names))

        header_written = False
        total_rows = 0
        with open(out_csv, "w", newline="", encoding="utf-8") as fout:
            writer = None
            for name in csv_names:
                logger.info("  Merging %s ...", name)
                with zf.open(name) as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                    reader = csv.reader(text)
                    for i, row in enumerate(reader):
                        if i == 0:
                            if not header_written:
                                writer = csv.writer(fout)
                                writer.writerow(row)
                                header_written = True
                            # skip header on subsequent files
                            continue
                        writer.writerow(row)
                        total_rows += 1

    logger.info("  ✓ %s: %d data rows → %s", notice_type, total_rows, out_csv.name)


def download_combined(can_out: Path, cfc_out: Path) -> None:
    """Download the pre-bundled 2018-2023 combined ZIPs (single download each)."""
    can_zip = DATA_DIR / "ted-can-2018-2023.zip"
    cfc_zip = DATA_DIR / "ted-cfc-2018-2023.zip"

    download_zip(f"{BASE_URL}/ted-contract-award-notices-2018-2023.zip", can_zip, "CAN 2018-2023")
    extract_csv_from_zip(can_zip, can_out, "CAN")
    can_zip.unlink(missing_ok=True)

    download_zip(f"{BASE_URL}/ted-contract-notices-2018-2023.zip", cfc_zip, "CFC 2018-2023")
    extract_csv_from_zip(cfc_zip, cfc_out, "CFC")
    cfc_zip.unlink(missing_ok=True)


def download_by_years(years: list[int], can_out: Path, cfc_out: Path) -> None:
    """Download per-year ZIPs and concatenate into the combined output files."""
    can_parts, cfc_parts = [], []

    for year in sorted(years):
        can_zip = DATA_DIR / f"ted-can-{year}.zip"
        cfc_zip = DATA_DIR / f"ted-cfc-{year}.zip"
        can_csv = DATA_DIR / f"can_{year}_tmp.csv"
        cfc_csv = DATA_DIR / f"cfc_{year}_tmp.csv"

        download_zip(f"{BASE_URL}/ted-contract-award-notices-{year}.zip", can_zip, f"CAN {year}")
        extract_csv_from_zip(can_zip, can_csv, f"CAN {year}")
        can_zip.unlink(missing_ok=True)
        can_parts.append(can_csv)

        download_zip(f"{BASE_URL}/ted-contract-notices-{year}.zip", cfc_zip, f"CFC {year}")
        extract_csv_from_zip(cfc_zip, cfc_csv, f"CFC {year}")
        cfc_zip.unlink(missing_ok=True)
        cfc_parts.append(cfc_csv)

    _merge_csv_parts(can_parts, can_out, "CAN")
    _merge_csv_parts(cfc_parts, cfc_out, "CFC")


def _merge_csv_parts(parts: list[Path], out: Path, label: str) -> None:
    logger.info("Merging %d %s part(s) → %s", len(parts), label, out.name)
    header_written = False
    total = 0
    with open(out, "w", newline="", encoding="utf-8") as fout:
        writer = None
        for part in parts:
            with open(part, newline="", encoding="utf-8") as fin:
                reader = csv.reader(fin)
                for i, row in enumerate(reader):
                    if i == 0:
                        if not header_written:
                            writer = csv.writer(fout)
                            writer.writerow(row)
                            header_written = True
                        continue
                    writer.writerow(row)
                    total += 1
            part.unlink(missing_ok=True)
    logger.info("  ✓ %s merged: %d rows → %s", label, total, out.name)


def main():
    parser = argparse.ArgumentParser(description="Download TED CSV bulk data from EU portal")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--combined", action="store_true",
                       help="Download pre-bundled 2018-2023 combined ZIPs (default)")
    group.add_argument("--years", nargs="+", type=int, metavar="YEAR",
                       help="Download specific years, e.g. --years 2022 2023")
    parser.add_argument("--can-out", type=Path, default=OUT_CAN,
                        help="Output path for merged CAN CSV")
    parser.add_argument("--cfc-out", type=Path, default=OUT_CFC,
                        help="Output path for merged CFC CSV")
    args = parser.parse_args()

    can_out = Path(args.can_out)
    cfc_out = Path(args.cfc_out)

    logger.info("=" * 60)
    logger.info("TED CSV Download")
    logger.info("Output CAN → %s", can_out)
    logger.info("Output CFC → %s", cfc_out)
    logger.info("=" * 60)

    t0 = time.time()
    if args.years:
        download_by_years(args.years, can_out, cfc_out)
    else:
        download_combined(can_out, cfc_out)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Download complete in %.0fs", elapsed)
    logger.info("  CAN: %s (%.1f MB)", can_out, can_out.stat().st_size / 1_048_576)
    logger.info("  CFC: %s (%.1f MB)", cfc_out, cfc_out.stat().st_size / 1_048_576)


if __name__ == "__main__":
    main()
