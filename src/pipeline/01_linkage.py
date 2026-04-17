"""
Phase 1a: CFC–CAN Linkage Resolution
=====================================
Parses TED_NOTICE_URL from the CAN file to extract the TED notice number,
then maps it to FUTURE_CAN_ID in the CFC file to build a clean linkage table.

TED_NOTICE_URL format: "ted.europa.eu/udl?uri=TED:NOTICE:4-2018:TEXT:EN:HTML"
  → TED notice number = 4, year = 2018 → composite key = "4-2018" or padded "2018000004"

FUTURE_CAN_ID format in CFC: 2021265222 → year=2021, notice=265222
  → This is YYYYNNNNNN format (year prefix + 6-digit notice number)
"""

import os
import polars as pl
import re
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Procurement data")
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

CAN_FILE = os.path.join(DATA_DIR, "export_CAN_2023_2018.csv")
CFC_FILE = os.path.join(DATA_DIR, "export_CFC_2018_2023.csv")


def parse_ted_url(url: str) -> str | None:
    """
    Extract TED notice composite key from URL.
    'ted.europa.eu/udl?uri=TED:NOTICE:4-2018:TEXT:EN:HTML' -> '2018000004'
    """
    if not url:
        return None
    m = re.search(r'NOTICE:(\d+)-(\d{4})', url)
    if m:
        notice_num = int(m.group(1))
        year = int(m.group(2))
        # Pad notice to 6 digits: 2018000004
        return f"{year}{notice_num:06d}"
    return None


def future_can_id_to_key(fcan_id) -> str | None:
    """
    Convert FUTURE_CAN_ID (float like 2021265222.0) to composite key.
    2021265222 → year=2021, notice=265222 → '2021265222'
    Note: it's already in YYYYNNNNNN format (10 digits).
    """
    if fcan_id is None:
        return None
    try:
        val = str(int(float(fcan_id)))
        return val  # Already in YYYYNNNNNN format
    except (ValueError, TypeError):
        return None


print("=" * 60)
print("PHASE 1a: CFC-CAN LINKAGE RESOLUTION")
print("=" * 60)

# ── Step 1: Build CAN key table ─────────────────────────────────
print("\n[1/4] Reading CAN TED notice URLs...")
t0 = time.time()

can_urls = pl.read_csv(
    CAN_FILE,
    columns=["ID_NOTICE_CAN", "TED_NOTICE_URL", "YEAR"],
    infer_schema_length=5000,
    ignore_errors=True
)
print(f"  CAN rows loaded: {len(can_urls):,}  ({time.time()-t0:.1f}s)")

# Parse TED notice key from URL
print("  Parsing TED notice keys from URLs...")
ted_keys = []
for url in can_urls["TED_NOTICE_URL"].to_list():
    ted_keys.append(parse_ted_url(str(url) if url else ""))

can_urls = can_urls.with_columns(
    pl.Series("TED_KEY", ted_keys)
)

# Check parsing success
parsed = sum(1 for k in ted_keys if k is not None)
print(f"  Successfully parsed: {parsed:,} / {len(ted_keys):,} ({100*parsed/len(ted_keys):.1f}%)")

# Sample
print("\n  Sample TED keys:")
sample = can_urls.filter(pl.col("TED_KEY").is_not_null()).head(5)
for row in sample.iter_rows(named=True):
    print(f"    ID={row['ID_NOTICE_CAN']}  URL_snippet=...NOTICE:{row['TED_NOTICE_URL'].split('NOTICE:')[1][:15]}  KEY={row['TED_KEY']}")


# ── Step 2: Read CFC FUTURE_CAN_ID ──────────────────────────────
print("\n[2/4] Reading CFC FUTURE_CAN_ID values...")
t0 = time.time()

cfc_ids = pl.read_csv(
    CFC_FILE,
    columns=["ID_NOTICE_CN", "FUTURE_CAN_ID", "FUTURE_CAN_ID_ESTIMATED", "YEAR"],
    infer_schema_length=5000,
    ignore_errors=True
)
print(f"  CFC rows loaded: {len(cfc_ids):,}  ({time.time()-t0:.1f}s)")

# Convert FUTURE_CAN_ID to composite key
print("  Converting FUTURE_CAN_ID to composite keys...")
fcan_keys = []
for fid in cfc_ids["FUTURE_CAN_ID"].to_list():
    fcan_keys.append(future_can_id_to_key(fid))

cfc_ids = cfc_ids.with_columns(
    pl.Series("FUTURE_KEY", fcan_keys)
)

non_null_fcan = sum(1 for k in fcan_keys if k is not None)
print(f"  CFC records with FUTURE_CAN_ID: {non_null_fcan:,} / {len(fcan_keys):,} ({100*non_null_fcan/len(fcan_keys):.1f}%)")

# Sample
print("\n  Sample FUTURE_CAN_ID → KEY:")
sample = cfc_ids.filter(pl.col("FUTURE_KEY").is_not_null()).head(5)
for row in sample.iter_rows(named=True):
    print(f"    CFC_ID={row['ID_NOTICE_CN']}  FUTURE_CAN_ID={row['FUTURE_CAN_ID']}  KEY={row['FUTURE_KEY']}")


# ── Step 3: Join to build linkage table ─────────────────────────
print("\n[3/4] Joining CFC and CAN via TED key...")

# CAN lookup: TED_KEY → ID_NOTICE_CAN
can_lookup = can_urls.filter(pl.col("TED_KEY").is_not_null()).select(
    ["TED_KEY", "ID_NOTICE_CAN"]
).unique(subset=["TED_KEY"])  # deduplicate (CAN can have multiple lots)

print(f"  Unique TED keys in CAN: {len(can_lookup):,}")

# CFC with non-null future key
cfc_with_key = cfc_ids.filter(pl.col("FUTURE_KEY").is_not_null())
print(f"  CFC records with linkage key: {len(cfc_with_key):,}")

# Join
linkage = cfc_with_key.join(
    can_lookup,
    left_on="FUTURE_KEY",
    right_on="TED_KEY",
    how="left"
)

matched = linkage.filter(pl.col("ID_NOTICE_CAN").is_not_null())
unmatched = linkage.filter(pl.col("ID_NOTICE_CAN").is_null())

print(f"\n  ✓ Successfully linked:   {len(matched):,} ({100*len(matched)/len(cfc_with_key):.1f}%)")
print(f"  ✗ Unmatched CFC records: {len(unmatched):,} ({100*len(unmatched)/len(cfc_with_key):.1f}%)")

# ── Step 4: Save linkage table ───────────────────────────────────
print("\n[4/4] Saving linkage table...")
linkage_table = linkage.select([
    "ID_NOTICE_CN", "FUTURE_CAN_ID", "FUTURE_KEY", "ID_NOTICE_CAN"
])
linkage_table.write_parquet(f"{OUT_DIR}/cfc_can_linkage.parquet")
print(f"  Saved → {OUT_DIR}/cfc_can_linkage.parquet")

# Also save can key table for reuse
can_urls.select(["ID_NOTICE_CAN", "TED_KEY"]).write_parquet(f"{OUT_DIR}/can_ted_keys.parquet")
print(f"  Saved → {OUT_DIR}/can_ted_keys.parquet")

print("\n✅ Phase 1a complete.\n")
print(f"  Total CFC records:      {len(cfc_ids):,}")
print(f"  CFC with FUTURE_CAN_ID: {non_null_fcan:,}")
print(f"  Linked to CAN:          {len(matched):,}")
print(f"  Linkage rate:           {100*len(matched)/len(cfc_ids):.1f}% of all CFC records")
