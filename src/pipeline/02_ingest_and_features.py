"""
Phase 1b+1c: Full Ingestion + Feature Engineering (Streaming)
==============================================================
Uses polars streaming engine to handle 14M rows within 3GB RAM.

Strategy:
  Step 1: Stream CAN → aggregate by contract → can_outcomes.parquet
  Step 2: Stream CFC → deduplicate → cfc_deduped.parquet
  Step 3: Join (both now ~1.5M rows each, fits in RAM)
  Step 4: Feature engineering on joined dataframe
  Step 5: Save procedure_records.parquet + cfc_unlinked.parquet
"""

import polars as pl
import os, time

DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Procurement data")
PROC_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "processed")
FEAT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "features")
os.makedirs(FEAT_DIR, exist_ok=True)

CAN_FILE  = f"{DATA_DIR}/export_CAN_2023_2018.csv"
CFC_FILE  = f"{DATA_DIR}/export_CFC_2018_2023.csv"
LINK_FILE = f"{PROC_DIR}/cfc_can_linkage.parquet"

CPV_SECTORS = {
    "03":"Agriculture & Forestry","09":"Petroleum Products","14":"Mining & Quarrying",
    "15":"Food & Beverages","16":"Agricultural Machinery","18":"Clothing & Footwear",
    "22":"Printed Matter","24":"Chemical Products","30":"IT Equipment",
    "31":"Electrical Equipment","32":"Radio & Comms Equipment","33":"Medical & Pharma",
    "34":"Transport Equipment","35":"Security Equipment","38":"Laboratory Equipment",
    "39":"Furniture & Fittings","41":"Water & Utilities","42":"Industrial Machinery",
    "44":"Construction Materials","45":"Construction Works","48":"Software",
    "50":"Repair & Maintenance","51":"Installation Services","55":"Hotel & Restaurant",
    "60":"Transport Services","63":"Transport Support","64":"Postal Services",
    "65":"Gas & Electricity","66":"Financial Services","70":"Real Estate",
    "71":"Architecture & Engineering","72":"IT Services","73":"R&D Services",
    "75":"Public Administration","77":"Agricultural Services","79":"Business Services",
    "80":"Education","85":"Health & Social Work","90":"Waste & Environment",
    "92":"Recreation & Culture","98":"Other Services",
}
COUNTRY_CLUSTERS = {
    "BE":"Benelux","NL":"Benelux","LU":"Benelux",
    "DE":"Germanic","AT":"Germanic","CH":"Germanic",
    "FR":"Western","IT":"Western",
    "ES":"Iberian","PT":"Iberian",
    "SE":"Nordic","DK":"Nordic","FI":"Nordic","NO":"Nordic","IS":"Nordic",
    "PL":"CEE","CZ":"CEE","SK":"CEE","HU":"CEE","RO":"CEE","BG":"CEE",
    "LT":"Baltic","LV":"Baltic","EE":"Baltic",
    "HR":"Balkan","SI":"Balkan","MK":"Balkan",
    "GR":"Mediterranean","CY":"Mediterranean","MT":"Mediterranean",
    "IE":"Anglophone","UK":"Anglophone",
}
PROC_LABELS = {
    "OPE":"Open","RES":"Restricted","NIC":"Negotiated w/ prior call",
    "NOC":"Negotiated w/o competition","AWP":"Award w/o prior publication",
    "NOP":"Negotiated w/o prior call","COD":"Competitive dialogue",
    "INP":"Innovation partnership","NEC":"Negotiated (utilities)","NEG":"Negotiated",
}
CONTRACT_LABELS = {"S":"Services","U":"Supplies","W":"Works"}
CRIT_LABELS     = {"L":"Lowest price","M":"MEAT (best value)"}

print("="*60)
print("PHASE 1b+1c: INGESTION + FEATURE ENGINEERING")
print("="*60)

# ══════════════════════════════════════════════════════════════════
# STEP 1 — CAN: stream → aggregate → save
# ══════════════════════════════════════════════════════════════════
print("\n[1/5] Streaming CAN → contract-level outcomes...")
t0 = time.time()

can = (
    pl.scan_csv(CAN_FILE, infer_schema_length=5000, ignore_errors=True, null_values=["","NA","N/A"])
    .select(["ID_NOTICE_CAN","DT_AWARD","WIN_COUNTRY_CODE","B_CONTRACTOR_SME",
             "NUMBER_OFFERS","NUMBER_TENDERS_SME","NUMBER_TENDERS_OTHER_EU",
             "NUMBER_TENDERS_NON_EU","AWARD_VALUE_EURO","AWARD_EST_VALUE_EURO",
             "INFO_ON_NON_AWARD"])
    .with_columns([
        pl.col("NUMBER_OFFERS").cast(pl.Float64, strict=False),
        pl.col("NUMBER_TENDERS_OTHER_EU").cast(pl.Float64, strict=False),
        pl.col("NUMBER_TENDERS_NON_EU").cast(pl.Float64, strict=False),
        pl.col("NUMBER_TENDERS_SME").cast(pl.Float64, strict=False),
        pl.col("AWARD_VALUE_EURO").cast(pl.Float64, strict=False),
        pl.col("AWARD_EST_VALUE_EURO").cast(pl.Float64, strict=False),
    ])
    .group_by("ID_NOTICE_CAN")
    .agg([
        pl.col("DT_AWARD").first(),
        pl.col("WIN_COUNTRY_CODE").first(),
        pl.col("B_CONTRACTOR_SME").first(),
        pl.col("NUMBER_OFFERS").max(),
        pl.col("NUMBER_TENDERS_OTHER_EU").max(),
        pl.col("NUMBER_TENDERS_NON_EU").max(),
        pl.col("NUMBER_TENDERS_SME").max(),
        pl.col("AWARD_VALUE_EURO").sum(),
        pl.col("AWARD_EST_VALUE_EURO").sum(),
        pl.col("INFO_ON_NON_AWARD").first(),
    ])
    .collect(engine="streaming")
)
can.write_parquet(f"{FEAT_DIR}/can_outcomes.parquet", compression="zstd")
print(f"  {len(can):,} unique contracts  ({time.time()-t0:.1f}s)")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — CFC: stream → deduplicate → save
# ══════════════════════════════════════════════════════════════════
print("\n[2/5] Streaming CFC → unique notices...")
t0 = time.time()

CFC_COLS = [
    "ID_NOTICE_CN","YEAR","DT_DISPATCH","DT_APPLICATIONS",
    "ISO_COUNTRY_CODE","CAE_TYPE","MAIN_ACTIVITY",
    "TYPE_OF_CONTRACT","CPV","TAL_LOCATION_NUTS",
    "TOP_TYPE","B_ACCELERATED","B_GPA","B_EU_FUNDS",
    "B_FRA_AGREEMENT","B_DYN_PURCH_SYST","B_ELECTRONIC_AUCTION",
    "VALUE_EURO","LOTS_NUMBER","DURATION",
    "CRIT_CODE","CRIT_PRICE_WEIGHT","FUTURE_CAN_ID","CANCELLED",
]

cfc = (
    pl.scan_csv(CFC_FILE, infer_schema_length=5000, ignore_errors=True, null_values=["","NA","N/A"])
    .select(CFC_COLS)
    .unique(subset=["ID_NOTICE_CN"], keep="first")
    .collect(engine="streaming")
)
cfc.write_parquet(f"{FEAT_DIR}/cfc_deduped.parquet", compression="zstd")
print(f"  {len(cfc):,} unique notices  ({time.time()-t0:.1f}s)")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — Join CFC + linkage + CAN
# ══════════════════════════════════════════════════════════════════
print("\n[3/5] Joining CFC → linkage → CAN outcomes...")
t0 = time.time()

linkage = (
    pl.read_parquet(LINK_FILE)
    .select(["ID_NOTICE_CN","ID_NOTICE_CAN"])
    .unique()
    .with_columns([
        pl.col("ID_NOTICE_CN").cast(pl.Int64, strict=False),
        pl.col("ID_NOTICE_CAN").cast(pl.Int64, strict=False),
    ])
)

cfc = cfc.with_columns(pl.col("ID_NOTICE_CN").cast(pl.Int64, strict=False))
can = can.with_columns(pl.col("ID_NOTICE_CAN").cast(pl.Int64, strict=False))

proc = cfc.join(linkage, on="ID_NOTICE_CN", how="left").join(can, on="ID_NOTICE_CAN", how="left")

n_linked   = proc["ID_NOTICE_CAN"].is_not_null().sum()
n_unlinked = proc["ID_NOTICE_CAN"].is_null().sum()
print(f"  Total: {len(proc):,}  Linked: {n_linked:,} ({100*n_linked/len(proc):.1f}%)  Unlinked: {n_unlinked:,}  ({time.time()-t0:.1f}s)")

# ══════════════════════════════════════════════════════════════════
# STEP 4 — Feature engineering
# ══════════════════════════════════════════════════════════════════
print("\n[4/5] Engineering features...")
t0 = time.time()

# Dates
proc = proc.with_columns([
    pl.col("DT_DISPATCH").str.strptime(pl.Date, "%d/%m/%y", strict=False).alias("dt_publication"),
    pl.col("DT_APPLICATIONS").str.strptime(pl.Date, "%d/%m/%y", strict=False).alias("dt_deadline"),
    pl.col("DT_AWARD").str.strptime(pl.Date, "%d/%m/%y", strict=False).alias("dt_award"),
])

# Timing
proc = proc.with_columns([
    (pl.col("dt_deadline") - pl.col("dt_publication")).dt.total_days().alias("_p"),
    (pl.col("dt_award")    - pl.col("dt_publication")).dt.total_days().alias("_d"),
]).with_columns([
    pl.when(pl.col("_p").is_between(1,365)).then(pl.col("_p")).otherwise(None).alias("prep_time_days"),
    pl.when(pl.col("_d").is_between(0,1095)).then(pl.col("_d")).otherwise(None).alias("proc_duration_days"),
]).drop(["_p","_d"])

# CPV
proc = proc.with_columns([
    pl.col("CPV").cast(pl.Utf8).str.slice(0,2).alias("cpv_division"),
    pl.col("CPV").cast(pl.Utf8).str.slice(0,5).alias("cpv_group"),
])
proc = proc.with_columns(
    pl.Series("cpv_sector", [CPV_SECTORS.get(str(d),"Other") if d else "Unknown" for d in proc["cpv_division"].to_list()])
)

# Value
proc = proc.with_columns([
    pl.col("VALUE_EURO").cast(pl.Float64, strict=False).alias("value_euro"),
    pl.col("AWARD_VALUE_EURO").cast(pl.Float64, strict=False).alias("award_value_euro"),
    pl.col("AWARD_EST_VALUE_EURO").cast(pl.Float64, strict=False).alias("est_value_euro"),
])
proc = proc.with_columns([
    pl.when(pl.col("value_euro") > 0).then(pl.col("value_euro").log(base=10.0)).otherwise(None).alias("log10_value"),
    pl.when((pl.col("award_value_euro") > 0) & (pl.col("est_value_euro") > 0))
      .then((pl.col("award_value_euro") / pl.col("est_value_euro")).clip(0.1, 3.0))
      .otherwise(None).alias("price_ratio"),
])

proc = proc.with_columns(
    pl.when(pl.col("value_euro").is_null() | (pl.col("value_euro") <= 0)).then(pl.lit("Unknown"))
      .when(pl.col("value_euro") < 135_000).then(pl.lit("Below 135k"))
      .when(pl.col("value_euro") < 215_000).then(pl.lit("135k-215k"))
      .when(pl.col("value_euro") < 431_000).then(pl.lit("215k-431k"))
      .when(pl.col("value_euro") < 5_000_000).then(pl.lit("431k-5M"))
      .when(pl.col("value_euro") < 50_000_000).then(pl.lit("5M-50M"))
      .otherwise(pl.lit(">50M"))
      .alias("value_bracket")
)

# Competition
proc = proc.with_columns([
    pl.col("NUMBER_OFFERS").cast(pl.Float64, strict=False).alias("n_offers"),
    pl.col("NUMBER_TENDERS_OTHER_EU").cast(pl.Float64, strict=False).alias("n_offers_crossborder"),
    pl.col("NUMBER_TENDERS_NON_EU").cast(pl.Float64, strict=False).alias("n_offers_noneu"),
    pl.col("NUMBER_TENDERS_SME").cast(pl.Float64, strict=False).alias("n_offers_sme"),
]).with_columns([
    pl.when(pl.col("n_offers").is_between(0,200)).then(pl.col("n_offers")).otherwise(None).alias("n_offers"),
]).with_columns([
    pl.when(pl.col("n_offers").is_not_null()).then((pl.col("n_offers")<=1).cast(pl.Int8)).otherwise(None).alias("single_bid_flag"),
])

# Cross-border
# WIN_COUNTRY_CODE can be a '---'-separated list for multi-lot/consortium contracts
# (e.g. "PL---PL---PL" for a domestic consortium, or "IT---RO" for a mixed one).
# A simple string comparison ("PL---PL---PL" != "PL") would produce false positives.
# Correct logic: cross_border_win = 1 if ANY valid 2-char winner code differs from buyer.
def _is_cross_border(win_code: str, buyer: str) -> int | None:
    if not win_code or not buyer:
        return None
    parts = [p.strip() for p in win_code.split("---") if len(p.strip()) == 2]
    if not parts:
        return None
    return int(any(p != buyer for p in parts))

proc = proc.with_columns(
    pl.struct(["WIN_COUNTRY_CODE", "ISO_COUNTRY_CODE"]).map_elements(
        lambda x: _is_cross_border(x["WIN_COUNTRY_CODE"], x["ISO_COUNTRY_CODE"]),
        return_dtype=pl.Int8,
    ).alias("cross_border_win")
)

# SME winner
def psme(v):
    if v is None: return None
    s = str(v).split("---")[0].strip()
    return 1 if s=="Y" else (0 if s=="N" else None)
proc = proc.with_columns(
    pl.Series("sme_winner", [psme(v) for v in proc["B_CONTRACTOR_SME"].to_list()], dtype=pl.Int8)
)

# Country cluster
proc = proc.with_columns(
    pl.Series("country_cluster", [COUNTRY_CLUSTERS.get(str(c),"Other") if c else "Unknown" for c in proc["ISO_COUNTRY_CODE"].to_list()])
)

# Labels
proc = proc.with_columns([
    pl.Series("procedure_label",    [PROC_LABELS.get(str(t),"Other") if t else "Unknown" for t in proc["TOP_TYPE"].to_list()]),
    pl.Series("criteria_label",     [CRIT_LABELS.get(str(c),"Other") if c else "Unknown" for c in proc["CRIT_CODE"].to_list()]),
    pl.Series("contract_type_label",[CONTRACT_LABELS.get(str(t),str(t)) if t else "Unknown" for t in proc["TYPE_OF_CONTRACT"].to_list()]),
])
proc = proc.with_columns(pl.col("CRIT_PRICE_WEIGHT").cast(pl.Float64, strict=False).alias("price_weight_pct"))
proc = proc.with_columns(pl.col("DURATION").cast(pl.Float64, strict=False).alias("contract_duration_months"))

# Boolean flags
for col in ["B_GPA","B_EU_FUNDS","B_FRA_AGREEMENT","B_DYN_PURCH_SYST","B_ELECTRONIC_AUCTION","B_ACCELERATED","CANCELLED"]:
    if col in proc.columns:
        proc = proc.with_columns(
            pl.when(pl.col(col).cast(pl.Utf8).is_in(["Y","1"])).then(pl.lit(1))
              .when(pl.col(col).cast(pl.Utf8).is_in(["N","0"])).then(pl.lit(0))
              .otherwise(None).cast(pl.Int8).alias(f"flag_{col.lower()}")
        )

print(f"  Done ({time.time()-t0:.1f}s)  —  {proc.width} columns")

# ══════════════════════════════════════════════════════════════════
# STEP 5 — Select final columns and save
# ══════════════════════════════════════════════════════════════════
print("\n[5/5] Saving feature store...")

KEEP = [
    "ID_NOTICE_CN","ID_NOTICE_CAN","YEAR",
    "ISO_COUNTRY_CODE","country_cluster","CAE_TYPE","MAIN_ACTIVITY",
    "TOP_TYPE","procedure_label","TYPE_OF_CONTRACT","contract_type_label",
    "CPV","cpv_division","cpv_group","cpv_sector","TAL_LOCATION_NUTS",
    "CRIT_CODE","criteria_label","price_weight_pct",
    "contract_duration_months","LOTS_NUMBER",
    "value_euro","est_value_euro","log10_value","value_bracket",
    "dt_publication","dt_deadline","dt_award",
    "prep_time_days","proc_duration_days",
    "flag_b_gpa","flag_b_eu_funds","flag_b_fra_agreement",
    "flag_b_dyn_purch_syst","flag_b_electronic_auction",
    "flag_b_accelerated","flag_cancelled",
    "n_offers","single_bid_flag",
    "n_offers_crossborder","n_offers_noneu","n_offers_sme",
    "cross_border_win","sme_winner",
    "award_value_euro","price_ratio",
    "WIN_COUNTRY_CODE",
]
avail = [c for c in KEEP if c in proc.columns]
proc_final = proc.select(avail)

linked   = proc_final.filter(pl.col("ID_NOTICE_CAN").is_not_null())
unlinked = proc_final.filter(pl.col("ID_NOTICE_CAN").is_null())

linked.write_parquet(f"{FEAT_DIR}/procedure_records.parquet",  compression="zstd")
unlinked.write_parquet(f"{FEAT_DIR}/cfc_unlinked.parquet",     compression="zstd")

pr_mb = os.path.getsize(f"{FEAT_DIR}/procedure_records.parquet")/1e6
ul_mb = os.path.getsize(f"{FEAT_DIR}/cfc_unlinked.parquet")/1e6
print(f"  procedure_records.parquet : {pr_mb:.1f} MB  ({len(linked):,} rows)")
print(f"  cfc_unlinked.parquet      : {ul_mb:.1f} MB  ({len(unlinked):,} rows)")

# ── Summary ───────────────────────────────────────────────────────
print("\n"+"="*60)
print("FEATURE STORE SUMMARY")
print("="*60)
df = linked

def stat(col, label, pct=False):
    s = df[col].drop_nulls()
    if len(s)==0: print(f"  {label}: no data"); return
    if pct: print(f"  {label}: {s.mean()*100:.1f}%  (n={len(s):,})")
    else:   print(f"  {label}: median={s.median():.1f}  mean={s.mean():.1f}  n={len(s):,}")

print(f"\nLinked records: {len(df):,}  |  Columns: {df.width}")
print("\n── Timing ──")
stat("prep_time_days","Prep time (days)")
stat("proc_duration_days","Procedure duration (days)")
print("\n── Competition ──")
stat("n_offers","Offers received")
stat("single_bid_flag","Single-bid rate", pct=True)
print("\n── Cross-border & SME ──")
stat("cross_border_win","Cross-border win rate", pct=True)
stat("sme_winner","SME winner rate", pct=True)
print("\n── Price ──")
stat("price_ratio","Price ratio (award/estimate)")
print("\n── Procedure types (top 8) ──")
for r in df.group_by("TOP_TYPE").agg(pl.len().alias("n")).sort("n",descending=True).head(8).iter_rows(named=True):
    print(f"  {r['TOP_TYPE']:8s}  {r['n']:>8,}  ({100*r['n']/len(df):.1f}%)")
print("\n── Top 10 countries ──")
for r in df.group_by("ISO_COUNTRY_CODE").agg(pl.len().alias("n")).sort("n",descending=True).head(10).iter_rows(named=True):
    print(f"  {r['ISO_COUNTRY_CODE']:5s}  {r['n']:>8,}  ({100*r['n']/len(df):.1f}%)")
print("\n── Top 8 CPV sectors ──")
for r in df.group_by("cpv_sector").agg(pl.len().alias("n")).sort("n",descending=True).head(8).iter_rows(named=True):
    print(f"  {r['cpv_sector']:35s}  {r['n']:>7,}")

print("\n✅ Phase 1b+1c complete.\n")
