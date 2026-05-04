import pandas as pd
import numpy as np
import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ============================================================
# STAP 1: INLADEN
# ============================================================
na = pd.read_csv("Data/Raw_Compustat-NorthAmerica-Final.csv")
gl = pd.read_csv("Data/Raw_Compustat-Global-Final.csv")
gl_price = pd.read_csv("Data/Raw_Compustat-SharePrice-Global-Final.csv")

print(f"NA raw: {len(na)} rijen")
print(f"Global raw: {len(gl)} rijen")
print(f"Global share price raw: {len(gl_price)} rijen")

na = na.rename(columns={"Gcis-Sub": "Gcis_Sub"})
gl = gl.rename(columns={"Gcis-Sub": "Gcis_Sub"})

# ============================================================
# GICS CODES DEFINITIE
# ============================================================
# GICS 45: alle Information Technology subcodes
# GICS 35: alleen R&D-intensieve Healthcare subcodes
healthcare_rd_subcodes = [35201010, 35202010, 35203010, 35103010]

def is_valid_gics(row):
    gcis = row["Gcis"]
    gcis_sub = row["Gcis_Sub"]
    if gcis == 45:
        return True
    if gcis_sub in healthcare_rd_subcodes:
        return True
    return False

# ============================================================
# STAP 2: SHARE PRICE MERGEN MET GLOBAL
# ============================================================

gl_price = gl_price[["Isin", "Share_End"]].drop_duplicates(subset=["Isin"], keep="first")
print(f"Global share price na dedup: {len(gl_price)} rijen")

gl = gl.merge(gl_price, on="Isin", how="left")
print(f"Global na share price merge: {len(gl)} rijen")

# ============================================================
# STAP 3: FILTEREN
# ============================================================

na = na[na["Loc"] == "USA"]
na = na[na.apply(is_valid_gics, axis=1)]
print(f"\nNA na filter (USA + GICS): {len(na)} rijen")

cme_countries = ["DEU", "NLD", "SWE", "FIN", "DNK", "NOR", "AUT", "BEL", "CHE"]
gl = gl[gl["Loc"].isin(cme_countries)]
gl = gl[gl.apply(is_valid_gics, axis=1)]
print(f"Global na filter (CME + GICS): {len(gl)} rijen")

gl_before = len(gl)
gl = gl.drop_duplicates(subset=["Comp_Id"], keep="first")
print(f"Global na dedup: {len(gl)} rijen (dropped {gl_before - len(gl)} duplicates)")

# ============================================================
# STAP 4: MARKET CAP BEREKENEN
# ============================================================

na["Market_Cap"] = pd.to_numeric(na["Share_End"], errors="coerce") * pd.to_numeric(na["Shares_Out"], errors="coerce")
gl["Market_Cap"] = pd.to_numeric(gl["Share_End"], errors="coerce") * pd.to_numeric(gl["Shares_Out"], errors="coerce")

print(f"\nNA missing market cap: {na['Market_Cap'].isnull().sum()}")
print(f"Global missing market cap: {gl['Market_Cap'].isnull().sum()}")

# ============================================================
# STAP 5: ISIN GENEREREN VOOR US FIRMS (CUSIP -> ISIN)
# ============================================================
# ISIN formaat: 2-letter country code + 9-char NSIN + 1 check digit
# Voor US firms: NSIN = 9-digit CUSIP (gepad met leading zeros)
# Check digit via Luhn-algoritme op de alfanumerieke string

def compute_isin_check_digit(isin_11):
    """Bereken ISIN check digit via dubbel-Luhn op alfanumerieke string."""
    digits = ""
    for ch in isin_11:
        if ch.isdigit():
            digits += ch
        elif ch.isalpha():
            digits += str(ord(ch.upper()) - 55)  # A=10, B=11, ..., Z=35
    
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    
    return str((10 - (total % 10)) % 10)

def cusip_to_isin(cusip):
    """Converteer een Compustat CUSIP naar een volledig 12-karakter ISIN."""
    if pd.isna(cusip):
        return None
    cusip_str = str(cusip).strip()
    if cusip_str == "" or cusip_str == "nan":
        return None
    cusip_str = cusip_str.zfill(9)  # pad naar 9 karakters (alfanumeriek)
    base = f"US{cusip_str}"
    check = compute_isin_check_digit(base)
    return f"{base}{check}"

na["Isin"] = na["Cusip"].apply(cusip_to_isin)

isin_generated = na["Isin"].notna().sum()
isin_failed = na["Isin"].isna().sum()
print(f"\nISIN generatie: {isin_generated} geslaagd, {isin_failed} gefaald")

# Global heeft al ISINs, CUSIP kolom op None
gl["Cusip"] = None

# ============================================================
# STAP 6: SAMENVOEGEN
# ============================================================
df = pd.concat([na, gl], ignore_index=True)
print(f"\nSamengevoegd: {len(df)} firms totaal")

# ============================================================
# STAP 7: INSTITUTIONAL CONTEXT + SECTOR LABEL
# ============================================================
df["Institutional_Context"] = df["Loc"].apply(lambda x: 1 if x == "USA" else 0)
df["Sector"] = df["Gcis"].apply(lambda x: "Tech" if x == 45 else "Healthcare")

# ============================================================
# STAP 8: MARKET CAP OVERZICHT
# ============================================================
print(f"\nMarket Cap statistieken per land:")
print(df.groupby("Loc")["Market_Cap"].describe().to_string())

# ============================================================
# STAP 9: AFGELEIDE VARIABELEN
# ============================================================

df["RD_Intensity"] = pd.to_numeric(df["RD_Expense"], errors="coerce") / pd.to_numeric(df["Rev_Total"], errors="coerce")
df["Firm_Size_Ln"] = np.log(pd.to_numeric(df["Total_Assets"], errors="coerce").replace(0, np.nan))
df["ROA"] = pd.to_numeric(df["Net_Income"], errors="coerce") / pd.to_numeric(df["Total_Assets"], errors="coerce")
df["Leverage"] = (
    pd.to_numeric(df["Debt_Liabilities"], errors="coerce").fillna(0)
    + pd.to_numeric(df["Debt_Longterm"], errors="coerce").fillna(0)
) / pd.to_numeric(df["Total_Assets"], errors="coerce")

# ============================================================
# STAP 10: DROP ONNODIGE KOLOMMEN
# ============================================================
drop_cols = ["Active", "Fyear"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ============================================================
# STAP 11: MISSING DATA CHECK
# ============================================================
print("\n--- MISSING DATA ---")
for col in ["RD_Expense", "Rev_Total", "Total_Assets", "Net_Income", "Share_End", "Market_Cap", "Isin"]:
    if col in df.columns:
        missing = df[col].isnull().sum()
        empty = (df[col] == "").sum() if df[col].dtype == object else 0
        print(f"{col}: {missing + empty} missing/empty")

# ============================================================
# STAP 12: OVERZICHT
# ============================================================
print(f"\n--- SAMPLE OVERZICHT ---")
print(f"Totaal firms: {len(df)}")
print(f"US (LME): {len(df[df['Institutional_Context'] == 1])}")
print(f"EU (CME): {len(df[df['Institutional_Context'] == 0])}")
print(f"\nPer sector:")
print(df["Sector"].value_counts())
print(f"\nLanden verdeling:")
print(df["Loc"].value_counts())

# ============================================================
# STAP 13: VALUTA CONVERSIE MARKET CAP NAAR EUR
# ============================================================
conversion_rates = {
    "USD": 0.9662,
    "SEK": 0.08727,
    "EUR": 1.0,
    "DKK": 0.1341,
    "NOK": 0.08487,
    "CHF": 1.0644,
}

def convert_to_eur(row):
    curr = row["Curr"]
    mc = row["Market_Cap"]
    if pd.isna(mc) or pd.isna(curr):
        return np.nan
    return mc * conversion_rates.get(curr, np.nan)

df["Market_Cap_EUR"] = df.apply(convert_to_eur, axis=1)
print(f"\n--- NA VALUTA CONVERSIE ---")
print(f"Missing Market_Cap_EUR: {df['Market_Cap_EUR'].isnull().sum()}")

# ============================================================
# STAP 14: DROP MISSING DATA
# ============================================================
before = len(df)
df = df.dropna(subset=["Market_Cap_EUR"])
print(f"Dropped {before - len(df)} firms zonder market cap -> {len(df)} over")

before = len(df)
df = df.dropna(subset=["Rev_Total"])
df = df[df["Rev_Total"] > 0]
print(f"Dropped {before - len(df)} firms zonder/0 revenue -> {len(df)} over")

before = len(df)
df = df.dropna(subset=["RD_Expense"])
print(f"Dropped {before - len(df)} firms zonder R&D data -> {len(df)} over")

before = len(df)
df = df[df["RD_Expense"] > 0]
print(f"Dropped {before - len(df)} firms met RD_Expense = 0 -> {len(df)} over")

before = len(df)
df = df.dropna(subset=["Rev_Total", "Total_Assets"])
print(f"Dropped {before - len(df)} firms zonder revenue/assets -> {len(df)} over")

# ============================================================
# STAP 15: MARKET CAP FILTER
# ============================================================
before = len(df)
df = df[df["Market_Cap_EUR"] >= 0]
print(f"Dropped {before - len(df)} firms onder market cap threshold -> {len(df)} over")

# ============================================================
# STAP 16: ISIN VALIDATIE
# ============================================================
isin_coverage = df["Isin"].notna().sum()
isin_missing = df["Isin"].isna().sum()
print(f"\n--- ISIN DEKKING IN FINALE DATASET ---")
print(f"Met ISIN: {isin_coverage}")
print(f"Zonder ISIN: {isin_missing}")

# ============================================================
# STAP 17: DATASET OVERZICHT
# ============================================================
print(f"\n============================")
print(f"FINALE DATASET")
print(f"============================")
print(f"Totaal firms: {len(df)}")
print(f"\nInstitutionele context:")
print(f"  US (LME):  {len(df[df['Institutional_Context'] == 1])}")
print(f"  EU (CME):  {len(df[df['Institutional_Context'] == 0])}")
print(f"\nPer sector:")
print(f"  Tech:       {len(df[df['Sector'] == 'Tech'])}")
print(f"  Healthcare: {len(df[df['Sector'] == 'Healthcare'])}")
print(f"\nPer land:")
for country in df["Loc"].value_counts().index:
    print(f"  {country}: {len(df[df['Loc'] == country])}")

# ============================================================
# STAP 18: OPSLAAN
# ============================================================
df.to_csv("Data/01_compustat_merged_clean.csv", index=False)
print(f"\nOpgeslagen als '01_compustat_merged_clean.csv'")
print(f"Kolommen: {list(df.columns)}")