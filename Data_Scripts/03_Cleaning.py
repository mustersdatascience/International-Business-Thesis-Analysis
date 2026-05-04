import pandas as pd
import numpy as np
import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

pd.set_option("display.float_format", lambda x: f"{x:.3f}")

# ============================================================
# STAP 1: INLADEN
# ============================================================
df = pd.read_csv("Data/02_Thesis_data_complete.csv")
print(f"Ingeladen: {len(df)} firms")
print(f"  US (LME): {len(df[df['Institutional_Context'] == 1])}")
print(f"  EU (CME): {len(df[df['Institutional_Context'] == 0])}")

# ============================================================
# STAP 2: MARKET CAP FILTER (EUR 25M)
# ============================================================
before = len(df)
df = df[df["Market_Cap_EUR"] >= 25]
dropped = before - len(df)
print(f"\nMarket cap filter (>= EUR 25M): {dropped} firms gedropped -> {len(df)} over")
print(f"  US: {len(df[df['Institutional_Context'] == 1])}")
print(f"  EU: {len(df[df['Institutional_Context'] == 0])}")

# ============================================================
# STAP 3: DROP MISSING Internationality_Ratio (55 firms)
# ============================================================
# MCAR-test toonde aan dat missingness niet random is:
# firms met missing Internationality_Ratio zijn systematisch kleiner
# (lagere Board_Size, Firm_Size_Ln, Market_Cap). Imputen zou
# een vertekend gemiddelde invullen. Droppen is eerlijker.
before = len(df)
df = df.dropna(subset=["Internationality_Ratio"])
dropped = before - len(df)
print(f"\nDrop missing Internationality_Ratio: {dropped} firms gedropped -> {len(df)} over")
print(f"  US: {len(df[df['Institutional_Context'] == 1])}")
print(f"  EU: {len(df[df['Institutional_Context'] == 0])}")

# ============================================================
# STAP 4: DROP MISSING AGE_DIVERSITY (1 firm)
# ============================================================
before = len(df)
df = df.dropna(subset=["Age_Diversity"])
dropped = before - len(df)
print(f"\nDrop missing Age_Diversity: {dropped} firms gedropped -> {len(df)} over")

# ============================================================
# STAP 5: WINSORIZING OP P1/P99
# ============================================================
# Capping van extreme waarden op het 1e en 99e percentiel.
# Standaard in corporate governance research (Adams, 2017).
# Voorkomt dat outliers afstandsmaten in cluster analysis
# vertekenen en OLS schattingen domineren.

winsorize_vars = ["RD_Intensity", "ROA", "Leverage"]

print(f"\nWinsorizing op P1/P99:")
for var in winsorize_vars:
    p1 = df[var].quantile(0.01)
    p99 = df[var].quantile(0.99)
    below = (df[var] < p1).sum()
    above = (df[var] > p99).sum()
    df[var] = df[var].clip(lower=p1, upper=p99)
    print(f"  {var}: P1={p1:.4f}, P99={p99:.4f} | {below} onder, {above} boven gecapped")

# ============================================================
# STAP 6: LOG-TRANSFORMATIE R&D INTENSITY
# ============================================================
# R&D Intensity is extreem rechtsscheef (skew=16.5, kurtosis=310).
# Log-transformatie is standaard voor ratio-variabelen in de
# corporate governance literatuur. Constante 0.001 toegevoegd
# om ln(0) te voorkomen voor firms met R&D Intensity dicht bij 0.

df["RD_Intensity_Ln"] = np.log(df["RD_Intensity"] + 0.001)

print(f"\nLog-transformatie R&D Intensity:")
print(f"  Raw:         mean={df['RD_Intensity'].mean():.3f}, median={df['RD_Intensity'].median():.3f}, skew={df['RD_Intensity'].skew():.3f}")
print(f"  Ln-versie:   mean={df['RD_Intensity_Ln'].mean():.3f}, median={df['RD_Intensity_Ln'].median():.3f}, skew={df['RD_Intensity_Ln'].skew():.3f}")

# ============================================================
# STAP 7: RANDOM SAMPLING US FIRMS
# ============================================================
# Om een gebalanceerde sample te krijgen met minimaal 25% EU firms,
# reduceren we de US sample via random sampling. Dit gebeurt NA
# alle cleaning stappen zodat de finale proporties kloppen.
#
# Methode: Random sampling (niet PSM) omdat:
# 1. PSM zou dubbel controleren (eerst sampling, dan regressie controls)
# 2. PSM gooit systematisch mega-caps weg -> verlies interessante configs
# 3. Random sampling behoudt externe validiteit van US sample
# 4. Firm-level confounders worden afgehandeld in regressie-stap

TARGET_EU_PERCENTAGE = 0.25
RANDOM_SEED = 42  # Voor reproduceerbaarheid

us_firms = df[df["Institutional_Context"] == 1]
eu_firms = df[df["Institutional_Context"] == 0]

n_eu = len(eu_firms)
# Bereken hoeveel US firms nodig voor 25% EU: n_eu / 0.25 = totaal, totaal - n_eu = n_us
target_us = int(n_eu / TARGET_EU_PERCENTAGE) - n_eu

print(f"\n--- RANDOM SAMPLING US FIRMS ---")
print(f"EU firms (behouden): {n_eu}")
print(f"US firms voor sampling: {len(us_firms)}")
print(f"Target US firms voor {TARGET_EU_PERCENTAGE*100:.0f}% EU: {target_us}")

# Check of we genoeg US firms hebben
if len(us_firms) < target_us:
    print(f"WAARSCHUWING: Niet genoeg US firms ({len(us_firms)}) voor target ({target_us})")
    print(f"Alle US firms worden behouden.")
    us_sampled = us_firms
else:
    us_sampled = us_firms.sample(n=target_us, random_state=RANDOM_SEED)
    print(f"US firms na sampling: {len(us_sampled)}")

# Combineer met alle EU firms
df = pd.concat([us_sampled, eu_firms], ignore_index=True)

print(f"Totaal na sampling: {len(df)}")
print(f"EU percentage: {n_eu/len(df)*100:.1f}%")

# ============================================================
# STAP 8: NORMALITEIT NA CLEANING
# ============================================================
from scipy import stats as sp_stats

print(f"\nNormaliteit na cleaning:")
print(f"  {'Variabele':25s} {'Skewness':>10s} {'Kurtosis':>10s} {'Oordeel':>12s}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12}")

check_vars = ["Board_Independence", "Gender_Diversity", "Avg_Tenure",
              "Internationality_Ratio", "Board_Size", "Board_Busyness",
              "Age_Diversity", "RD_Intensity", "RD_Intensity_Ln",
              "ROA", "Leverage", "Firm_Size_Ln"]

for var in check_vars:
    vals = df[var].dropna()
    skew = vals.skew()
    kurt = vals.kurtosis()
    if abs(skew) > 2 or abs(kurt) > 7:
        oordeel = "PROBLEEM"
    elif abs(skew) > 1:
        oordeel = "matig"
    else:
        oordeel = "ok"
    print(f"  {var:25s} {skew:10.3f} {kurt:10.3f} {oordeel:>12s}")

# ============================================================
# STAP 9: FINALE MISSING DATA CHECK
# ============================================================
analysis_vars = ["Board_Independence", "Gender_Diversity", "Avg_Tenure",
                 "CEO_Duality", "Internationality_Ratio", "Board_Size",
                 "Board_Busyness", "Age_Diversity", "Avg_Qualifications",
                 "RD_Intensity", "RD_Intensity_Ln", "Firm_Size_Ln",
                 "ROA", "Leverage"]

print(f"\nMissing data check:")
any_missing = False
for var in analysis_vars:
    m = df[var].isnull().sum()
    if m > 0:
        print(f"  {var}: {m} missing")
        any_missing = True
if not any_missing:
    print(f"  Geen missing values in analyse-variabelen")

# ============================================================
# STAP 10: DOLNICAR CHECK
# ============================================================
cluster_cols = ["Board_Independence", "Gender_Diversity", "Avg_Tenure",
                "Internationality_Ratio", "Board_Size", "Board_Busyness"]

n_cluster_vars = len(cluster_cols)
obs_per_var = len(df) / n_cluster_vars

print(f"\n--- DOLNICAR CHECK ---")
print(f"Clustering variabelen: {n_cluster_vars}")
print(f"Observaties per variabele: {obs_per_var:.1f} (minimum: 70)")
print(f"Status: {'OK' if obs_per_var >= 70 else 'WAARSCHUWING'}")

# ============================================================
# STAP 11: FINALE DATASET OVERZICHT
# ============================================================
us = df[df["Institutional_Context"] == 1]
eu = df[df["Institutional_Context"] == 0]

print(f"\n{'='*50}")
print(f"FINALE DATASET")
print(f"{'='*50}")
print(f"Totaal firms: {len(df)}")
print(f"  US (LME): {len(us)} ({len(us)/len(df)*100:.1f}%)")
print(f"  EU (CME): {len(eu)} ({len(eu)/len(df)*100:.1f}%)")
print(f"\nPer sector:")
print(f"  Tech:       {len(df[df['Sector'] == 'Tech'])}")
print(f"  Healthcare: {len(df[df['Sector'] == 'Healthcare'])}")
print(f"\nPer land:")
country_names = {
    "USA": "United States", "DEU": "Germany", "NLD": "Netherlands",
    "SWE": "Sweden", "FIN": "Finland", "DNK": "Denmark",
    "NOR": "Norway", "AUT": "Austria", "BEL": "Belgium", "CHE": "Switzerland",
}
for loc in df["Loc"].value_counts().index:
    n = len(df[df["Loc"] == loc])
    name = country_names.get(loc, loc)
    print(f"  {name:20s}: {n:4d}")

print(f"\nKolommen: {list(df.columns)}")

# ============================================================
# STAP 12: OPSLAAN
# ============================================================
df.to_csv("Data/03_Thesis_data_clean.csv", index=False)
print(f"\nOpgeslagen als '03_Thesis_data_clean.csv'")
print(f"Dimensies: {df.shape[0]} rijen x {df.shape[1]} kolommen")