import pandas as pd
import numpy as np
import os
from scipy import stats as sp_stats

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")

# ============================================================
# INLADEN
# ============================================================
df = pd.read_csv("Data/03_Thesis_data_clean.csv")
us = df[df["Institutional_Context"] == 1]
eu = df[df["Institutional_Context"] == 0]

# ============================================================
# 1. SAMPLE COMPOSITIE
# ============================================================
print("=" * 70)
print("1. SAMPLE COMPOSITIE")
print("=" * 70)

print(f"\nTotaal: {len(df)} firms")
print(f"  LME (US):  {len(us)} ({len(us)/len(df)*100:.1f}%)")
print(f"  CME (EU):  {len(eu)} ({len(eu)/len(df)*100:.1f}%)")

print(f"\nPer sector:")
for sector in ["Tech", "Healthcare"]:
    n = len(df[df["Sector"] == sector])
    n_us = len(us[us["Sector"] == sector])
    n_eu = len(eu[eu["Sector"] == sector])
    print(f"  {sector:12s}: {n:4d} totaal | US: {n_us:4d} | EU: {n_eu:3d}")

print(f"\nPer land:")
country_names = {
    "USA": "United States", "DEU": "Germany", "NLD": "Netherlands",
    "SWE": "Sweden", "FIN": "Finland", "DNK": "Denmark",
    "NOR": "Norway", "AUT": "Austria", "BEL": "Belgium", "CHE": "Switzerland",
}
for loc in df["Loc"].value_counts().index:
    n = len(df[df["Loc"] == loc])
    name = country_names.get(loc, loc)
    context = "LME" if loc == "USA" else "CME"
    print(f"  {name:20s} ({context}): {n:4d} ({n/len(df)*100:.1f}%)")

print(f"\nPer GICS subsector (top 10):")
sub_counts = df.groupby(["Gcis_Sub", "Sector"]).size().reset_index(name="n")
sub_counts = sub_counts.sort_values("n", ascending=False).head(10)
for _, row in sub_counts.iterrows():
    print(f"  {int(row['Gcis_Sub'])} ({row['Sector']:12s}): {row['n']:4d}")

# ============================================================
# 2. DESCRIPTIVE STATISTICS - FULL SAMPLE
# ============================================================
print("\n" + "=" * 70)
print("2. DESCRIPTIVE STATISTICS - FULL SAMPLE")
print("=" * 70)

board_vars = [
    "Board_Independence", "Gender_Diversity", "Avg_Tenure",
    "CEO_Duality", "Internationality_Ratio", "Board_Size",
    "Board_Busyness", "Age_Diversity", "Avg_Qualifications",
]

fin_vars = [
    "RD_Intensity", "Firm_Size_Ln", "ROA", "Leverage", "Market_Cap_EUR",
]

all_vars = board_vars + fin_vars

stats = df[all_vars].describe().T
stats["missing"] = df[all_vars].isnull().sum()
stats["median"] = df[all_vars].median()
stats = stats[["count", "missing", "mean", "median", "std", "min", "25%", "75%", "max"]]
print(f"\n{stats.to_string()}")

# ============================================================
# 3. DESCRIPTIVES PER INSTITUTIONAL CONTEXT (LME vs CME)
# ============================================================
print("\n" + "=" * 70)
print("3. DESCRIPTIVES - LME (US) vs CME (EU)")
print("=" * 70)

for var in all_vars:
    us_vals = us[var].dropna()
    eu_vals = eu[var].dropna()
    print(f"\n  {var}:")
    print(f"    US (n={len(us_vals):3d}): mean={us_vals.mean():.3f}  std={us_vals.std():.3f}  median={us_vals.median():.3f}")
    print(f"    EU (n={len(eu_vals):3d}): mean={eu_vals.mean():.3f}  std={eu_vals.std():.3f}  median={eu_vals.median():.3f}")

# ============================================================
# 4. CEO DUALITY BREAKDOWN
# ============================================================
print("\n" + "=" * 70)
print("4. CEO DUALITY BREAKDOWN")
print("=" * 70)

print(f"\n  Full sample: {df['CEO_Duality'].sum()}/{len(df)} ({df['CEO_Duality'].mean()*100:.1f}%)")
print(f"  US (LME):   {us['CEO_Duality'].sum()}/{len(us)} ({us['CEO_Duality'].mean()*100:.1f}%)")
print(f"  EU (CME):   {eu['CEO_Duality'].sum()}/{len(eu)} ({eu['CEO_Duality'].mean()*100:.1f}%)")

print(f"\n  Per land:")
for loc in df["Loc"].value_counts().index:
    sub = df[df["Loc"] == loc]
    d = sub["CEO_Duality"].sum()
    name = country_names.get(loc, loc)
    print(f"    {name:20s}: {d:3d}/{len(sub):4d} ({sub['CEO_Duality'].mean()*100:.1f}%)")

# ============================================================
# 5. NORMALITEITSANALYSE
# ============================================================
# Relevant voor: OLS regressie (residuals), cluster analysis
# (extreme skewness vertekent afstandsmaten)
print("\n" + "=" * 70)
print("5. NORMALITEITSANALYSE")
print("=" * 70)

continuous_vars = [v for v in all_vars if v not in ["CEO_Duality"]]
print(f"\n  {'Variabele':25s} {'Skewness':>10s} {'Kurtosis':>10s} {'Shapiro W':>10s} {'Shapiro p':>10s} {'Oordeel':>12s}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

for var in continuous_vars:
    vals = df[var].dropna()
    skew = vals.skew()
    kurt = vals.kurtosis()
    # Shapiro-Wilk: max 5000 observaties
    if len(vals) > 5000:
        sw_stat, sw_p = sp_stats.shapiro(vals.sample(5000, random_state=42))
    else:
        sw_stat, sw_p = sp_stats.shapiro(vals)
    
    # Oordeel: |skew| > 2 of |kurt| > 7 = problematisch (Curran et al., 1996)
    if abs(skew) > 2 or abs(kurt) > 7:
        oordeel = "PROBLEEM"
    elif abs(skew) > 1:
        oordeel = "matig"
    else:
        oordeel = "ok"
    
    print(f"  {var:25s} {skew:10.3f} {kurt:10.3f} {sw_stat:10.4f} {sw_p:10.4f} {oordeel:>12s}")

print(f"\n  Drempels: |skewness| > 2 of |kurtosis| > 7 = problematisch (Curran et al., 1996)")
print(f"  Shapiro-Wilk: p < 0.05 = niet-normaal verdeeld")

# ============================================================
# 6. FORMELE TESTS LME vs CME VERSCHILLEN
# ============================================================
# Valideert de institutionele vergelijking: zijn er daadwerkelijk
# significante verschillen tussen de twee groepen?
print("\n" + "=" * 70)
print("6. FORMELE TESTS LME vs CME VERSCHILLEN")
print("=" * 70)

print(f"\n  {'Variabele':25s} {'t-stat':>8s} {'t-p':>8s} {'MW-U':>12s} {'MW-p':>8s} {'Cohen d':>8s} {'Sig':>6s}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*6}")

for var in continuous_vars:
    us_vals = us[var].dropna()
    eu_vals = eu[var].dropna()
    
    if len(us_vals) < 2 or len(eu_vals) < 2:
        continue
    
    # Welch's t-test (ongelijke varianties)
    t_stat, t_p = sp_stats.ttest_ind(us_vals, eu_vals, equal_var=False)
    
    # Mann-Whitney U (non-parametrisch alternatief)
    mw_stat, mw_p = sp_stats.mannwhitneyu(us_vals, eu_vals, alternative="two-sided")
    
    # Cohen's d (effectgrootte)
    pooled_std = np.sqrt(
        ((len(us_vals) - 1) * us_vals.std()**2 + (len(eu_vals) - 1) * eu_vals.std()**2)
        / (len(us_vals) + len(eu_vals) - 2)
    )
    cohens_d = (us_vals.mean() - eu_vals.mean()) / pooled_std if pooled_std > 0 else 0
    
    sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else ""
    
    print(f"  {var:25s} {t_stat:8.3f} {t_p:8.4f} {mw_stat:12.0f} {mw_p:8.4f} {cohens_d:8.3f} {sig:>6s}")

# Chi-square voor CEO Duality (categorisch)
contingency = pd.crosstab(df["Institutional_Context"], df["CEO_Duality"])
chi2, chi_p, dof, expected = sp_stats.chi2_contingency(contingency)
sig = "***" if chi_p < 0.001 else "**" if chi_p < 0.01 else "*" if chi_p < 0.05 else ""
print(f"  {'CEO_Duality (chi2)':25s} {chi2:8.3f} {chi_p:8.4f} {'':12s} {'':8s} {'':8s} {sig:>6s}")

print(f"\n  * p<0.05  ** p<0.01  *** p<0.001")
print(f"  Cohen's d: 0.2=klein, 0.5=medium, 0.8=groot")

# ============================================================
# 7. CORRELATIEMATRIX + MULTICOLLINEARITEIT (VIF)
# ============================================================
print("\n" + "=" * 70)
print("7. CORRELATIEMATRIX (Pearson)")
print("=" * 70)

corr_vars = board_vars + ["RD_Intensity"]
corr = df[corr_vars].corr()
print(f"\n{corr.round(3).to_string()}")

# Hoge correlaties flaggen (|r| > 0.5)
print(f"\n  Hoge correlaties (|r| > 0.5):")
flagged = False
for i in range(len(corr_vars)):
    for j in range(i+1, len(corr_vars)):
        r = corr.iloc[i, j]
        if abs(r) > 0.5:
            print(f"    {corr_vars[i]} x {corr_vars[j]}: r = {r:.3f}")
            flagged = True
if not flagged:
    print(f"    Geen correlaties > 0.5 gevonden")

# VIF voor clustering variabelen
print(f"\n  VARIANCE INFLATION FACTORS (clustering variabelen):")
print(f"  VIF > 5 = problematisch, VIF > 10 = ernstig")
cluster_vars = ["Board_Independence", "Gender_Diversity", "Avg_Tenure",
                "CEO_Duality", "Internationality_Ratio", "Board_Size"]
vif_df = df[cluster_vars].dropna()
vif_matrix = sm.add_constant(vif_df)
for i, var in enumerate(cluster_vars):
    vif = variance_inflation_factor(vif_matrix.values, i + 1)
    flag = " <-- PROBLEEM" if vif > 5 else ""
    print(f"    {var:25s}: VIF = {vif:.2f}{flag}")

# VIF voor regressie (controls + archetype dummies komen later)
print(f"\n  VIF (regressie controls):")
reg_controls = ["Firm_Size_Ln", "ROA", "Leverage"]
reg_df = df[reg_controls].dropna()
reg_matrix = sm.add_constant(reg_df)
for i, var in enumerate(reg_controls):
    vif = variance_inflation_factor(reg_matrix.values, i + 1)
    flag = " <-- PROBLEEM" if vif > 5 else ""
    print(f"    {var:25s}: VIF = {vif:.2f}{flag}")

# ============================================================
# 8. HETEROSCEDASTICITEIT TEST
# ============================================================
# Breusch-Pagan test op een voorlopig OLS model:
# RD_Intensity ~ Board variabelen + Controls
print("\n" + "=" * 70)
print("8. HETEROSCEDASTICITEIT (Breusch-Pagan)")
print("=" * 70)

reg_vars = cluster_vars + ["Firm_Size_Ln", "ROA", "Leverage", "Institutional_Context"]
reg_data = df[reg_vars + ["RD_Intensity"]].dropna()

X = sm.add_constant(reg_data[reg_vars])
y = reg_data["RD_Intensity"]
model = sm.OLS(y, X).fit()
bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(model.resid, X)

print(f"\n  Breusch-Pagan LM stat: {bp_stat:.3f}")
print(f"  Breusch-Pagan p-value: {bp_p:.4f}")
print(f"  F-stat: {bp_f:.3f}, F p-value: {bp_fp:.4f}")
if bp_p < 0.05:
    print(f"  RESULTAAT: Heteroscedasticiteit gedetecteerd (p < 0.05)")
    print(f"  ACTIE: Gebruik robust standard errors (HC1) in OLS regressie")
else:
    print(f"  RESULTAAT: Geen significant bewijs voor heteroscedasticiteit")

# Test ook met log-getransformeerde DV
print(f"\n  Hertest met ln(RD_Intensity + 0.001):")
y_log = np.log(reg_data["RD_Intensity"] + 0.001)
model_log = sm.OLS(y_log, X).fit()
bp_stat2, bp_p2, bp_f2, bp_fp2 = het_breuschpagan(model_log.resid, X)
print(f"  Breusch-Pagan LM stat: {bp_stat2:.3f}")
print(f"  Breusch-Pagan p-value: {bp_p2:.4f}")
if bp_p2 < 0.05:
    print(f"  RESULTAAT: Nog steeds heteroscedasticiteit -> robust SE nodig")
else:
    print(f"  RESULTAAT: Log-transformatie lost heteroscedasticiteit op")

# ============================================================
# 9. MISSING DATA ANALYSE
# ============================================================
print("\n" + "=" * 70)
print("9. MISSING DATA ANALYSE")
print("=" * 70)

for var in all_vars:
    m = df[var].isnull().sum()
    if m > 0:
        m_us = us[var].isnull().sum()
        m_eu = eu[var].isnull().sum()
        print(f"  {var}: {m} missing (US: {m_us}, EU: {m_eu})")

total_missing = df[all_vars].isnull().any(axis=1).sum()
print(f"\n  Rijen met minstens 1 missing: {total_missing}/{len(df)}")
print(f"  Complete cases: {len(df) - total_missing}/{len(df)}")

# Little's MCAR-achtige check: zijn missings random verdeeld?
# Vergelijk kenmerken van firms met/zonder missing Internationality_Ratio
print(f"\n  MCAR indicatie (Internationality_Ratio):")
has_data = df[df["Internationality_Ratio"].notna()]
missing_data = df[df["Internationality_Ratio"].isna()]
test_vars = ["Board_Size", "Firm_Size_Ln", "RD_Intensity", "Market_Cap_EUR"]
for tv in test_vars:
    v1 = has_data[tv].dropna()
    v2 = missing_data[tv].dropna()
    if len(v2) >= 2:
        t, p = sp_stats.ttest_ind(v1, v2, equal_var=False)
        sig = "*" if p < 0.05 else ""
        print(f"    {tv:25s}: mean(has)={v1.mean():.3f}  mean(miss)={v2.mean():.3f}  p={p:.4f} {sig}")

# ============================================================
# 10. EXTREME WAARDEN CHECK
# ============================================================
print("\n" + "=" * 70)
print("10. EXTREME WAARDEN (potentiele outliers)")
print("=" * 70)

outlier_vars = ["RD_Intensity", "Avg_Tenure", "Board_Independence",
                "Gender_Diversity", "Internationality_Ratio", "Board_Busyness",
                "ROA", "Leverage", "Firm_Size_Ln", "Board_Size", "Age_Diversity"]

print(f"\n  {'Variabele':25s} {'P1':>8s} {'P99':>8s} {'<P1':>5s} {'>P99':>5s} {'IQR outliers':>13s}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*13}")

for var in outlier_vars:
    vals = df[var].dropna()
    p1 = vals.quantile(0.01)
    p99 = vals.quantile(0.99)
    q1 = vals.quantile(0.25)
    q3 = vals.quantile(0.75)
    iqr = q3 - q1
    iqr_outliers = ((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum()
    below = (vals < p1).sum()
    above = (vals > p99).sum()
    print(f"  {var:25s} {p1:8.3f} {p99:8.3f} {below:5d} {above:5d} {iqr_outliers:13d}")

# R&D Intensity specifiek (we weten dat deze extreem scheef is)
print(f"\n  R&D Intensity detail:")
print(f"    Mean:   {df['RD_Intensity'].mean():.3f}")
print(f"    Median: {df['RD_Intensity'].median():.3f}")
print(f"    Ratio mean/median: {df['RD_Intensity'].mean()/df['RD_Intensity'].median():.1f}x")
print(f"    Firms met RD_Intensity > 1: {(df['RD_Intensity'] > 1).sum()}")
print(f"    Firms met RD_Intensity > 10: {(df['RD_Intensity'] > 10).sum()}")
print(f"    Firms met RD_Intensity > 100: {(df['RD_Intensity'] > 100).sum()}")

# ============================================================
# 11. CLUSTER ANALYSIS READINESS
# ============================================================
print("\n" + "=" * 70)
print("11. CLUSTER ANALYSIS READINESS")
print("=" * 70)

print(f"\n  Sample size check (Dolnicar et al., 2014):")
n_cluster_vars = 6
min_sample = n_cluster_vars * 70
print(f"    Clustering variabelen: {n_cluster_vars}")
print(f"    Minimum sample (70 per var): {min_sample}")
print(f"    Huidige sample: {len(df)}")
print(f"    Complete cases: {len(df) - total_missing}")
print(f"    Status: {'OK' if (len(df) - total_missing) >= min_sample else 'ONDER MINIMUM'}")

print(f"\n  Schaalverschillen (reden voor z-score standaardisatie):")
for var in cluster_vars:
    vals = df[var].dropna()
    print(f"    {var:25s}: range [{vals.min():.3f}, {vals.max():.3f}], std={vals.std():.3f}")

print(f"\n  Binaire variabelen in clustering:")
print(f"    CEO_Duality: {df['CEO_Duality'].mean()*100:.1f}% = 1 (voldoende variatie)")

print("\n" + "=" * 70)
print("KLAAR")
print("=" * 70)