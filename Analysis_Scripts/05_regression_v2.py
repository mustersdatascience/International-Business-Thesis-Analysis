import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
import os
os.makedirs("Output", exist_ok=True)

pd.set_option("display.float_format", lambda x: f"{x:.3f}")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)

# ============================================================
# HELPER FUNCTIES
# ============================================================
def sig_stars(pval):
    """Return significance markers using only †, *, ** (no *** per Jasper).
    Thresholds: † p < .10, * p < .05, ** p < .01"""
    if pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    elif pval < 0.10:
        return "†"
    else:
        return ""

def fmt_coef(coef, pval, se=None):
    """Format a coefficient with stars AND exact p-value (so Jasper sees real p)."""
    stars = sig_stars(pval)
    if se is not None:
        return f"β = {coef:+.3f}{stars} (SE={se:.3f}, p={pval:.4f})"
    return f"β = {coef:+.3f}{stars} (p={pval:.4f})"

# ============================================================
# STAP 1: INLADEN
# ============================================================
df = pd.read_csv("Data/05_Thesis_data_clustered.csv")
print(f"Ingeladen: {len(df)} firms")
print(f"  US: {(df['Institutional_Context'] == 1).sum()}")
print(f"  EU: {(df['Institutional_Context'] == 0).sum()}")

# ============================================================
# STAP 2: DUMMIES AANMAKEN
# ============================================================
print(f"\n{'='*70}")
print("DUMMIES AANMAKEN")
print(f"{'='*70}")

# Cluster dummies (referentie = Cluster 2: Internationally Connected Boards)
df["Cluster_0"] = (df["Cluster"] == 0).astype(int)  # Long-Tenured Boards
df["Cluster_1"] = (df["Cluster"] == 1).astype(int)  # Low-Diversity Boards
df["Cluster_3"] = (df["Cluster"] == 3).astype(int)  # Short-Tenured Domestic Boards
# Cluster 2 is referentiecategorie (niet als dummy opnemen)

print(f"  Cluster dummies aangemaakt (referentie: Cluster 2)")
print(f"    Cluster 0 (Long-Tenured):    {df['Cluster_0'].sum()} firms")
print(f"    Cluster 1 (Low-Diversity):       {df['Cluster_1'].sum()} firms")
print(f"    Cluster 2 (International): {(df['Cluster'] == 2).sum()} firms [REF]")
print(f"    Cluster 3 (Fresh):         {df['Cluster_3'].sum()} firms")

# Sector dummy (referentie = Healthcare)
df["Sector_Tech"] = (df["Sector"] == "Tech").astype(int)
print(f"\n  Sector dummy aangemaakt (referentie: Healthcare)")
print(f"    Tech:       {df['Sector_Tech'].sum()} firms")
print(f"    Healthcare: {(df['Sector_Tech'] == 0).sum()} firms [REF]")

# Institutional_Context is al binary (1=US/LME, 0=EU/CME)
print(f"\n  Institutional_Context al binary")
print(f"    US (LME): {(df['Institutional_Context'] == 1).sum()} firms")
print(f"    EU (CME): {(df['Institutional_Context'] == 0).sum()} firms [REF]")

# ============================================================
# STAP 3: INTERACTIE-TERMEN AANMAKEN
# ============================================================
# Cluster × Institutional_Context
# Test of archetype-effecten verschillen tussen US en EU

df["Cluster_0_x_US"] = df["Cluster_0"] * df["Institutional_Context"]
df["Cluster_1_x_US"] = df["Cluster_1"] * df["Institutional_Context"]
df["Cluster_3_x_US"] = df["Cluster_3"] * df["Institutional_Context"]

print(f"\n  Interactie-termen aangemaakt:")
print(f"    Cluster_0 × US: {df['Cluster_0_x_US'].sum()} firms")
print(f"    Cluster_1 × US: {df['Cluster_1_x_US'].sum()} firms")
print(f"    Cluster_3 × US: {df['Cluster_3_x_US'].sum()} firms")

# ============================================================
# STAP 4: VARIABELEN DEFINIËREN
# ============================================================
# Dependent variable
y_var = "RD_Intensity_Ln"

# Independent variables per model
cluster_vars = ["Cluster_0", "Cluster_1", "Cluster_3"]
control_vars = ["Firm_Size_Ln", "ROA", "Leverage"]
context_vars = ["Sector_Tech"]  # Sector blijft baseline control in Model 3
inst_vars = ["Institutional_Context"]  # Inst_Context alleen in Model 4 (per Kirsten #44)
interaction_vars = ["Cluster_0_x_US", "Cluster_1_x_US", "Cluster_3_x_US"]

# ============================================================
# STAP 5: MODEL SPECIFICATIES
# ============================================================
models = {
    "Model 1": control_vars,                                                                 # firm controls
    "Model 2": control_vars + context_vars,                                                  # + sector
    "Model 3": control_vars + context_vars + cluster_vars,                                   # + clusters = MAIN
    "Model 4": control_vars + context_vars + cluster_vars + inst_vars + interaction_vars,    # + inst context as moderator
}
# ============================================================
# STAP 6: OLS REGRESSIES UITVOEREN
# ============================================================
print(f"\n{'='*70}")
print("OLS REGRESSIE RESULTATEN")
print(f"{'='*70}")

results = {}
for model_name, x_vars in models.items():
    # Prepare data
    X = df[x_vars].copy()
    X = sm.add_constant(X)
    y = df[y_var]
    
    # Fit OLS with robust standard errors (HC1)
    model = sm.OLS(y, X).fit(cov_type='HC1')
    results[model_name] = model
    
    print(f"\n{'─'*70}")
    print(f"{model_name}: {' + '.join(x_vars[:3])}{'...' if len(x_vars) > 3 else ''}")
    print(f"{'─'*70}")
    print(f"  N = {int(model.nobs)}, R² = {model.rsquared:.4f}, Adj. R² = {model.rsquared_adj:.4f}")
    print(f"  F({int(model.df_model)}, {int(model.df_resid)}) = {model.fvalue:.3f}, p = {model.f_pvalue:.4f}")

# ============================================================
# STAP 7: COEFFICIENT TABEL
# ============================================================
print(f"\n{'='*70}")
print("COEFFICIENT OVERZICHT (alle modellen)")
print(f"{'='*70}")

# Verzamel alle variabelen
all_vars = ["const"] + cluster_vars + control_vars + context_vars + inst_vars + interaction_vars

# Header
print(f"\n{'Variabele':25s} {'Model 1':>14s} {'Model 2':>14s} {'Model 3':>14s} {'Model 4':>14s}")
print(f"{'-'*25} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")

for var in all_vars:
    row = f"{var:25s}"
    for model_name in models.keys():
        model = results[model_name]
        if var in model.params.index:
            coef = model.params[var]
            pval = model.pvalues[var]
            sig = sig_stars(pval)
            row += f" {coef:10.3f}{sig:>3s}"
        else:
            row += f" {'':>14s}"
    print(row)

# Model stats
print(f"{'-'*25} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
print(f"{'N':25s}", end="")
for model_name in models.keys():
    print(f" {int(results[model_name].nobs):>14d}", end="")
print()

print(f"{'R²':25s}", end="")
for model_name in models.keys():
    print(f" {results[model_name].rsquared:>14.4f}", end="")
print()

print(f"{'Adj. R²':25s}", end="")
for model_name in models.keys():
    print(f" {results[model_name].rsquared_adj:>14.4f}", end="")
print()

print(f"{'F-statistic':25s}", end="")
for model_name in models.keys():
    print(f" {results[model_name].fvalue:>14.3f}", end="")
print()

print(f"\n  † p<0.10, * p<0.05, ** p<0.01")
print(f"  Robust standard errors (HC1) gebruikt")

# ============================================================
# STAP 8: GEDETAILLEERDE OUTPUT MODEL 3 (HOOFDMODEL)
# ============================================================
print(f"\n{'='*70}")
print("GEDETAILLEERDE OUTPUT - MODEL 3 (HOOFDMODEL)")
print(f"{'='*70}")

model3 = results["Model 3"]
print(model3.summary())

# ============================================================
# STAP 9: GEDETAILLEERDE OUTPUT MODEL 4 (INTERACTIES)
# ============================================================
print(f"\n{'='*70}")
print("GEDETAILLEERDE OUTPUT - MODEL 4 (INTERACTIES)")
print(f"{'='*70}")

model4 = results["Model 4"]
print(model4.summary())

# ============================================================
# STAP 10: VIF CHECK VOOR MODEL 3
# ============================================================
print(f"\n{'='*70}")
print("MULTICOLLINEARITEIT CHECK (Model 3)")
print(f"{'='*70}")

X_m3 = df[cluster_vars + control_vars + context_vars].copy()
X_m3 = sm.add_constant(X_m3)

print(f"\n  {'Variabele':25s} {'VIF':>10s}")
print(f"  {'-'*25} {'-'*10}")
for i, var in enumerate(X_m3.columns):
    if var != "const":
        vif = variance_inflation_factor(X_m3.values, i)
        flag = " <-- HOOG" if vif > 5 else ""
        print(f"  {var:25s} {vif:10.2f}{flag}")

# ============================================================
# STAP 10b: CONTROLS-ONLY MODEL (voor incrementele R²)
# ============================================================
print(f"\n{'='*70}")
print("CONTROLS-ONLY MODEL (zonder archetypes)")
print(f"{'='*70}")

x_controls_only = control_vars + context_vars
X_co = sm.add_constant(df[x_controls_only])
model_controls = sm.OLS(df[y_var], X_co).fit(cov_type='HC1')

print(f"\n  Controls-only: R² = {model_controls.rsquared:.4f}, Adj. R² = {model_controls.rsquared_adj:.4f}")
print(f"  Model 3:       R² = {model3.rsquared:.4f}, Adj. R² = {model3.rsquared_adj:.4f}")
incremental_r2 = model3.rsquared - model_controls.rsquared
print(f"  Incrementele R² (archetypes): {incremental_r2:.4f}")
print(f"  F-test Model 3 vs Controls-only:")

from scipy import stats as sp_stats
df_num = model_controls.df_resid - model3.df_resid
df_den = model3.df_resid
f_incr = ((model_controls.ssr - model3.ssr) / df_num) / (model3.ssr / df_den)
p_incr = 1 - sp_stats.f.cdf(f_incr, df_num, df_den)
print(f"    F({df_num:.0f}, {df_den:.0f}) = {f_incr:.3f}, p = {p_incr:.4f}")

# ============================================================
# STAP 10c: P3 SUPPLEMENTARY TEST (continue variabelen)
# ============================================================
print(f"\n{'='*70}")
print("P3 SUPPLEMENTARY TEST: INTERNATIONALITY & BUSYNESS ALS CONTINUE VARIABELEN")
print(f"{'='*70}")

x_p3 = ["Internationality_Ratio", "Board_Busyness", "Firm_Size_Ln", "ROA", "Leverage", "Sector_Tech"]  # Inst_Context weggelaten, consistent met Model 3
X_p3 = sm.add_constant(df[x_p3])
model_p3 = sm.OLS(df[y_var], X_p3).fit(cov_type='HC1')

print(f"\n  N = {int(model_p3.nobs)}, R² = {model_p3.rsquared:.4f}, Adj. R² = {model_p3.rsquared_adj:.4f}")
print(f"\n  {'Variabele':30s} {'β':>10s} {'SE':>10s} {'p':>10s} {'Sig':>6s}")
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
for var in x_p3:
    coef = model_p3.params[var]
    se = model_p3.bse[var]
    pval = model_p3.pvalues[var]
    sig = sig_stars(pval)
    print(f"  {var:30s} {coef:10.3f} {se:10.3f} {pval:10.4f} {sig:>6s}")

# ============================================================
# STAP 10d: R&D INTENSITY PER GICS SUBSECTOR
# ============================================================
print(f"\n{'='*70}")
print("R&D INTENSITY PER GICS SUBSECTOR")
print(f"{'='*70}")

subsector_stats = df.groupby("Gcis_Sub").agg(
    n=("RD_Intensity_Ln", "count"),
    mean_rd_ln=("RD_Intensity_Ln", "mean"),
    sector=("Sector", "first"),
).sort_values("mean_rd_ln", ascending=False)

print(f"\n  {'GICS Sub':>10s} {'Sector':>12s} {'N':>5s} {'Mean Ln(R&D)':>14s}")
print(f"  {'-'*10} {'-'*12} {'-'*5} {'-'*14}")
for idx, row in subsector_stats.iterrows():
    print(f"  {idx:>10d} {row['sector']:>12s} {row['n']:>5d} {row['mean_rd_ln']:>14.3f}")

# ============================================================
# STAP 11: INTERPRETATIE CLUSTER EFFECTEN
# ============================================================
print(f"\n{'='*70}")
print("INTERPRETATIE CLUSTER EFFECTEN (Model 3)")
print(f"{'='*70}")

print(f"""
  Referentiecategorie: Cluster 2 (Internationally Connected Boards)
  
  Interpretatie coefficiënten:
""")

for cluster_var in cluster_vars:
    coef = model3.params[cluster_var]
    pval = model3.pvalues[cluster_var]
    se = model3.bse[cluster_var]
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    
    cluster_num = cluster_var.split("_")[1]
    cluster_names = {"0": "Long-Tenured", "1": "Low-Diversity", "3": "Short-Tenured Domestic"}
    cluster_name = cluster_names[cluster_num]
    
    sig = sig_stars(pval) or " (n.s.)"
    
    direction = "lager" if coef < 0 else "hoger"
    pct_effect = (np.exp(coef) - 1) * 100  # Voor log-level interpretatie
    
    print(f"  {cluster_var} ({cluster_name}):")
    print(f"    β = {coef:.3f}{sig}, SE = {se:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"    → {cluster_name} Boards hebben {abs(pct_effect):.1f}% {direction} R&D intensity")
    print(f"       dan Internationally Connected Boards (ceteris paribus)")
    print()

# ============================================================
# STAP 12: INTERPRETATIE INTERACTIE-EFFECTEN
# ============================================================
print(f"\n{'='*70}")
print("INTERPRETATIE INTERACTIE-EFFECTEN (Model 4)")
print(f"{'='*70}")

print(f"""
  Vraag: Werken board archetypes anders in US (LME) vs EU (CME)?
  
  Interpretatie interactie-termen:
""")

model4 = results["Model 4"]

# Main effects in Model 4
print(f"  MAIN EFFECTS (baseline = EU firms in Cluster 2):")
for var in cluster_vars + ["Institutional_Context"]:
    coef = model4.params[var]
    pval = model4.pvalues[var]
    sig = sig_stars(pval)
    print(f"    {var:25s}: β = {coef:7.3f} {sig}")

print(f"\n  INTERACTIE-EFFECTEN:")
for var in interaction_vars:
    coef = model4.params[var]
    pval = model4.pvalues[var]
    sig = sig_stars(pval) or " (n.s.)"
    
    cluster_num = var.split("_")[1]
    cluster_names = {"0": "Long-Tenured", "1": "Low-Diversity", "3": "Short-Tenured Domestic"}
    cluster_name = cluster_names[cluster_num]
    
    print(f"    {var:25s}: β = {coef:7.3f} {sig}")
    if pval < 0.05:
        direction = "sterker" if coef > 0 else "zwakker"
        print(f"      → Het {cluster_name} effect is {direction} in US dan in EU")
    else:
        print(f"      → Geen significant verschil tussen US en EU")
    print()

# ============================================================
# STAP 13: MODEL VERGELIJKING
# ============================================================
print(f"\n{'='*70}")
print("MODEL VERGELIJKING")
print(f"{'='*70}")

print(f"\n  {'Model':10s} {'R²':>10s} {'Adj. R²':>10s} {'AIC':>12s} {'BIC':>12s}")
print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")

for model_name, model in results.items():
    print(f"  {model_name:10s} {model.rsquared:10.4f} {model.rsquared_adj:10.4f} {model.aic:12.2f} {model.bic:12.2f}")

# Delta R² tussen modellen
print(f"\n  Incrementele R² bijdrage:")
print(f"    Model 1 → 2 (controls):     +{results['Model 2'].rsquared - results['Model 1'].rsquared:.4f}")
print(f"    Model 2 → 3 (context):      +{results['Model 3'].rsquared - results['Model 2'].rsquared:.4f}")
print(f"    Model 3 → 4 (interacties):  +{results['Model 4'].rsquared - results['Model 3'].rsquared:.4f}")

# ============================================================
# STAP 14: F-TEST VOOR INTERACTIE-EFFECTEN
# ============================================================
print(f"\n{'='*70}")
print("F-TEST: ZIJN INTERACTIE-EFFECTEN GEZAMENLIJK SIGNIFICANT?")
print(f"{'='*70}")

# Wald-test op alleen de 3 interactie-termen (niet de Institutional_Context main effect)
wald = results["Model 4"].f_test("Cluster_0_x_US = 0, Cluster_1_x_US = 0, Cluster_3_x_US = 0")
f_stat = float(wald.statistic)
f_pval = float(wald.pvalue)
df_num = int(wald.df_num)
df_den = int(wald.df_denom)

print(f"\n  H0: Alle interactie-termen = 0")
print(f"  H1: Minstens één interactie-term ≠ 0")
print(f"\n  F({df_num}, {df_den}) = {f_stat:.3f}")
print(f"  p-value = {f_pval:.4f}")
print(f"\n  Conclusie: {'VERWERP H0 - Interacties zijn gezamenlijk significant' if f_pval < 0.10 else 'KAN H0 NIET VERWERPEN - Interacties niet gezamenlijk significant'}")

# ============================================================
# STAP 15: SAMENVATTING VOOR THESIS
# ============================================================
print(f"\n{'='*70}")
print("SAMENVATTING VOOR THESIS")
print(f"{'='*70}")

print(f"""
  REGRESSIE RESULTATEN
  
  Dependent Variable: Ln(R&D Intensity)
  N = {int(model3.nobs)} firms
  
  HOOFDBEVINDINGEN (Model 3):
  
  1. ARCHETYPE EFFECTEN (vs. Internationally Connected Boards):
""")

for cluster_var in cluster_vars:
    coef = model3.params[cluster_var]
    pval = model3.pvalues[cluster_var]
    sig = sig_stars(pval) or "(n.s.)"
    cluster_num = cluster_var.split("_")[1]
    cluster_names = {"0": "Long-Tenured Boards", "1": "Low-Diversity Boards", "3": "Short-Tenured Domestic Boards"}
    print(f"     {cluster_names[cluster_num]:30s}: β = {coef:6.3f} {sig}")

print(f"""
  2. CONTROL VARIABELEN:
     Firm_Size_Ln:    β = {model3.params['Firm_Size_Ln']:6.3f} {sig_stars(model3.pvalues['Firm_Size_Ln']) or '(n.s.)'}
     ROA:             β = {model3.params['ROA']:6.3f} {sig_stars(model3.pvalues['ROA']) or '(n.s.)'}
     Leverage:        β = {model3.params['Leverage']:6.3f} {sig_stars(model3.pvalues['Leverage']) or '(n.s.)'}
  
  3. CONTEXT VARIABELEN:
     Sector (Tech):   β = {model3.params['Sector_Tech']:6.3f} {sig_stars(model3.pvalues['Sector_Tech']) or '(n.s.)'}
     [Institutional Context verschoven naar Model 4 als pure moderator, per Kirsten #44]
  
  4. MODEL FIT:
     R² = {model3.rsquared:.4f}
     Adj. R² = {model3.rsquared_adj:.4f}
     F-statistic = {model3.fvalue:.3f} (p = {model3.f_pvalue:.6f})
  
  5. INSTITUTIONAL CONTEXT (Model 4):
     US main effect: β = {model4.params['Institutional_Context']:6.3f} {sig_stars(model4.pvalues['Institutional_Context']) or '(n.s.)'}
     Joint F-test interacties: F({df_num}, {df_den}) = {f_stat:.3f}, p = {f_pval:.4f}
     {'Archetype effecten VERSCHILLEN tussen US en EU (P2 supported)' if f_pval < 0.10 else 'Geen significante moderatie'}
""")

# ============================================================
# STAP 16: ROBUSTNESS CHECK 1 - GICS SUBSECTOR DUMMIES
# ============================================================
print(f"\n{'='*70}")
print("ROBUSTNESS CHECK 1: GICS SUBSECTOR DUMMIES")
print(f"{'='*70}")

# Maak GICS subsector dummies (convert to string first, then to int)
gics_dummies = pd.get_dummies(df["Gcis_Sub"].astype(str), prefix="GICS", drop_first=True).astype(int)
df_robust1 = pd.concat([df, gics_dummies], axis=1)

# Model met GICS dummies ipv Sector dummy
gics_cols = [col for col in gics_dummies.columns]
x_vars_robust1 = cluster_vars + control_vars + gics_cols  # GICS dummies vervangen Sector; geen Inst_Context (consistent met Model 3)

X_r1 = df_robust1[x_vars_robust1].copy().astype(float)
X_r1 = sm.add_constant(X_r1)
y_r1 = df_robust1[y_var].astype(float)

model_robust1 = sm.OLS(y_r1, X_r1).fit(cov_type='HC1')

print(f"\n  Model met GICS subsector dummies (ipv Sector dummy)")
print(f"  N = {int(model_robust1.nobs)}, R² = {model_robust1.rsquared:.4f}, Adj. R² = {model_robust1.rsquared_adj:.4f}")
print(f"\n  Cluster effecten:")
for cluster_var in cluster_vars:
    coef = model_robust1.params[cluster_var]
    pval = model_robust1.pvalues[cluster_var]
    sig = sig_stars(pval) or "(n.s.)"
    print(f"    {cluster_var}: β = {coef:6.3f} {sig}")

print(f"\n  Vergelijking met Model 3:")
print(f"    Model 3 R²:        {model3.rsquared:.4f}")
print(f"    Robustness 1 R²:   {model_robust1.rsquared:.4f}")
print(f"    Cluster effecten:  {'CONSISTENT' if (model_robust1.pvalues['Cluster_1'] < 0.05) == (model3.pvalues['Cluster_1'] < 0.05) else 'INCONSISTENT'}")

# ============================================================
# STAP 17: ROBUSTNESS CHECK 2 - MARKET CAP ipv FIRM SIZE
# ============================================================
print(f"\n{'='*70}")
print("ROBUSTNESS CHECK 2: MARKET CAP (Ln) ipv FIRM SIZE")
print(f"{'='*70}")

# Log-transform market cap
df["Market_Cap_Ln"] = np.log(df["Market_Cap_EUR"])

x_vars_robust2 = cluster_vars + ["Market_Cap_Ln", "ROA", "Leverage"] + context_vars

X_r2 = df[x_vars_robust2].copy()
X_r2 = sm.add_constant(X_r2)
y_r2 = df[y_var]

model_robust2 = sm.OLS(y_r2, X_r2).fit(cov_type='HC1')

print(f"\n  Model met Market_Cap_Ln ipv Firm_Size_Ln")
print(f"  N = {int(model_robust2.nobs)}, R² = {model_robust2.rsquared:.4f}, Adj. R² = {model_robust2.rsquared_adj:.4f}")
print(f"\n  Cluster effecten:")
for cluster_var in cluster_vars:
    coef = model_robust2.params[cluster_var]
    pval = model_robust2.pvalues[cluster_var]
    sig = sig_stars(pval) or "(n.s.)"
    print(f"    {cluster_var}: β = {coef:6.3f} {sig}")

print(f"\n  Market_Cap_Ln effect: β = {model_robust2.params['Market_Cap_Ln']:.3f} {sig_stars(model_robust2.pvalues['Market_Cap_Ln']) or '(n.s.)'}")
print(f"\n  Vergelijking met Model 3:")
print(f"    Model 3 R²:        {model3.rsquared:.4f}")
print(f"    Robustness 2 R²:   {model_robust2.rsquared:.4f}")
print(f"    Cluster effecten:  {'CONSISTENT' if (model_robust2.pvalues['Cluster_1'] < 0.05) == (model3.pvalues['Cluster_1'] < 0.05) else 'INCONSISTENT'}")

# ============================================================
# STAP 18: ROBUSTNESS CHECK 3 - RAW R&D INTENSITY (niet Ln)
# ============================================================
print(f"\n{'='*70}")
print("ROBUSTNESS CHECK 3: RAW R&D INTENSITY (niet log-transformed)")
print(f"{'='*70}")

x_vars_robust3 = cluster_vars + control_vars + context_vars

X_r3 = df[x_vars_robust3].copy()
X_r3 = sm.add_constant(X_r3)
y_r3 = df["RD_Intensity"]  # Raw, niet Ln

model_robust3 = sm.OLS(y_r3, X_r3).fit(cov_type='HC1')

print(f"\n  Model met raw R&D Intensity als DV")
print(f"  N = {int(model_robust3.nobs)}, R² = {model_robust3.rsquared:.4f}, Adj. R² = {model_robust3.rsquared_adj:.4f}")
print(f"\n  Cluster effecten:")
for cluster_var in cluster_vars:
    coef = model_robust3.params[cluster_var]
    pval = model_robust3.pvalues[cluster_var]
    sig = sig_stars(pval) or "(n.s.)"
    print(f"    {cluster_var}: β = {coef:6.3f} {sig}")

print(f"\n  WAARSCHUWING: Raw R&D Intensity is extreem scheef verdeeld.")
print(f"  Dit model is minder betrouwbaar dan Model 3 (Ln-transformed).")
print(f"\n  Vergelijking met Model 3:")
print(f"    Model 3 R²:        {model3.rsquared:.4f}")
print(f"    Robustness 3 R²:   {model_robust3.rsquared:.4f}")

# ============================================================
# STAP 19: ROBUSTNESS CHECK 4 - ZONDER OUTLIERS
# ============================================================
print(f"\n{'='*70}")
print("ROBUSTNESS CHECK 4: ZONDER EXTREME R&D INTENSITY OUTLIERS")
print(f"{'='*70}")

# Verwijder firms met R&D Intensity > P99
p99 = df["RD_Intensity"].quantile(0.99)
df_no_outliers = df[df["RD_Intensity"] <= p99].copy()

X_r4 = df_no_outliers[cluster_vars + control_vars + context_vars].copy()
X_r4 = sm.add_constant(X_r4)
y_r4 = df_no_outliers[y_var]

model_robust4 = sm.OLS(y_r4, X_r4).fit(cov_type='HC1')

print(f"\n  Model zonder R&D Intensity outliers (> P99)")
print(f"  Verwijderd: {len(df) - len(df_no_outliers)} firms")
print(f"  N = {int(model_robust4.nobs)}, R² = {model_robust4.rsquared:.4f}, Adj. R² = {model_robust4.rsquared_adj:.4f}")
print(f"\n  Cluster effecten:")
for cluster_var in cluster_vars:
    coef = model_robust4.params[cluster_var]
    pval = model_robust4.pvalues[cluster_var]
    sig = sig_stars(pval) or "(n.s.)"
    print(f"    {cluster_var}: β = {coef:6.3f} {sig}")

print(f"\n  Vergelijking met Model 3:")
print(f"    Model 3 R²:        {model3.rsquared:.4f}")
print(f"    Robustness 4 R²:   {model_robust4.rsquared:.4f}")
print(f"    Cluster effecten:  {'CONSISTENT' if (model_robust4.pvalues['Cluster_1'] < 0.05) == (model3.pvalues['Cluster_1'] < 0.05) else 'INCONSISTENT'}")

# ============================================================
# STAP 19b: ROBUSTNESS CHECK 5 - REFERENCE ROTATION (Cluster 3 als ref)
# Per Kirsten comment #40: Cluster 3 heeft hoogste mean R&D, dus
# conventioneel de juiste reference. Test of bevindingen reference-onafhankelijk zijn.
# ============================================================
print(f"\n{'='*70}")
print("ROBUSTNESS CHECK 5: REFERENCE ROTATION (Cluster 3 = highest mean R&D)")
print(f"{'='*70}")

# Maak Cluster_2 als dummy (was reference); Cluster_3 wordt nu reference (geen dummy)
df["Cluster_2_dummy"] = (df["Cluster"] == 2).astype(int)
cluster_vars_r5 = ["Cluster_0", "Cluster_1", "Cluster_2_dummy"]
x_vars_r5 = control_vars + ["Sector_Tech"] + cluster_vars_r5

X_r5 = sm.add_constant(df[x_vars_r5])
y_r5 = df[y_var]
model_robust5 = sm.OLS(y_r5, X_r5).fit(cov_type='HC1')

print(f"\n  Model met Cluster 3 (Short-Tenured Domestic) als reference")
print(f"  N = {int(model_robust5.nobs)}, R² = {model_robust5.rsquared:.4f}, Adj. R² = {model_robust5.rsquared_adj:.4f}")
print(f"\n  Cluster effecten (vs. Cluster 3 = highest mean R&D):")
cluster_labels_r5 = {
    "Cluster_0": "Long-Tenured Boards",
    "Cluster_1": "Low-Diversity Boards",
    "Cluster_2_dummy": "Internationally Connected Boards"
}
for cv in cluster_vars_r5:
    coef = model_robust5.params[cv]
    pval = model_robust5.pvalues[cv]
    se = model_robust5.bse[cv]
    sig = sig_stars(pval) or "(n.s.)"
    print(f"    {cluster_labels_r5[cv]:35s}: β = {coef:+.3f} {sig:>7s}  (SE={se:.3f}, p={pval:.4f})")

print(f"\n  Vergelijking met Model 3 (Cluster 2 als ref):")
print(f"    Cluster 1 effect Model 3: β = {model3.params['Cluster_1']:+.3f}, p = {model3.pvalues['Cluster_1']:.4f}")
print(f"    Cluster 1 effect R5:      β = {model_robust5.params['Cluster_1']:+.3f}, p = {model_robust5.pvalues['Cluster_1']:.4f}")
print(f"    → Core finding (Cluster 1 lager) blijft {'OVEREIND' if model_robust5.pvalues['Cluster_1'] < 0.05 else 'NIET overeind'} onder rotation")

# ============================================================
# STAP 20: ROBUSTNESS SAMENVATTING (alle 3 clusters per spec - per #51)
# ============================================================
print(f"\n{'='*70}")
print("ROBUSTNESS CHECKS SAMENVATTING (alle cluster coefficients)")
print(f"{'='*70}")

models_summary = [
    ("Model 3 (Hoofdmodel)", model3, "Cluster_2"),
    ("R1: GICS Subsector",   model_robust1, "Cluster_2"),
    ("R2: Market Cap",       model_robust2, "Cluster_2"),
    ("R3: Raw R&D Intensity",model_robust3, "Cluster_2"),
    ("R4: Zonder outliers",  model_robust4, "Cluster_2"),
    ("R5: Cluster 3 als ref",model_robust5, "Cluster_3"),
]

print(f"\n  {'Model':<24s} {'Ref':<10s} {'R²':>6s} | {'C0 β':>8s} {'sig':>4s} {'p':>6s} | {'C1 β':>8s} {'sig':>4s} {'p':>6s} | {'Other β':>8s} {'sig':>4s} {'p':>6s}")
print(f"  {'-'*24} {'-'*10} {'-'*6} | {'-'*8} {'-'*4} {'-'*6} | {'-'*8} {'-'*4} {'-'*6} | {'-'*8} {'-'*4} {'-'*6}")

for name, model, ref in models_summary:
    r2 = model.rsquared
    row = f"  {name:<24s} C{ref[-1]:1s} ref    {r2:>6.3f}"
    # Cluster 0 (always present)
    if "Cluster_0" in model.params.index:
        b, p = model.params["Cluster_0"], model.pvalues["Cluster_0"]
        row += f" | {b:>+8.3f} {sig_stars(p):>4s} {p:>6.3f}"
    else:
        row += f" | {'n/a':>20s}"
    # Cluster 1 (always present, this is the key one)
    if "Cluster_1" in model.params.index:
        b, p = model.params["Cluster_1"], model.pvalues["Cluster_1"]
        row += f" | {b:>+8.3f} {sig_stars(p):>4s} {p:>6.3f}"
    else:
        row += f" | {'n/a':>20s}"
    # "Other" = Cluster 3 in main spec, Cluster 2_dummy in R5
    if "Cluster_3" in model.params.index:
        b, p = model.params["Cluster_3"], model.pvalues["Cluster_3"]
        row += f" | C3:{b:>+5.3f} {sig_stars(p):>4s} {p:>6.3f}"
    elif "Cluster_2_dummy" in model.params.index:
        b, p = model.params["Cluster_2_dummy"], model.pvalues["Cluster_2_dummy"]
        row += f" | C2:{b:>+5.3f} {sig_stars(p):>4s} {p:>6.3f}"
    else:
        row += f" | {'n/a':>20s}"
    print(row)

print(f"\n  Significantie: † p<.10, * p<.05, ** p<.01")

# Robuustheid van Cluster 1 effect specifiek
c1_sig = sum(1 for _, m, _ in models_summary if m.pvalues["Cluster_1"] < 0.05)
c1_marg = sum(1 for _, m, _ in models_summary if m.pvalues["Cluster_1"] < 0.10)
print(f"\n  Cluster 1 (Low-Diversity) effect:")
print(f"    Significant op p<.05 in {c1_sig}/{len(models_summary)} specificaties")
print(f"    Marginaal+ op p<.10 in   {c1_marg}/{len(models_summary)} specificaties")
print(f"    Conclusie: {'ROBUUST' if c1_sig >= 4 else 'GEMENGD'}")

print(f"\n{'='*70}")
print("KLAAR")
print(f"{'='*70}")