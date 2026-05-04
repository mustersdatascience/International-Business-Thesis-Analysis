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

# Cluster dummies (referentie = Cluster 2: International Advisory Boards)
df["Cluster_0"] = (df["Cluster"] == 0).astype(int)  # Entrenched Boards
df["Cluster_1"] = (df["Cluster"] == 1).astype(int)  # Insider Boards
df["Cluster_3"] = (df["Cluster"] == 3).astype(int)  # Fresh Monitoring Boards
# Cluster 2 is referentiecategorie (niet als dummy opnemen)

print(f"  Cluster dummies aangemaakt (referentie: Cluster 2)")
print(f"    Cluster 0 (Entrenched):    {df['Cluster_0'].sum()} firms")
print(f"    Cluster 1 (Insider):       {df['Cluster_1'].sum()} firms")
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
context_vars = ["Sector_Tech", "Institutional_Context"]
interaction_vars = ["Cluster_0_x_US", "Cluster_1_x_US", "Cluster_3_x_US"]

# ============================================================
# STAP 5: MODEL SPECIFICATIES
# ============================================================
models = {
    "Model 1": control_vars,
    "Model 2": control_vars + cluster_vars,
    "Model 3": control_vars + cluster_vars + context_vars,
    "Model 4": control_vars + cluster_vars + context_vars + interaction_vars,
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
all_vars = ["const"] + cluster_vars + control_vars + context_vars + interaction_vars

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
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
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

print(f"\n  * p<0.05, ** p<0.01, *** p<0.001")
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

x_p3 = ["Internationality_Ratio", "Board_Busyness", "Firm_Size_Ln", "ROA", "Leverage", "Sector_Tech", "Institutional_Context"]
X_p3 = sm.add_constant(df[x_p3])
model_p3 = sm.OLS(df[y_var], X_p3).fit(cov_type='HC1')

print(f"\n  N = {int(model_p3.nobs)}, R² = {model_p3.rsquared:.4f}, Adj. R² = {model_p3.rsquared_adj:.4f}")
print(f"\n  {'Variabele':30s} {'β':>10s} {'SE':>10s} {'p':>10s} {'Sig':>6s}")
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
for var in x_p3:
    coef = model_p3.params[var]
    se = model_p3.bse[var]
    pval = model_p3.pvalues[var]
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
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
  Referentiecategorie: Cluster 2 (International Advisory Boards)
  
  Interpretatie coefficiënten:
""")

for cluster_var in cluster_vars:
    coef = model3.params[cluster_var]
    pval = model3.pvalues[cluster_var]
    se = model3.bse[cluster_var]
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    
    cluster_num = cluster_var.split("_")[1]
    cluster_names = {"0": "Entrenched", "1": "Insider", "3": "Fresh Monitoring"}
    cluster_name = cluster_names[cluster_num]
    
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else " (n.s.)"
    
    direction = "lager" if coef < 0 else "hoger"
    pct_effect = (np.exp(coef) - 1) * 100  # Voor log-level interpretatie
    
    print(f"  {cluster_var} ({cluster_name}):")
    print(f"    β = {coef:.3f}{sig}, SE = {se:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"    → {cluster_name} Boards hebben {abs(pct_effect):.1f}% {direction} R&D intensity")
    print(f"       dan International Advisory Boards (ceteris paribus)")
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
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"    {var:25s}: β = {coef:7.3f} {sig}")

print(f"\n  INTERACTIE-EFFECTEN:")
for var in interaction_vars:
    coef = model4.params[var]
    pval = model4.pvalues[var]
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else " (n.s.)"
    
    cluster_num = var.split("_")[1]
    cluster_names = {"0": "Entrenched", "1": "Insider", "3": "Fresh Monitoring"}
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

# Vergelijk Model 3 (restricted) vs Model 4 (unrestricted)
r2_restricted = results["Model 3"].rsquared
r2_unrestricted = results["Model 4"].rsquared
n = results["Model 4"].nobs
k_unrestricted = results["Model 4"].df_model + 1  # +1 voor constante
k_restricted = results["Model 3"].df_model + 1
q = k_unrestricted - k_restricted  # Aantal restricties

f_stat = ((r2_unrestricted - r2_restricted) / q) / ((1 - r2_unrestricted) / (n - k_unrestricted))
f_pval = 1 - stats.f.cdf(f_stat, q, n - k_unrestricted)

print(f"\n  H0: Alle interactie-termen = 0")
print(f"  H1: Minstens één interactie-term ≠ 0")
print(f"\n  F({int(q)}, {int(n - k_unrestricted)}) = {f_stat:.3f}")
print(f"  p-value = {f_pval:.4f}")
print(f"\n  Conclusie: {'VERWERP H0 - Interacties zijn gezamenlijk significant' if f_pval < 0.05 else 'KAN H0 NIET VERWERPEN - Interacties niet gezamenlijk significant'}")

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
  
  1. ARCHETYPE EFFECTEN (vs. International Advisory Boards):
""")

for cluster_var in cluster_vars:
    coef = model3.params[cluster_var]
    pval = model3.pvalues[cluster_var]
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "(n.s.)"
    cluster_num = cluster_var.split("_")[1]
    cluster_names = {"0": "Entrenched Boards", "1": "Insider Boards", "3": "Fresh Monitoring Boards"}
    print(f"     {cluster_names[cluster_num]:30s}: β = {coef:6.3f} {sig}")

print(f"""
  2. CONTROL VARIABELEN:
     Firm_Size_Ln:    β = {model3.params['Firm_Size_Ln']:6.3f} {'***' if model3.pvalues['Firm_Size_Ln'] < 0.001 else '**' if model3.pvalues['Firm_Size_Ln'] < 0.01 else '*' if model3.pvalues['Firm_Size_Ln'] < 0.05 else '(n.s.)'}
     ROA:             β = {model3.params['ROA']:6.3f} {'***' if model3.pvalues['ROA'] < 0.001 else '**' if model3.pvalues['ROA'] < 0.01 else '*' if model3.pvalues['ROA'] < 0.05 else '(n.s.)'}
     Leverage:        β = {model3.params['Leverage']:6.3f} {'***' if model3.pvalues['Leverage'] < 0.001 else '**' if model3.pvalues['Leverage'] < 0.01 else '*' if model3.pvalues['Leverage'] < 0.05 else '(n.s.)'}
  
  3. CONTEXT VARIABELEN:
     Sector (Tech):   β = {model3.params['Sector_Tech']:6.3f} {'***' if model3.pvalues['Sector_Tech'] < 0.001 else '**' if model3.pvalues['Sector_Tech'] < 0.01 else '*' if model3.pvalues['Sector_Tech'] < 0.05 else '(n.s.)'}
     US (vs EU):      β = {model3.params['Institutional_Context']:6.3f} {'***' if model3.pvalues['Institutional_Context'] < 0.001 else '**' if model3.pvalues['Institutional_Context'] < 0.01 else '*' if model3.pvalues['Institutional_Context'] < 0.05 else '(n.s.)'}
  
  4. MODEL FIT:
     R² = {model3.rsquared:.4f}
     Adj. R² = {model3.rsquared_adj:.4f}
     F-statistic = {model3.fvalue:.3f} (p < 0.001)
  
  5. INTERACTIE-EFFECTEN (Model 4):
     Gezamenlijke F-test: F = {f_stat:.3f}, p = {f_pval:.4f}
     {'Archetype effecten VERSCHILLEN tussen US en EU' if f_pval < 0.05 else 'Archetype effecten zijn VERGELIJKBAAR in US en EU'}
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
x_vars_robust1 = cluster_vars + control_vars + ["Institutional_Context"] + gics_cols

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
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "(n.s.)"
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
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "(n.s.)"
    print(f"    {cluster_var}: β = {coef:6.3f} {sig}")

print(f"\n  Market_Cap_Ln effect: β = {model_robust2.params['Market_Cap_Ln']:.3f} {'***' if model_robust2.pvalues['Market_Cap_Ln'] < 0.001 else '**' if model_robust2.pvalues['Market_Cap_Ln'] < 0.01 else '*' if model_robust2.pvalues['Market_Cap_Ln'] < 0.05 else '(n.s.)'}")
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
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "(n.s.)"
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
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "(n.s.)"
    print(f"    {cluster_var}: β = {coef:6.3f} {sig}")

print(f"\n  Vergelijking met Model 3:")
print(f"    Model 3 R²:        {model3.rsquared:.4f}")
print(f"    Robustness 4 R²:   {model_robust4.rsquared:.4f}")
print(f"    Cluster effecten:  {'CONSISTENT' if (model_robust4.pvalues['Cluster_1'] < 0.05) == (model3.pvalues['Cluster_1'] < 0.05) else 'INCONSISTENT'}")

# ============================================================
# STAP 20: ROBUSTNESS SAMENVATTING
# ============================================================
print(f"\n{'='*70}")
print("ROBUSTNESS CHECKS SAMENVATTING")
print(f"{'='*70}")

print(f"""
  {'Model':<30s} {'R²':>8s} {'Cluster_1 β':>12s} {'Sig':>6s}
  {'-'*30} {'-'*8} {'-'*12} {'-'*6}""")

models_summary = [
    ("Model 3 (Hoofdmodel)", model3),
    ("R1: GICS Subsector", model_robust1),
    ("R2: Market Cap", model_robust2),
    ("R3: Raw R&D Intensity", model_robust3),
    ("R4: Zonder outliers", model_robust4),
]

for name, model in models_summary:
    r2 = model.rsquared
    beta = model.params["Cluster_1"]
    pval = model.pvalues["Cluster_1"]
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {name:<30s} {r2:>8.4f} {beta:>12.3f} {sig:>6s}")

# Check consistentie
consistent_count = sum(1 for _, m in models_summary if m.pvalues["Cluster_1"] < 0.05)
print(f"\n  Cluster_1 (Insider Boards) significant in {consistent_count}/{len(models_summary)} modellen")
print(f"  Conclusie: {'ROBUUST - Resultaat houdt stand' if consistent_count >= 4 else 'FRAGIEL - Resultaat varieert per specificatie'}")

print(f"\n{'='*70}")
print("KLAAR")
print(f"{'='*70}")