import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.float_format", lambda x: f"{x:.3f}")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)

# ============================================================
# STAP 1: INLADEN
# ============================================================
df = pd.read_csv("Data/04_Thesis_data_standardized.csv")
print(f"Ingeladen: {len(df)} firms")

# Clustering variabelen (z-scores)
cluster_vars_z = [
    "Board_Independence_z",
    "Gender_Diversity_z",
    "Avg_Tenure_z",
    "Internationality_Ratio_z",
    "Board_Size_z",
    "Board_Busyness_z",
]

# Originele variabelen (voor interpretatie)
cluster_vars = [
    "Board_Independence",
    "Gender_Diversity",
    "Avg_Tenure",
    "Internationality_Ratio",
    "Board_Size",
    "Board_Busyness",
]

X = df[cluster_vars_z].values

# ============================================================
# STAP 2: K-MEANS CLUSTERING (k=4)
# ============================================================
print(f"\n{'='*70}")
print("K-MEANS CLUSTERING (k=4)")
print(f"{'='*70}")

K = 4
RANDOM_SEED = 42

# Run K-Means met meerdere initialisaties voor stabiliteit
kmeans = KMeans(
    n_clusters=K,
    random_state=RANDOM_SEED,
    n_init=25,      # 25 verschillende startpunten
    max_iter=300,   # Max iteraties per run
)

df["Cluster"] = kmeans.fit_predict(X)

# Silhouette score
sil_score = silhouette_score(X, df["Cluster"])
print(f"\n  Silhouette score: {sil_score:.4f}")

# Cluster groottes
print(f"\n  Cluster groottes:")
for c in range(K):
    n = (df["Cluster"] == c).sum()
    pct = n / len(df) * 100
    print(f"    Cluster {c}: {n:4d} firms ({pct:5.1f}%)")

lda = LinearDiscriminantAnalysis()
lda_accuracy = lda.fit(X, df["Cluster"]).score(X, df["Cluster"])
print(f"\n  LDA classificatie accuracy: {lda_accuracy:.4f} ({lda_accuracy*100:.1f}%)")
print(f"  (vs. {1/K*100:.0f}% random chance)")

# Percentage negatieve silhouette scores
sil_samples = silhouette_samples(X, df["Cluster"])
neg_sil_pct = (sil_samples < 0).mean() * 100
print(f"\n  Negatieve silhouette scores: {neg_sil_pct:.1f}% van observaties")

# ANOVA per clustering variabele (discriminerend vermogen)
print(f"\n  Discriminerend vermogen per variabele:")
print(f"  {'Variabele':30s} {'F-stat':>10s} {'p-value':>10s} {'Eta²':>8s}")
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8}")
for var in cluster_vars_z:
    groups_var = [df[df["Cluster"] == c][var].values for c in range(K)]
    f_val, p_val = stats.f_oneway(*groups_var)
    ss_b = sum(len(g) * (np.mean(g) - df[var].mean())**2 for g in groups_var)
    ss_t = sum((df[var] - df[var].mean())**2)
    eta2 = ss_b / ss_t
    print(f"  {var:30s} {f_val:10.2f} {p_val:10.4f} {eta2:8.4f}")

lda_cv = cross_val_score(LinearDiscriminantAnalysis(), X, df["Cluster"], cv=10)
print(f"\n  LDA 10-fold CV accuracy: {lda_cv.mean():.4f} ({lda_cv.mean()*100:.1f}%)")
print(f"  LDA CV std: {lda_cv.std():.4f}")

# ============================================================
# STAP 3: CLUSTER PROFIELEN (Z-SCORES)
# ============================================================
print(f"\n{'='*70}")
print("CLUSTER PROFIELEN (Z-Scores)")
print(f"{'='*70}")

# Gemiddelde z-scores per cluster
profile_z = df.groupby("Cluster")[cluster_vars_z].mean()
print(f"\n{profile_z.round(3).to_string()}")

# Interpretatie helper
print(f"\n  Interpretatie z-scores:")
print(f"    > +0.5: boven gemiddeld")
print(f"    < -0.5: onder gemiddeld")
print(f"    -0.5 tot +0.5: rond gemiddeld")

# ============================================================
# STAP 4: CLUSTER PROFIELEN (ORIGINELE WAARDEN)
# ============================================================
print(f"\n{'='*70}")
print("CLUSTER PROFIELEN (Originele waarden)")
print(f"{'='*70}")

# Gemiddelde originele waarden per cluster
profile_orig = df.groupby("Cluster")[cluster_vars].mean()
print(f"\n{profile_orig.round(3).to_string()}")

# Sample gemiddelden voor vergelijking
print(f"\n  Sample gemiddelden (ter vergelijking):")
for var in cluster_vars:
    print(f"    {var:25s}: {df[var].mean():.3f}")

# ============================================================
# STAP 5: CLUSTER KARAKTERISERING
# ============================================================
print(f"\n{'='*70}")
print("CLUSTER KARAKTERISERING")
print(f"{'='*70}")

# Automatische karakterisering op basis van z-scores
def characterize_cluster(row):
    """Genereer beschrijving op basis van extreme z-scores."""
    traits = []
    for var in cluster_vars_z:
        val = row[var]
        var_name = var.replace("_z", "")
        if val > 0.5:
            traits.append(f"hoge {var_name}")
        elif val < -0.5:
            traits.append(f"lage {var_name}")
    return traits

for c in range(K):
    row = profile_z.loc[c]
    n = (df["Cluster"] == c).sum()
    traits = characterize_cluster(row)
    
    print(f"\n  CLUSTER {c} (n={n}):")
    if traits:
        print(f"    Kenmerken: {', '.join(traits)}")
    else:
        print(f"    Kenmerken: rond gemiddeld op alle variabelen")
    
    # Toon de meest extreme scores
    extremes = row.abs().sort_values(ascending=False)
    print(f"    Meest onderscheidend: {extremes.index[0].replace('_z', '')} (z={row[extremes.index[0]]:.2f})")

# ============================================================
# STAP 6: R&D INTENSITY PER CLUSTER
# ============================================================
print(f"\n{'='*70}")
print("R&D INTENSITY PER CLUSTER")
print(f"{'='*70}")

# Descriptives
rd_stats = df.groupby("Cluster").agg({
    "RD_Intensity": ["mean", "median", "std"],
    "RD_Intensity_Ln": ["mean", "median", "std"],
}).round(3)

print(f"\n  Raw R&D Intensity:")
print(f"  {'Cluster':>8s} {'Mean':>10s} {'Median':>10s} {'Std':>10s}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
for c in range(K):
    mean = df[df["Cluster"] == c]["RD_Intensity"].mean()
    med = df[df["Cluster"] == c]["RD_Intensity"].median()
    std = df[df["Cluster"] == c]["RD_Intensity"].std()
    print(f"  {c:>8d} {mean:10.3f} {med:10.3f} {std:10.3f}")

print(f"\n  Ln(R&D Intensity):")
print(f"  {'Cluster':>8s} {'Mean':>10s} {'Median':>10s} {'Std':>10s}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
for c in range(K):
    mean = df[df["Cluster"] == c]["RD_Intensity_Ln"].mean()
    med = df[df["Cluster"] == c]["RD_Intensity_Ln"].median()
    std = df[df["Cluster"] == c]["RD_Intensity_Ln"].std()
    print(f"  {c:>8d} {mean:10.3f} {med:10.3f} {std:10.3f}")

# ============================================================
# STAP 7: ANOVA - VERSCHILLEN IN R&D INTENSITY
# ============================================================
print(f"\n{'='*70}")
print("ANOVA: R&D INTENSITY VERSCHILLEN TUSSEN CLUSTERS")
print(f"{'='*70}")

# Groepen voor ANOVA
groups = [df[df["Cluster"] == c]["RD_Intensity_Ln"].values for c in range(K)]

# One-way ANOVA
f_stat, p_value = stats.f_oneway(*groups)
print(f"\n  One-way ANOVA (RD_Intensity_Ln):")
print(f"    F-statistic: {f_stat:.3f}")
print(f"    p-value:     {p_value:.4f}")
print(f"    Significant: {'JA' if p_value < 0.05 else 'NEE'} (α = 0.05)")

# Kruskal-Wallis (non-parametrisch alternatief)
h_stat, kw_p = stats.kruskal(*groups)
print(f"\n  Kruskal-Wallis (non-parametrisch):")
print(f"    H-statistic: {h_stat:.3f}")
print(f"    p-value:     {kw_p:.4f}")
print(f"    Significant: {'JA' if kw_p < 0.05 else 'NEE'} (α = 0.05)")

# Effect size: Eta-squared
ss_between = sum(len(g) * (np.mean(g) - df["RD_Intensity_Ln"].mean())**2 for g in groups)
ss_total = sum((df["RD_Intensity_Ln"] - df["RD_Intensity_Ln"].mean())**2)
eta_squared = ss_between / ss_total
print(f"\n  Effect size:")
print(f"    Eta-squared: {eta_squared:.4f}")
print(f"    Interpretatie: {'klein' if eta_squared < 0.06 else 'medium' if eta_squared < 0.14 else 'groot'}")
print(f"    (klein < 0.06, medium 0.06-0.14, groot > 0.14)")

# ============================================================
# STAP 8: POST-HOC TESTS (TUKEY HSD)
# ============================================================
print(f"\n{'='*70}")
print("POST-HOC: PAIRWISE VERGELIJKINGEN")
print(f"{'='*70}")

from itertools import combinations

print(f"\n  Pairwise t-tests (Bonferroni gecorrigeerd):")
print(f"  {'Vergelijking':>15s} {'Mean diff':>12s} {'t-stat':>10s} {'p-value':>10s} {'Sig':>6s}")
print(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*10} {'-'*6}")

n_comparisons = len(list(combinations(range(K), 2)))
alpha_bonf = 0.05 / n_comparisons

for c1, c2 in combinations(range(K), 2):
    g1 = df[df["Cluster"] == c1]["RD_Intensity_Ln"]
    g2 = df[df["Cluster"] == c2]["RD_Intensity_Ln"]
    
    t_stat, p_val = stats.ttest_ind(g1, g2)
    mean_diff = g1.mean() - g2.mean()
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < alpha_bonf else ""
    
    print(f"  {c1} vs {c2:>11d} {mean_diff:12.3f} {t_stat:10.3f} {p_val:10.4f} {sig:>6s}")

print(f"\n  Bonferroni α = {alpha_bonf:.4f}")

# ============================================================
# STAP 9: INSTITUTIONELE CONTEXT PER CLUSTER
# ============================================================
print(f"\n{'='*70}")
print("INSTITUTIONELE CONTEXT PER CLUSTER (US vs EU)")
print(f"{'='*70}")

print(f"\n  {'Cluster':>8s} {'US (n)':>10s} {'EU (n)':>10s} {'US %':>10s} {'EU %':>10s}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for c in range(K):
    cluster_data = df[df["Cluster"] == c]
    n_us = (cluster_data["Institutional_Context"] == 1).sum()
    n_eu = (cluster_data["Institutional_Context"] == 0).sum()
    pct_us = n_us / len(cluster_data) * 100
    pct_eu = n_eu / len(cluster_data) * 100
    print(f"  {c:>8d} {n_us:>10d} {n_eu:>10d} {pct_us:>9.1f}% {pct_eu:>9.1f}%")

# Sample baseline
n_us_total = (df["Institutional_Context"] == 1).sum()
n_eu_total = (df["Institutional_Context"] == 0).sum()
print(f"  {'Sample':>8s} {n_us_total:>10d} {n_eu_total:>10d} {n_us_total/len(df)*100:>9.1f}% {n_eu_total/len(df)*100:>9.1f}%")

# Chi-square test: is de verdeling US/EU significant anders per cluster?
contingency = pd.crosstab(df["Cluster"], df["Institutional_Context"])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
print(f"\n  Chi-square test (cluster x institutional context):")
print(f"    Chi2: {chi2:.3f}, p-value: {chi_p:.4f}")
print(f"    Significant: {'JA' if chi_p < 0.05 else 'NEE'} - {'Clusters verschillen in US/EU samenstelling' if chi_p < 0.05 else 'Clusters hebben vergelijkbare US/EU verdeling'}")

# ============================================================
# STAP 10: SECTOR PER CLUSTER
# ============================================================
print(f"\n{'='*70}")
print("SECTOR PER CLUSTER")
print(f"{'='*70}")

print(f"\n  {'Cluster':>8s} {'Tech (n)':>10s} {'Health (n)':>12s} {'Tech %':>10s}")
print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10}")

for c in range(K):
    cluster_data = df[df["Cluster"] == c]
    n_tech = (cluster_data["Sector"] == "Tech").sum()
    n_health = (cluster_data["Sector"] == "Healthcare").sum()
    pct_tech = n_tech / len(cluster_data) * 100
    print(f"  {c:>8d} {n_tech:>10d} {n_health:>12d} {pct_tech:>9.1f}%")

# ============================================================
# STAP 11: VISUALISATIES
# ============================================================
print(f"\n{'='*70}")
print("VISUALISATIES OPSLAAN")
print(f"{'='*70}")

# Plot 1: Cluster profielen (radar/spider chart style als bar chart)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Subplot 1: Cluster profielen (z-scores)
ax1 = axes[0, 0]
x = np.arange(len(cluster_vars))
width = 0.2
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for c in range(K):
    offset = (c - 1.5) * width
    vals = profile_z.loc[c].values
    ax1.bar(x + offset, vals, width, label=f'Cluster {c}', color=colors[c], alpha=0.8)

ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax1.axhline(y=-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax1.set_ylabel('Z-Score')
ax1.set_title('Cluster Profielen (Z-Scores)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([v.replace('_', '\n') for v in cluster_vars], fontsize=9)
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# Subplot 2: R&D Intensity per cluster (boxplot)
ax2 = axes[0, 1]
box_data = [df[df["Cluster"] == c]["RD_Intensity_Ln"] for c in range(K)]
bp = ax2.boxplot(box_data, labels=[f'Cluster {c}' for c in range(K)], patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax2.set_ylabel('Ln(R&D Intensity)')
ax2.set_title(f'R&D Intensity per Cluster\n(ANOVA p={p_value:.4f})', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Subplot 3: Cluster groottes
ax3 = axes[1, 0]
sizes = [len(df[df["Cluster"] == c]) for c in range(K)]
bars = ax3.bar([f'Cluster {c}' for c in range(K)], sizes, color=colors, alpha=0.8)
ax3.set_ylabel('Aantal firms')
ax3.set_title('Cluster Groottes', fontweight='bold')
for bar, size in zip(bars, sizes):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{size}', ha='center', va='bottom', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Subplot 4: US/EU verdeling per cluster
ax4 = axes[1, 1]
us_counts = [len(df[(df["Cluster"] == c) & (df["Institutional_Context"] == 1)]) for c in range(K)]
eu_counts = [len(df[(df["Cluster"] == c) & (df["Institutional_Context"] == 0)]) for c in range(K)]
x = np.arange(K)
ax4.bar(x - 0.2, us_counts, 0.4, label='US (LME)', color='#1f77b4', alpha=0.8)
ax4.bar(x + 0.2, eu_counts, 0.4, label='EU (CME)', color='#ff7f0e', alpha=0.8)
ax4.set_ylabel('Aantal firms')
ax4.set_title('Institutionele Context per Cluster', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([f'Cluster {c}' for c in range(K)])
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('Output/cluster_analysis_k4.png', dpi=150, bbox_inches='tight')
print(f"  Opgeslagen: 'Output/cluster_analysis_k4.png'")

# Plot 2: Silhouette plot
fig2, ax = plt.subplots(figsize=(10, 8))
sample_silhouettes = silhouette_samples(X, df["Cluster"])

y_lower = 10
for c in range(K):
    cluster_silhouettes = sample_silhouettes[df["Cluster"] == c]
    cluster_silhouettes.sort()
    
    size = len(cluster_silhouettes)
    y_upper = y_lower + size
    
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouettes,
                     facecolor=colors[c], edgecolor=colors[c], alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size, f'Cluster {c}', fontsize=10, fontweight='bold')
    
    y_lower = y_upper + 10

ax.axvline(x=sil_score, color='red', linestyle='--', label=f'Avg: {sil_score:.3f}')
ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Cluster')
ax.set_title('Silhouette Plot (K-Means, k=4)', fontweight='bold')
ax.legend(loc='upper right')
ax.set_xlim([-0.2, 0.6])

plt.tight_layout()
plt.savefig('Output/silhouette_k4.png', dpi=150, bbox_inches='tight')
print(f"  Opgeslagen: 'Output/silhouette_k4.png'")

plt.close('all')

# ============================================================
# STAP 12: OPSLAAN DATASET MET CLUSTER LABELS
# ============================================================
df.to_csv("Data/05_Thesis_data_clustered.csv", index=False)
print(f"\n  Dataset opgeslagen: 'Data/05_Thesis_data_clustered.csv'")

# ============================================================
# STAP 13: SAMENVATTING VOOR THESIS
# ============================================================
print(f"\n{'='*70}")
print("SAMENVATTING VOOR THESIS")
print(f"{'='*70}")

print(f"""
  CLUSTERING RESULTATEN (K-Means, k=4)
  
  Sample: {len(df)} firms (US: {n_us_total}, EU: {n_eu_total})
  Silhouette score: {sil_score:.3f}
  
  CLUSTER GROOTTES:
""")

for c in range(K):
    n = (df["Cluster"] == c).sum()
    pct = n / len(df) * 100
    traits = characterize_cluster(profile_z.loc[c])
    traits_str = ", ".join(traits) if traits else "gemiddeld profiel"
    print(f"    Cluster {c}: {n:3d} firms ({pct:4.1f}%) - {traits_str}")

print(f"""
  ANOVA RESULTATEN:
    F({K-1}, {len(df)-K}) = {f_stat:.3f}, p = {p_value:.4f}
    {'Significante verschillen in R&D Intensity tussen clusters' if p_value < 0.05 else 'Geen significante verschillen'}
    Eta-squared = {eta_squared:.4f} ({'klein' if eta_squared < 0.06 else 'medium' if eta_squared < 0.14 else 'groot'} effect)
  
  INSTITUTIONELE VERDELING:
    Chi2 = {chi2:.3f}, p = {chi_p:.4f}
    {'Clusters verschillen significant in US/EU samenstelling' if chi_p < 0.05 else 'Geen significante verschillen in US/EU verdeling'}
""")

print(f"\n{'='*70}")
print("KLAAR")
print(f"{'='*70}")
print(f"\nVolgende stap: 05_regression.py")
print(f"  -> OLS regressie met cluster dummies")
print(f"  -> Interactie-effecten cluster x institutional context")