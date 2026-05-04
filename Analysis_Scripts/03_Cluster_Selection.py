import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings("ignore")


os.makedirs("Output", exist_ok=True)


# ============================================================
# STAP 1: INLADEN
# ============================================================
df = pd.read_csv("Data/04_Thesis_data_standardized.csv")
print(f"Ingeladen: {len(df)} firms")

# Z-score kolommen voor clustering
cluster_vars_z = [
    "Board_Independence_z",
    "Gender_Diversity_z",
    "Avg_Tenure_z",
    "Internationality_Ratio_z",
    "Board_Size_z",
    "Board_Busyness_z",
]

X = df[cluster_vars_z].values
print(f"Clustering matrix: {X.shape[0]} firms x {X.shape[1]} variabelen")

# ============================================================
# STAP 2: ELBOW METHOD (K-MEANS)
# ============================================================
# De elbow method plot de within-cluster sum of squares (WCSS)
# tegen het aantal clusters. Het "elleboog" punt is waar de
# afname in WCSS begint af te vlakken.

print(f"\n{'='*70}")
print("ELBOW METHOD (K-Means)")
print(f"{'='*70}")

k_range = range(2, 11)
wcss = []  # Within-Cluster Sum of Squares (inertia)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    print(f"  k={k}: WCSS = {kmeans.inertia_:.2f}")

# Bereken de "elbow" via de grootste daling
wcss_diff = np.diff(wcss)
wcss_diff2 = np.diff(wcss_diff)  # Tweede afgeleide
elbow_k = list(k_range)[np.argmax(wcss_diff2) + 1]  # +1 voor offset
print(f"\n  Geschatte elbow: k = {elbow_k}")

# ============================================================
# STAP 3: SILHOUETTE SCORES
# ============================================================
# Silhouette score meet hoe goed elke observatie bij zijn cluster
# past vs. andere clusters. Range: -1 (slecht) tot +1 (perfect).
# Hoger = beter gedefinieerde clusters.

print(f"\n{'='*70}")
print("SILHOUETTE SCORES")
print(f"{'='*70}")

silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"  k={k}: Silhouette = {score:.4f}")

best_k_silhouette = list(k_range)[np.argmax(silhouette_scores)]
print(f"\n  Beste k (hoogste silhouette): k = {best_k_silhouette} (score = {max(silhouette_scores):.4f})")

# Interpretatie
print(f"\n  Interpretatie silhouette scores:")
print(f"    0.71-1.00: Sterke structuur")
print(f"    0.51-0.70: Redelijke structuur")
print(f"    0.26-0.50: Zwakke structuur")
print(f"    < 0.25: Geen substantiele structuur")

# ============================================================
# STAP 4: HIERARCHICAL CLUSTERING (WARD'S METHOD)
# ============================================================
# Ward's method minimaliseert de totale within-cluster variantie.
# Dit is de meest gebruikte methode voor agglomeratieve clustering.

print(f"\n{'='*70}")
print("HIERARCHICAL CLUSTERING (Ward's Method)")
print(f"{'='*70}")

# Bereken linkage matrix
Z = linkage(X, method='ward')

# Bekijk de laatste 10 merges (hoogste niveau)
print(f"\n  Laatste 10 cluster merges:")
print(f"  {'Merge':>6s} {'Cluster1':>10s} {'Cluster2':>10s} {'Distance':>12s} {'N obs':>8s}")
print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")
for i in range(-10, 0):
    c1, c2, dist, n = Z[i]
    print(f"  {len(Z)+i+1:6d} {int(c1):10d} {int(c2):10d} {dist:12.2f} {int(n):8d}")

# ============================================================
# STAP 5: SILHOUETTE VOOR WARD'S METHOD
# ============================================================
print(f"\n{'='*70}")
print("SILHOUETTE SCORES (Ward's Hierarchical)")
print(f"{'='*70}")

ward_silhouettes = []

for k in k_range:
    labels = fcluster(Z, k, criterion='maxclust')
    score = silhouette_score(X, labels)
    ward_silhouettes.append(score)
    print(f"  k={k}: Silhouette = {score:.4f}")

best_k_ward = list(k_range)[np.argmax(ward_silhouettes)]
print(f"\n  Beste k (Ward's): k = {best_k_ward} (score = {max(ward_silhouettes):.4f})")

# ============================================================
# STAP 6: VERGELIJKING K-MEANS vs WARD'S
# ============================================================
print(f"\n{'='*70}")
print("VERGELIJKING K-MEANS vs WARD'S")
print(f"{'='*70}")

print(f"\n  {'k':>3s} {'K-Means':>12s} {'Ward':>12s} {'Beste':>10s}")
print(f"  {'-'*3} {'-'*12} {'-'*12} {'-'*10}")

for i, k in enumerate(k_range):
    km_score = silhouette_scores[i]
    ward_score = ward_silhouettes[i]
    beste = "K-Means" if km_score > ward_score else "Ward's"
    print(f"  {k:3d} {km_score:12.4f} {ward_score:12.4f} {beste:>10s}")

# ============================================================
# STAP 7: CLUSTER GROOTTE BIJ VERSCHILLENDE K
# ============================================================
print(f"\n{'='*70}")
print("CLUSTER GROOTTES (Ward's Method)")
print(f"{'='*70}")

for k in [3, 4, 5, 6]:
    labels = fcluster(Z, k, criterion='maxclust')
    sizes = pd.Series(labels).value_counts().sort_index()
    pcts = (sizes / len(labels) * 100).round(1)
    print(f"\n  k={k}:")
    for cluster, (size, pct) in enumerate(zip(sizes, pcts), 1):
        bar = "█" * int(pct / 2)
        print(f"    Cluster {cluster}: {size:4d} firms ({pct:5.1f}%) {bar}")

# ============================================================
# STAP 8: MAAK VISUALISATIES
# ============================================================
print(f"\n{'='*70}")
print("VISUALISATIES OPSLAAN")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Elbow Method
ax1 = axes[0, 0]
ax1.plot(list(k_range), wcss, 'b-o', linewidth=2, markersize=8)
ax1.axvline(x=elbow_k, color='r', linestyle='--', label=f'Elbow at k={elbow_k}')
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('Within-Cluster Sum of Squares', fontsize=12)
ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Silhouette Scores
ax2 = axes[0, 1]
ax2.plot(list(k_range), silhouette_scores, 'g-o', linewidth=2, markersize=8, label='K-Means')
ax2.plot(list(k_range), ward_silhouettes, 'b-s', linewidth=2, markersize=8, label="Ward's")
ax2.axvline(x=best_k_silhouette, color='g', linestyle='--', alpha=0.5)
ax2.axvline(x=best_k_ward, color='b', linestyle='--', alpha=0.5)
ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Dendrogram (full)
ax3 = axes[1, 0]
dendrogram(Z, truncate_mode='level', p=5, ax=ax3, 
           leaf_font_size=8, above_threshold_color='gray')
ax3.set_xlabel('Sample Index or Cluster Size', fontsize=12)
ax3.set_ylabel('Distance (Ward)', fontsize=12)
ax3.set_title('Dendrogram (Truncated)', fontsize=14, fontweight='bold')

# Plot 4: Dendrogram met cut lines voor k=3,4,5
ax4 = axes[1, 1]
dendrogram(Z, truncate_mode='level', p=6, ax=ax4,
           leaf_font_size=8, above_threshold_color='gray')

# Voeg horizontale lijnen toe voor verschillende k waarden
# De afstand voor k clusters is tussen de (n-k)e en (n-k+1)e merge
n = len(X)
for k, color, style in [(3, 'red', '--'), (4, 'blue', '-.'), (5, 'green', ':')]:
    if k < n:
        cut_dist = (Z[n-k, 2] + Z[n-k-1, 2]) / 2
        ax4.axhline(y=cut_dist, color=color, linestyle=style, 
                    label=f'k={k} (dist≈{cut_dist:.1f})', linewidth=2)

ax4.set_xlabel('Sample Index or Cluster Size', fontsize=12)
ax4.set_ylabel('Distance (Ward)', fontsize=12)
ax4.set_title('Dendrogram with Cluster Cuts', fontsize=14, fontweight='bold')
ax4.legend(loc='upper right')

plt.tight_layout()
plt.savefig('Output/cluster_selection_analysis.png', dpi=150, bbox_inches='tight')
print(f"  Opgeslagen: 'Output/cluster_selection_analysis.png'")

# ============================================================
# STAP 9: SILHOUETTE PLOT VOOR BESTE K
# ============================================================
# Dit toont de silhouette score per observatie, gegroepeerd per cluster

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

for idx, k in enumerate([3, 4, 5]):
    ax = axes2[idx]
    
    # Gebruik Ward's method
    labels = fcluster(Z, k, criterion='maxclust')
    
    # Bereken silhouette scores per sample
    sample_silhouettes = silhouette_samples(X, labels)
    avg_score = silhouette_score(X, labels)
    
    y_lower = 10
    for i in range(1, k + 1):
        cluster_silhouettes = sample_silhouettes[labels == i]
        cluster_silhouettes.sort()
        
        size_cluster = len(cluster_silhouettes)
        y_upper = y_lower + size_cluster
        
        color = plt.cm.tab10(i / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouettes,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        # Label in midden van cluster
        ax.text(-0.05, y_lower + 0.5 * size_cluster, str(i), fontsize=10, fontweight='bold')
        
        y_lower = y_upper + 10
    
    ax.axvline(x=avg_score, color='red', linestyle='--', 
               label=f'Avg: {avg_score:.3f}')
    ax.set_title(f'k = {k}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    ax.set_xlim([-0.2, 1])
    ax.legend(loc='upper right')

plt.suptitle('Silhouette Analysis per Cluster (Ward\'s Method)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('Output/silhouette_analysis.png', dpi=150, bbox_inches='tight')
print(f"  Opgeslagen: 'Output/silhouette_analysis.png'")

plt.close('all')

# ============================================================
# STAP 10: AANBEVELING
# ============================================================
print(f"\n{'='*70}")
print("AANBEVELING")
print(f"{'='*70}")

print(f"""
  Methode           Optimaal k
  ---------------   ----------
  Elbow Method      {elbow_k}
  Silhouette (KM)   {best_k_silhouette}
  Silhouette (Ward) {best_k_ward}

  OVERWEGINGEN:
  
  1. Statistische criteria suggereren k = {best_k_silhouette} of {best_k_ward}
  
  2. Maar de BESTE keuze hangt ook af van:
     - Interpreteerbaarheid: Kun je de clusters theoretisch duiden?
     - Sample size per cluster: Minimaal 30-50 firms per cluster voor
       betrouwbare ANOVA/regressie (464 / k = {464//3}-{464//6} per cluster)
     - Literatuur: Wat is gangbaar in corporate governance research?
  
  3. MIJN AANBEVELING: Test k = 3, 4, en 5
     - Run de clustering met elk van deze k-waarden
     - Profile de clusters (gemiddelden per variabele)
     - Kies de k die het meest interpreteerbaar is
  
  4. Voor je thesis: Rapporteer alle drie de methodes (elbow, silhouette,
     dendrogram) en leg uit waarom je voor een bepaalde k hebt gekozen.
""")

print(f"\n{'='*70}")
print("KLAAR")
print(f"{'='*70}")
print(f"\nVolgende stap: 04_clustering.py")
print(f"  -> Voer Ward's clustering uit met gekozen k")
print(f"  -> Profile de clusters")
print(f"  -> ANOVA op R&D Intensity per cluster")