import pandas as pd
import numpy as np
import os
from scipy import stats

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

pd.set_option("display.float_format", lambda x: f"{x:.3f}")

# ============================================================
# STAP 1: INLADEN CLEANED DATA
# ============================================================
df = pd.read_csv("Data/03_Thesis_data_clean.csv")
print(f"Ingeladen: {len(df)} firms")

# ============================================================
# STAP 2: DEFINIEER CLUSTERING VARIABELEN
# ============================================================
# Deze 6 variabelen worden gebruikt voor cluster analysis.
# CEO_Duality is NIET opgenomen vanwege institutionele confound
# (two-tier boards in DE/AT/NL verbieden CEO-chair combinatie).

cluster_vars = [
    "Board_Independence",    # Ratio onafhankelijke directors
    "Gender_Diversity",      # Ratio vrouwelijke directors
    "Avg_Tenure",            # Gemiddelde zittingsduur (jaren)
    "Internationality_Ratio", # Ratio directors met intl ervaring
    "Board_Size",            # Aantal directors
    "Board_Busyness",        # Gem. andere board seats per director
]

print(f"\nClustering variabelen ({len(cluster_vars)}):")
for var in cluster_vars:
    print(f"  - {var}")

# ============================================================
# STAP 3: BEKIJK ORIGINELE SCHALEN
# ============================================================
# Dit toont waarom standardisatie nodig is: de variabelen
# zitten op compleet verschillende schalen.

print(f"\n{'='*70}")
print("ORIGINELE SCHALEN (voor standardisatie)")
print(f"{'='*70}")
print(f"\n{'Variabele':25s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for var in cluster_vars:
    vals = df[var]
    print(f"{var:25s} {vals.mean():10.3f} {vals.std():10.3f} {vals.min():10.3f} {vals.max():10.3f}")

print(f"\n--> Board_Size (range 3-25) zou zonder standardisatie")
print(f"    veel zwaarder wegen dan Board_Independence (range 0.38-1.0)")

# ============================================================
# STAP 4: Z-SCORE TRANSFORMATIE
# ============================================================
# Formule: z = (x - mean) / std
# Resultaat: mean=0, std=1 voor elke variabele
#
# scipy.stats.zscore() doet dit automatisch.
# We maken nieuwe kolommen met suffix '_z' voor duidelijkheid.

print(f"\n{'='*70}")
print("Z-SCORE TRANSFORMATIE")
print(f"{'='*70}")

for var in cluster_vars:
    z_col = f"{var}_z"
    df[z_col] = stats.zscore(df[var])
    print(f"  {var} -> {z_col}")

# Maak lijst van z-score kolommen
cluster_vars_z = [f"{var}_z" for var in cluster_vars]

# ============================================================
# STAP 5: VERIFICATIE
# ============================================================
# Check dat mean≈0 en std≈1 voor alle z-scores

print(f"\n{'='*70}")
print("VERIFICATIE (na standardisatie)")
print(f"{'='*70}")
print(f"\n{'Variabele':30s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
print(f"{'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for var in cluster_vars_z:
    vals = df[var]
    print(f"{var:30s} {vals.mean():10.6f} {vals.std():10.6f} {vals.min():10.3f} {vals.max():10.3f}")

print(f"\n--> Alle variabelen nu op dezelfde schaal (mean≈0, std=1)")
print(f"    Min/max waarden tonen hoeveel standaarddeviaties van gemiddelde")

# ============================================================
# STAP 6: INTERPRETATIE VOORBEELD
# ============================================================
# Laat zien hoe je z-scores interpreteert

print(f"\n{'='*70}")
print("INTERPRETATIE VOORBEELD")
print(f"{'='*70}")

# Pak een willekeurige firm
example_firm = df.iloc[0]
print(f"\nFirm: {example_firm.get('Company_Name', 'Firm 1')}")
print(f"\n{'Variabele':25s} {'Origineel':>12s} {'Z-score':>10s} {'Interpretatie':>30s}")
print(f"{'-'*25} {'-'*12} {'-'*10} {'-'*30}")

for var in cluster_vars:
    orig = example_firm[var]
    z = example_firm[f"{var}_z"]
    
    if z > 1.5:
        interp = "ver boven gemiddeld"
    elif z > 0.5:
        interp = "boven gemiddeld"
    elif z > -0.5:
        interp = "rond gemiddeld"
    elif z > -1.5:
        interp = "onder gemiddeld"
    else:
        interp = "ver onder gemiddeld"
    
    print(f"{var:25s} {orig:12.3f} {z:10.3f} {interp:>30s}")

# ============================================================
# STAP 7: CORRELATIE CHECK (UNCHANGED)
# ============================================================
# Z-score transformatie verandert correlaties NIET.
# Dit is een sanity check.

print(f"\n{'='*70}")
print("CORRELATIE CHECK")
print(f"{'='*70}")

corr_orig = df[cluster_vars].corr()
corr_z = df[cluster_vars_z].corr()

# Vergelijk een paar correlaties
print(f"\nCorrelaties zijn onveranderd na z-score transformatie:")
print(f"  Board_Independence x Gender_Diversity:")
print(f"    Origineel: {corr_orig.loc['Board_Independence', 'Gender_Diversity']:.4f}")
print(f"    Z-scores:  {corr_z.loc['Board_Independence_z', 'Gender_Diversity_z']:.4f}")

print(f"\n  Board_Size x Internationality_Ratio:")
print(f"    Origineel: {corr_orig.loc['Board_Size', 'Internationality_Ratio']:.4f}")
print(f"    Z-scores:  {corr_z.loc['Board_Size_z', 'Internationality_Ratio_z']:.4f}")

# ============================================================
# STAP 8: OPSLAAN
# ============================================================
# Sla dataset op met zowel originele als z-score variabelen.
# Cluster analysis gebruikt de _z kolommen.
# Descriptives/interpretatie gebruikt de originele kolommen.

df.to_csv("Data/04_Thesis_data_standardized.csv", index=False)

print(f"\n{'='*70}")
print("OPGESLAGEN")
print(f"{'='*70}")
print(f"\nBestand: 'Data/04_Thesis_data_standardized.csv'")
print(f"Dimensies: {df.shape[0]} rijen x {df.shape[1]} kolommen")
print(f"\nNieuwe kolommen toegevoegd:")
for var in cluster_vars_z:
    print(f"  - {var}")

print(f"\n{'='*70}")
print("KLAAR - Dataset gereed voor cluster analysis")
print(f"{'='*70}")
print(f"\nVolgende stap: 03_cluster_selection.py")
print(f"  -> Elbow method, silhouette scores, dendrogram")
print(f"  -> Bepaal optimaal aantal clusters")