import pandas as pd
import numpy as np
import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ============================================================
# STAP 1: INLADEN
# ============================================================
bx_na = pd.read_csv("Data/Raw_BoardEx-NorthAmerica.csv")
bx_gl = pd.read_csv("Data/Raw_BoardEx-Global.csv")
compustat = pd.read_csv("Data/01_compustat_merged_clean.csv")

print(f"BoardEx NA: {len(bx_na)} rijen, {bx_na['Isin'].nunique()} firms")
print(f"BoardEx Global: {len(bx_gl)} rijen, {bx_gl['Isin'].nunique()} firms")
print(f"Compustat: {len(compustat)} firms")

# ============================================================
# STAP 2: BOARDEX SAMENVOEGEN
# ============================================================
bx = pd.concat([bx_na, bx_gl], ignore_index=True)
print(f"\nBoardEx gecombineerd: {len(bx)} rijen, {bx['Isin'].nunique()} firms")

# ============================================================
# STAP 3: CEO DUALITY DETECTIE
# ============================================================
# Logica: een firm heeft CEO duality als minstens een director
# in zijn/haar Role_Name zowel "CEO" als "Chair/Chairman/Chairwoman/
# Chairperson" heeft staan. BoardEx combineert duale rollen in
# een enkel veld, bijv. "Chairman/CEO" of "Chair/President/CEO".
#
# NB: CEO Duality wordt NIET gebruikt als clustering variabele
# vanwege institutionele confound (two-tier systemen in DE/AT
# maken duality structureel onmogelijk). We behouden de variabele
# wel voor descriptieve analyses.

def has_ceo_duality(role):
    if pd.isna(role):
        return False
    r = role.lower()
    is_ceo = "ceo" in r
    is_chair = any(x in r for x in ["chairman", "chairwoman", "chairperson", "chair/", "chair "])
    if not is_chair:
        is_chair = r == "chair" or r.startswith("chair/") or r.startswith("chair ")
    return is_ceo and is_chair

bx["_is_dual"] = bx["Role_Name"].apply(has_ceo_duality)
ceo_duality = bx.groupby("Isin")["_is_dual"].any().astype(int).reset_index()
ceo_duality.columns = ["Isin", "CEO_Duality"]

print(f"\nCEO Duality: {ceo_duality['CEO_Duality'].sum()} firms van {len(ceo_duality)}")

# ============================================================
# STAP 4: BOARD INDEPENDENCE
# ============================================================
# Logica: Director_Type = "Yes" betekent Non-Executive Director
# (NED) in BoardEx. We berekenen het aandeel NEDs als proxy
# voor board independence. Dit is conservatief: niet alle NEDs
# zijn formeel onafhankelijk, maar BoardEx classificeert
# onafhankelijke directors consistent als NED.

def calc_independence(group):
    total = len(group)
    independent = (group["Director_Type"] == "Yes").sum()
    return independent / total if total > 0 else np.nan

board_independence = bx.groupby("Isin").apply(calc_independence).reset_index()
board_independence.columns = ["Isin", "Board_Independence"]

# ============================================================
# STAP 5: AVERAGE TENURE
# ============================================================
# Logica: gemiddelde van Time_Board (jaren op het board) over
# alle directors van de firm. Time_Board is betrouwbaarder dan
# Time_Role omdat directors van rol kunnen wisselen binnen
# hetzelfde board.

avg_tenure = bx.groupby("Isin")["Time_Board"].mean().reset_index()
avg_tenure.columns = ["Isin", "Avg_Tenure"]

# ============================================================
# STAP 6: BOARD-LEVEL VARIABELEN (al per firm in BoardEx)
# ============================================================
# Deze variabelen zijn in BoardEx al op board-niveau berekend:
# dezelfde waarde voor elke director binnen een firm. We pakken
# simpelweg de eerste waarde per ISIN.
#
# - Gender_Ratio = aandeel mannelijke directors (geverifieerd)
#   -> Gender_Diversity = 1 - Gender_Ratio = aandeel vrouwelijk
# - Nationality_Mix = aandeel directors met andere nationaliteit
#   dan het hoofdkantoorland -> Internationality Ratio
# - Number_Directors = totaal board members -> Board Size
# - Stdev_age = standaarddeviatie leeftijd -> Age Diversity

board_level = bx.groupby("Isin").first()[
    ["Gender_Ratio", "Nationality_Mix", "Number_Directors", "Stdev_age"]
].reset_index()

board_level["Gender_Diversity"] = 1 - board_level["Gender_Ratio"]
board_level = board_level.rename(columns={
    "Nationality_Mix": "Internationality_Ratio",
    "Number_Directors": "Board_Size",
    "Stdev_age": "Age_Diversity",
})
board_level = board_level.drop(columns=["Gender_Ratio"])

# ============================================================
# STAP 7: BOARD BUSYNESS
# ============================================================
# Logica: gemiddeld aantal HUIDIGE bestuurszetels per director.
# Total_Boards_Current telt hoeveel listed boards een director
# momenteel bezet. Een hoog gemiddelde betekent dat directors
# hun aandacht verdelen over meerdere bedrijven (Agency Theory:
# minder monitoring capaciteit; RDT: meer netwerktoegang).

avg_busyness = bx.groupby("Isin")["Total_Boards_Current"].mean().reset_index()
avg_busyness.columns = ["Isin", "Board_Busyness"]

# ============================================================
# STAP 8: GEMIDDELD AANTAL QUALIFICATIES
# ============================================================
# Logica: gemiddeld aantal formele kwalificaties (graden/
# certificeringen) per director. Proxy voor menselijk kapitaal
# op het board (Upper Echelons Theory).

avg_quals = bx.groupby("Isin")["No_Qual"].mean().reset_index()
avg_quals.columns = ["Isin", "Avg_Qualifications"]

# ============================================================
# STAP 9: ALLES SAMENVOEGEN TOT FIRM-LEVEL BOARDEX DATASET
# ============================================================
boardex_firm = ceo_duality
for df_part in [board_independence, avg_tenure, board_level, avg_busyness, avg_quals]:
    boardex_firm = boardex_firm.merge(df_part, on="Isin", how="left")

print(f"\nBoardEx firm-level dataset: {len(boardex_firm)} firms")
print(f"Kolommen: {list(boardex_firm.columns)}")

# ============================================================
# STAP 10: MERGE MET COMPUSTAT
# ============================================================
before = len(compustat)
merged = compustat.merge(boardex_firm, on="Isin", how="inner")
print(f"\nCompustat firms: {before}")
print(f"Na merge met BoardEx (inner join): {len(merged)}")
print(f"Verloren door geen BoardEx data: {before - len(merged)}")

# ============================================================
# STAP 11: KWALITEITSCHECK
# ============================================================
print(f"\n--- MISSING VALUES IN BOARD VARIABELEN ---")
board_cols = ["Board_Independence", "Gender_Diversity", "Avg_Tenure",
              "CEO_Duality", "Internationality_Ratio", "Board_Size",
              "Board_Busyness", "Age_Diversity", "Avg_Qualifications"]
for col in board_cols:
    missing = merged[col].isnull().sum()
    print(f"  {col}: {missing} missing")

print(f"\n--- DESCRIPTIVE STATISTICS ---")
print(merged[board_cols].describe().round(3).to_string())

# ============================================================
# STAP 12: DATASET OVERZICHT
# ============================================================
us = merged[merged["Institutional_Context"] == 1]
eu = merged[merged["Institutional_Context"] == 0]

print(f"\n============================")
print(f"DATASET NA MERGE (voor cleaning)")
print(f"============================")
print(f"Totaal firms: {len(merged)}")
print(f"  US (LME):  {len(us)}")
print(f"  EU (CME):  {len(eu)}")
print(f"\nPer sector:")
print(f"  Tech:       {len(merged[merged['Sector'] == 'Tech'])}")
print(f"  Healthcare: {len(merged[merged['Sector'] == 'Healthcare'])}")
print(f"\nCEO Duality per context (descriptief, niet in clustering):")
print(f"  US: {us['CEO_Duality'].sum()}/{len(us)} ({us['CEO_Duality'].mean()*100:.1f}%)")
print(f"  EU: {eu['CEO_Duality'].sum()}/{len(eu)} ({eu['CEO_Duality'].mean()*100:.1f}%)")
print(f"\nGender Diversity gemiddeld:")
print(f"  US: {us['Gender_Diversity'].mean()*100:.1f}%")
print(f"  EU: {eu['Gender_Diversity'].mean()*100:.1f}%")
print(f"\nBoard Independence gemiddeld:")
print(f"  US: {us['Board_Independence'].mean()*100:.1f}%")
print(f"  EU: {eu['Board_Independence'].mean()*100:.1f}%")

# ============================================================
# STAP 13: OPSLAAN
# ============================================================
merged.to_csv("Data/02_Thesis_data_complete.csv", index=False)
print(f"\nOpgeslagen als '02_Thesis_data_complete.csv'")
print(f"Dimensies: {merged.shape[0]} rijen x {merged.shape[1]} kolommen")
print(f"Kolommen: {list(merged.columns)}")
print(f"\n>>> Volgende stap: run Cleaning.py voor filtering en random sampling <<<")