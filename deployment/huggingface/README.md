---
title: EGFR Bioactivity Predictor
emoji: 🧬
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
license: mit
short_description: Predict EGFR pIC50 from SMILES — LightGBM + ECFP4 + Lipinski Ro5
tags:
  - drug-discovery
  - cheminformatics
  - qsar
  - egfr
  - rdkit
  - lightgbm
  - streamlit
---

# 🧬 EGFR Bioactivity Predictor

Interactive Streamlit app: paste a SMILES string and get a predicted **pIC50**
against EGFR (Epidermal Growth Factor Receptor), an estimated **IC50 (nM)**, an
activity class, and a **Lipinski Rule-of-Five** drug-likeness report — with
the structure rendered alongside.

* **Model:** LightGBM regressor trained on **10,546 ChEMBL 33** EGFR IC50
  records (curated, deduplicated, pIC50 ∈ [3, 12]).
* **Features:** Morgan / ECFP4 fingerprints (radius 2, 2048 bits) +
  5 Lipinski descriptors (MW, LogP, HBD, HBA, TPSA).
* **Test-set performance:** R² ≈ 0.715, RMSE ≈ 0.704 (5-fold CV).

The model is trained on first launch from
[`data/egfr_bioactivity_curated.csv`](./data/egfr_bioactivity_curated.csv)
(~30 s) and cached to `outputs/best_model.joblib` for subsequent reloads.

## Source repository

The full pipeline (notebook with 26 sections — EDA, SHAP, applicability domain,
stacking, ChemBERTa, ADMET, GNN…) lives at
**https://github.com/OleksandrUK911/egfr-bioactivity-prediction**.

## Disclaimer

For research / educational use only. Predictions are model estimates; they
**must not** be used for clinical decision-making.
