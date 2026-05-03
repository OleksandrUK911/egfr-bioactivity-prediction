# Portfolio entry — `egfr-bioactivity-prediction`

Drop-in copy for a personal portfolio website (Hugo/Jekyll/Astro/Next.js
"projects" pages or a plain static site).

---

## Card / summary (≤ 60 words)

> **EGFR Bioactivity Predictor** — A 30-section end-to-end ML pipeline that
> turns 10,546 ChEMBL IC50 records into a deployable Streamlit app
> predicting pIC50 from a SMILES string. Best model: stacked ensemble,
> R² = 0.735 (scaffold-split R² = 0.607). Includes SHAP, applicability domain,
> conformal prediction, ChemBERTa, GCN, bootstrap CIs, activity cliff analysis,
> and learning curves.

**Stack:** Python · RDKit · scikit-learn · XGBoost · LightGBM · SHAP · Optuna ·
PyTorch · PyTorch Geometric · Hugging Face Transformers · PyTDC · Streamlit
· Hugging Face Spaces.

**Links:** [GitHub](https://github.com/OleksandrUK911/egfr-bioactivity-prediction) · Live demo (Hugging Face Space) · [Blog post](./blog/egfr-bioactivity-prediction)

---

## Long form (project page body)

### Problem

Predict the bioactivity (pIC50) of small-molecule inhibitors against EGFR —
the clinical target behind AstraZeneca's Osimertinib (Tagrisso) — directly
from a SMILES string, with calibrated uncertainty and a clear applicability
domain.

### Approach

1. **Curated dataset.** 10,546 ChEMBL 33 IC50 records, validated SMILES,
   median-aggregated by compound, pIC50 ∈ [3, 12].
2. **Feature engineering.** ECFP4 (Morgan, radius 2, 2048 bits), Lipinski
   Ro5 descriptors, MACCS keys, RDKit FP, RDKit 2D descriptors,
   3D conformer shape (PMI, asphericity, RoG), and ChemBERTa embeddings.
3. **Modelling.** Random Forest, XGBoost, LightGBM, SVR — Optuna-tuned
   (50 trials Bayesian) and stacked with a Ridge meta-learner.
4. **Interpretability & trust.** SHAP TreeExplainer, Murcko scaffold
   coverage, k-NN applicability domain (95th percentile), split conformal
   prediction (90 % empirical coverage).
5. **Phase 3 scope.** Multi-task ErbB selectivity, ADMET profile via PyTDC,
   3-layer GCN (PyTorch Geometric) baseline.
6. **Deployment.** Streamlit web app (`app.py`) + Hugging Face Spaces
   deploy package.

### Results

| Model              | Test R² | RMSE  | MAE   |
|--------------------|---------|-------|-------|
| LightGBM (best base)|  0.715 | 0.704 | 0.527 |
| Random Forest       |  0.695 | 0.728 | 0.559 |
| SVR (RBF)           |  0.692 | 0.732 | 0.543 |
| XGBoost             |  0.664 | 0.764 | 0.591 |
| **Stacked ensemble**| **0.735** | **0.679** | **0.512** |

### Why it's portfolio-worthy

* **Reflects industrial workflow** — curation, multiple baselines,
  interpretability, applicability domain, uncertainty, deployment.
* **Reproducible** — single `requirements.txt`, single notebook (also
  shipped as a fully-executed `bioactivity_prediction_executed.ipynb`),
  Colab badge.
* **Open** — MIT-licensed, all metrics in `outputs/model_summary.json`.

### Lessons learned

* Curation eats most of the time; logging row counts after every filter step
  is non-negotiable.
* Model diversity (stacking) buys more accuracy than hyperparameter tuning.
* Applicability domain is the most under-rated cheminformatics tool —
  it tells you when *not* to trust the model.
* GNNs are not free wins at the ~10 k single-target dataset scale.

### Tech callouts

* **Custom multi-source data loader** (`fetch_data.py`) — ChEMBL +
  PubChem AID 1851 + BindingDB + ExCAPE-DB → unified consensus table.
* **Phase 3 GNN section (§26)** — RDKit→PyG featuriser, 3 × GCNConv +
  global mean pool + MLP head, evaluated on the same train/test split as
  the fingerprint baselines.
* **Streamlit app** trains on first launch (~30 s), caches a joblib
  artifact, renders the input molecule and reports pIC50 + IC50 (nM) +
  activity class + Lipinski Ro5 violations.
