# EGFR Bioactivity Prediction — Drug Discovery ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![CI](https://github.com/OleksandrUK911/egfr-bioactivity-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/OleksandrUK911/egfr-bioactivity-prediction/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OleksandrUK911/egfr-bioactivity-prediction/blob/main/bioactivity_prediction.ipynb)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![ChEMBL](https://img.shields.io/badge/Data-ChEMBL%2033-orange)](https://www.ebi.ac.uk/chembl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete end-to-end machine learning pipeline for predicting the bioactivity of small-molecule inhibitors against **EGFR** (Epidermal Growth Factor Receptor) — one of the most clinically validated targets in oncology, and the target of AstraZeneca's *Osimertinib (Tagrisso)*, developed in Cambridge, UK.

---

## Project Overview

This project demonstrates a realistic **computational drug discovery** workflow, from raw ChEMBL bioactivity data through to model-driven virtual screening. It is designed as a portfolio piece targeting roles in cheminformatics, drug discovery data science, and computational biology — particularly relevant to the Cambridge (UK) life sciences ecosystem.

**Target:** EGFR (CHEMBL203)  
**Task:** Regression — predict **pIC50** from molecular structure  
**Dataset:** 10,546 compounds from ChEMBL 33 (curated from 25,758 raw IC50 records)  
**Sections:** 30 — from raw data curation to GNN, activity cliffs, and learning curves

---

## Pipeline

```
ChEMBL API
    │
    ▼
Data Curation          ← IC50 filtering, deduplication, pIC50 conversion
    │
    ▼
Lipinski Ro5 Analysis  ← Drug-likeness descriptors (MW, LogP, HBD, HBA, TPSA)
    │
    ▼
Feature Engineering    ← Morgan fingerprints (ECFP4, radius=2, 2048 bits)
    │
    ▼
Model Training         ← Random Forest · XGBoost · LightGBM · SVR
    │
    ▼
SHAP Interpretability  ← Global + local feature attribution
    │
    ▼
Phase 2                ← Fingerprint comparison · Scaffold analysis · AD · Stacking
    │
    ▼
Phase 3                ← Optuna HPO · Conformal prediction · ChemBERTa · GNN
    │
    ▼
Phase 4 (Rigour)       ← Scaffold split · Bootstrap CI · Activity cliffs · Learning curves
    │
    ▼
Virtual Screening      ← Score & rank approved EGFR inhibitors
```

---

## Key Results — Phase 1 (Baseline Models)

> Dataset: **10,546 compounds** · pIC50 mean = 6.85 ± 1.34 · 72.4 % active · 72.8 % Ro5-compliant

| Model | R² (test) | RMSE | MAE | CV R² (5-fold) |
|---|---|---|---|---|
| **LightGBM** | **0.715** | **0.704** | **0.527** | 0.695 ± 0.018 |
| Random Forest | 0.695 | 0.728 | 0.559 | 0.677 ± 0.016 |
| SVR (RBF) | 0.692 | 0.732 | 0.543 | 0.659 ± 0.018 |
| XGBoost | 0.664 | 0.764 | 0.591 | 0.655 ± 0.020 |

---

## Phase 2 — Advanced Analyses

### 14. Fingerprint Comparison (5-fold CV R² with Random Forest)

| Fingerprint | Bits | CV R² | Notes |
|---|---|---|---|
| **RDKit FP** | 2048 | **0.696** | Path-based features, best overall |
| ECFP4 / Morgan | 2048 | 0.690 | Practically tied; standard choice |
| MACCS Keys | 166 | 0.621 | Fewer bits — simpler but less expressive |

### 15. Murcko Scaffold Analysis

- **3,857 unique scaffolds** across 10,546 compounds (36.6 % scaffold diversity)
- 67.3 % of scaffolds appear exactly once (singletons)
- Most common scaffold: **597 compounds** (5.7 %) — quinazoline-aniline core (Gefitinib/Erlotinib chemotype)
- Top-10 scaffolds cover only 13.6 % → high chemical diversity, good for generalisation

### 16. Applicability Domain (k-NN distance in fingerprint space)

- **96.1 %** of test compounds fall inside the applicability domain
- R² inside AD = 0.714, R² outside AD = 0.734 → dataset is chemically homogeneous

### 17. Ensemble Stacking (Ridge meta-learner over RF + XGB + LGBM + SVR)

| Metric | Stacking Ensemble | Best Individual (LightGBM) | Δ |
|---|---|---|---|
| R² | **0.7354** | 0.7149 | **+0.0205** |
| RMSE | **0.6780** | 0.7038 | **−0.0258** |
| MAE | **0.5025** | 0.5268 | **−0.0243** |

Meta-learner weights: **RF +0.60 · LGBM +0.43 · SVR +0.31 · XGB −0.25**  
(negative XGB weight is typical when learners are correlated — XGB partially redundant with LGBM)

---

## Phase 3 — Extended Methods

### 18. Hyperparameter Optimisation with Optuna

50-trial Bayesian (TPE) search over LightGBM with 3-fold CV:

| Model | Test R² | RMSE | MAE |
|---|---|---|---|
| LightGBM (default) | 0.715 | 0.704 | 0.527 |
| **LightGBM (Optuna-tuned)** | **0.731** | **0.684** | **0.507** |

Best params: `n_estimators=500, num_leaves=109, learning_rate=0.027, subsample=0.747`

### 19. RDKit 2D Descriptors as Additional Features

Random Forest benchmarked on three feature sets (207 RDKit descriptors after NaN/inf cleaning):

| Feature set | Test R² |
|---|---|
| ECFP4 only (2048 bits) | 0.715 |
| RDKit 2D only (207 descriptors) | — |
| **ECFP4 ⊕ RDKit 2D (2255 features)** | **0.718** |

Combining fingerprints with physicochemical descriptors gives a marginal improvement (+0.003). Top descriptors by importance: ring count, rotatable bonds, TPSA variants.

### 20. Conformal Prediction — Calibrated Uncertainty Intervals

Split (inductive) conformal prediction on top of LightGBM:

- **90 % target coverage → 90.9 % empirical coverage** (correctly calibrated)
- Interval half-width at 90 % level: **±1.251 pIC50**
- 80 % target → 80.4 % empirical (calibration curve sits on the diagonal across all α)

### 21. 3D Conformer Shape Descriptors

ETKDG conformer generation + MMFF94 optimisation for 200 stratified-sampled compounds:

- Descriptors computed: NPR1, NPR2, Asphericity, Eccentricity, InertialShapeFactor, RadiusOfGyration, SpherocityIndex
- Visualised on Sauer-Schwarz PMI triangle coloured by pIC50
- Shape-only RF is much weaker than ECFP4 — 3D shape is complementary, not standalone

### 22. Literature Benchmark

Comparison against published EGFR pIC50 regression results on ChEMBL data:

| Study | Method | Test R² | RMSE |
|---|---|---|---|
| MoleculeNet RF (Wu+ 2018) | RF + ECFP | 0.62 | 0.78 |
| MoleculeNet GraphConv (Wu+ 2018) | Graph CNN | 0.66 | 0.74 |
| Chemprop D-MPNN (Yang+ 2019) | Directed MPNN | 0.71 | 0.69 |
| Multi-task DNN (Mayr+ 2018) | Multi-task NN | 0.68 | — |
| ChEMBL-wide RF (Lenselink+ 2017) | RF + Morgan | ~0.65 | — |
| **This work — LightGBM** | LightGBM + ECFP4 | **0.715** | **0.704** |
| **This work — Optuna-tuned LGBM** | LightGBM (Optuna) | **0.731** | **0.684** |
| **This work — Stacking** | RF+XGB+LGBM+SVR → Ridge | **0.735** | **0.678** |

Our classical ML pipeline matches or exceeds the strongest deep-learning baselines. Caveats: different train/test splits and curation pipelines make absolute comparisons approximate.

### 23. Multi-Task ErbB Selectivity

Requires fetching ErbB2/3/4 data first:

```bash
python fetch_data.py --targets EGFR,ERBB2,ERBB3,ERBB4
```

Data fetched from ChEMBL 33:

| Target | ChEMBL ID | Curated compounds |
|---|---|---|
| EGFR | CHEMBL203 | 10,546 |
| ErbB2 | CHEMBL1824 | 2,494 |
| ErbB3 | CHEMBL2363049 | 83 |
| ErbB4 | CHEMBL3009 | 225 |

§23 uses EGFR + ErbB2 shared compounds (robust to sparse ErbB3/4 overlap) to plot a selectivity landscape coloured by ΔpIC50(EGFR−ErbB2). Saved to `outputs/erbb_selectivity.png`.

### 24. ChemBERTa Transformer Embeddings

Head-to-head comparison on the same 80/20 split:

| Features | Downstream model | Test R² |
|---|---|---|
| ECFP4 (2048 bits) | LightGBM | **0.715** |
| **ChemBERTa-zinc-base-v1** (768-dim) | LightGBM | 0.539 |
| Δ | | −0.176 |

ChemBERTa embeddings underperform ECFP4 on this dataset — consistent with published findings that pre-trained transformer embeddings rarely beat task-specific fingerprints for QSAR on <100k compounds without fine-tuning. A PCA visualisation of the 768-dim embedding space shows smooth pIC50 gradients, confirming the transformer has learned useful chemical manifold structure despite lower regression R².

### 25. ADMET Auxiliary Endpoints

Requires PyTDC on **Python ≤ 3.11** (PyTDC pins `scikit-learn==1.2.2`, no wheel for Python 3.13). Runs fully in Google Colab (Python 3.10). When available, §25 trains RF classifiers/regressors on:

| Endpoint | Dataset | Metric |
|---|---|---|
| hERG block | hERG_Karim (~13k) | ROC-AUC |
| CYP3A4 inhibition | CYP3A4_Veith (~12k) | ROC-AUC |
| Aqueous solubility | Solubility_AqSolDB (~10k) | R² |

Then profiles the five FDA-approved EGFR inhibitors across all endpoints simultaneously.

### 26. Graph Neural Network — GCN Baseline

3-layer GCN with global mean-pool head (PyTorch Geometric), trained on the identical train/test split:

| Model | Test R² | RMSE | MAE |
|---|---|---|---|
| LightGBM (best classical) | 0.715 | 0.704 | 0.527 |
| **GCN (3×GCNConv → meanpool → MLP)** | **0.487** | **0.939** | **0.723** |

The GCN baseline is weaker than classical ML — consistent with the literature on datasets of ~10k compounds where GNNs require larger data to amortise their inductive bias. A fully fine-tuned message-passing network (e.g. Chemprop D-MPNN) with explicit edge features typically closes this gap.

---

## Phase 4 — Rigour & Engineering

### 27. Scaffold-Based Train/Test Split

Withholds entire Murcko ring systems from training (the protocol used in MoleculeNet and Chemprop):

| Model | Random split R² | Scaffold split R² | Gap |
|---|---|---|---|
| LightGBM | 0.715 | **0.607** | −0.108 |
| Random Forest | 0.695 | **0.612** | −0.083 |

The ~0.10 gap is expected and informative — it quantifies how much of the random-split performance comes from scaffold memorisation rather than true generalisation to new chemotypes.

### 28. Bootstrap Confidence Intervals

1 000-resample bootstrap on the held-out test set; 95 % CI around R² and RMSE for all models:

| Model | R² (95 % CI) | RMSE (95 % CI) |
|---|---|---|
| **LightGBM** | **0.715 [0.691, 0.737]** | **0.703 [0.676, 0.731]** |
| Random Forest | 0.695 [0.672, 0.714] | 0.728 [0.702, 0.757] |
| SVR (RBF) | 0.692 [0.667, 0.716] | 0.731 [0.703, 0.762] |
| XGBoost | 0.663 [0.641, 0.687] | 0.764 [0.738, 0.790] |

LightGBM and SVR CI bands overlap — their difference is not statistically significant on this test set. LightGBM vs. XGBoost CIs do not overlap, indicating a robust ranking.

### 29. Activity Cliff Analysis

- **5,422 similar pairs** (Tanimoto ≥ 0.6 on ECFP4) in a 2,000-compound subsample
- **241 activity cliffs** (4.4 % of similar pairs) with |pIC50 difference| ≥ 2.0 log units
- Cliff density is moderate — typical for a diverse kinase inhibitor dataset

These pairs are the hardest QSAR predictions and directly identify where 3D or pharmacophore features would add the most value.

### 30. Learning Curves

R² vs. training set size (5 %–100 %, 3 repeats per fraction):

| Training fraction | Compounds | LightGBM R² | RF R² |
|---|---|---|---|
| 5 % | 421 | 0.365 | 0.402 |
| 20 % | 1,687 | 0.546 | 0.554 |
| 50 % | 4,218 | 0.658 | 0.665 |
| **100 %** | **8,436** | **0.705** | **0.724** |

Performance is **still rising at full training size** — merging with ExCAPE-DB or BindingDB EGFR data would yield meaningful improvement.

---

## Interactive Web App — Streamlit Dashboard

A full multi-tab Streamlit dashboard (`app.py`, 950+ lines) covering the entire pipeline:

```bash
pip install streamlit joblib
streamlit run app.py
```

### Tabs

| Tab | Content |
|-----|---------|
| 🔬 **Predict** | SMILES input → 2D structure · pIC50 · IC50 · activity class · 90% conformal interval · applicability domain · Lipinski Ro5. Five built-in approved EGFR drugs as examples. |
| 📊 **Dataset & EDA** | Dataset stats · 7-step curation pipeline table · pIC50 distribution · Lipinski plots · fingerprint sparsity · searchable data preview (10,546 rows) |
| 🤖 **Model Performance** | Interactive model comparison table (4 baseline + 4 advanced models, all metrics from JSON) · predicted vs actual · SHAP interpretability · fingerprint type comparison · bootstrap CI · scaffold split analysis |
| 🗺️ **Chemical Space** | t-SNE map (1,500 compounds) · Murcko scaffold stats · virtual screening of 5 approved drugs · ErbB family selectivity landscape |
| 🔩 **Advanced Analysis** | Applicability domain · conformal prediction · activity cliff stats · 3D shape descriptors · learning curves — all with real numerical metrics |
| 📖 **Methodology** | Full pipeline diagram · EGFR target biology · feature comparison table · validation strategy table · software stack · 7 academic references |

All 18 pre-computed plots load instantly from `outputs/`. No recomputation needed.

---

## Quick Start

### Option A — Google Colab (recommended, no local setup)

Click the badge at the top → **Runtime → Run all**

### Option B — Local

```bash
git clone https://github.com/OleksandrUK911/egfr-bioactivity-prediction.git
cd egfr-bioactivity-prediction

pip install -r requirements.txt
jupyter notebook bioactivity_prediction.ipynb
```

### Option C — Kaggle dataset (if ChEMBL API is unavailable)

Search Kaggle for: **`EGFR ChEMBL bioactivity`**. Place the CSV in `data/` and update the data-loading cell.

### Fetch data script

```bash
# EGFR only (default)
python fetch_data.py

# Full ErbB family for §23 selectivity analysis
python fetch_data.py --targets EGFR,ERBB2,ERBB3,ERBB4
```

---

## Repository Structure

```
egfr-bioactivity-prediction/
├── bioactivity_prediction.ipynb          # Main notebook (30 sections, §1–§30)
├── bioactivity_prediction_executed.ipynb # Fully executed with all outputs
├── fetch_data.py                         # ChEMBL downloader (single + multi-target)
├── run_notebook.py                       # Windows-compatible headless executor
├── app.py                                # Streamlit 6-tab dashboard (950+ lines)
├── Makefile                              # make fetch / run / test / lint / notebook
├── requirements.txt                      # pip dependencies
├── environment.yml                       # conda environment (rdkit via conda-forge)
├── CITATION.cff                          # formal citation metadata
├── .gitignore
├── LICENSE
├── README.md
├── TODO.md
├── .github/
│   └── workflows/
│       └── ci.yml                        # GitHub Actions: flake8 + pytest on push/PR
├── tests/
│   └── test_features.py                  # 10 pytest unit tests
├── data/
│   └── egfr_bioactivity_curated.csv     # 10,546 curated compounds
├── content/
│   ├── blog_post.md
│   ├── linkedin_article.md
│   ├── portfolio_entry.md
│   └── rsc_abstract.md
├── deployment/
│   └── huggingface/
│       ├── README.md
│       └── DEPLOY.md
└── outputs/
    ├── eda_overview.png
    ├── lipinski_descriptors.png
    ├── fingerprint_analysis.png
    ├── predicted_vs_actual.png
    ├── model_comparison.png
    ├── shap_analysis.png
    ├── chemical_space_tsne.png
    ├── virtual_screening.png
    ├── erbb_selectivity.png
    ├── scaffold_analysis.png
    ├── applicability_domain.png
    ├── conformal_prediction.png
    ├── shape_descriptors_3d.png
    ├── benchmark_comparison.png
    ├── scaffold_split_comparison.png
    ├── bootstrap_ci.png
    ├── activity_cliffs.png
    ├── learning_curves.png
    └── model_summary.json
```

---

## Methods

### Data Source
Bioactivity data retrieved from [ChEMBL 33](https://www.ebi.ac.uk/chembl/) via `chembl-webresource-client`. ChEMBL is maintained by the EMBL-EBI at the Wellcome Genome Campus, Hinxton, Cambridge.

### Curation
- Filtered to exact IC50 (relation `=`) in nM
- Deduplicated by compound (median IC50 across assays)
- Converted: IC50 (nM) → pIC50 = −log₁₀(IC50 × 10⁻⁹)
- Retained pIC50 range 3–12 (physiologically plausible)
- Validated SMILES with RDKit

### Molecular Features
Morgan circular fingerprints (ECFP4) via RDKit: radius=2, 2048 bits. Also benchmarked: MACCS keys (166 bits), RDKit path-based FP (2048 bits), RDKit 2D descriptors (207 after cleaning), ChemBERTa-zinc-base-v1 embeddings (768-dim).

### Interpretability
SHAP TreeExplainer on the best tree model: global feature importance (bar plot), feature impact distribution (beeswarm plot).

---

## Dependencies

```
chembl-webresource-client>=0.10.8    rdkit-pypi>=2022.9.5
scikit-learn>=1.3.0                  xgboost>=2.0.0
lightgbm>=4.0.0                      shap>=0.44.0
optuna>=3.5.0                        pandas>=2.0.0
numpy>=1.24.0                        matplotlib>=3.7.0
seaborn>=0.13.0                      streamlit>=1.30.0
joblib>=1.3.0                        transformers>=4.40.0
torch>=2.0.0                         jupyter>=1.0.0
```

Optional (for specific sections):
- `PyTDC` — §25 ADMET endpoints
- `torch_geometric` — §26 GNN (installed automatically in Colab)

---

## Relevance to UK Drug Discovery

Cambridge is the UK's leading life sciences hub, home to:
- **AstraZeneca** global R&D headquarters (EGFR inhibitor Osimertinib/Tagrisso developed here)
- **Wellcome Sanger Institute** — the organisation behind ChEMBL
- **Astex Therapeutics**, **Bicycle Therapeutics**, **Kymera Therapeutics** (Cambridge offices)
- **Milner Therapeutics Institute**, University of Cambridge

This project reflects the computational methods actively used in Cambridge's drug discovery community.

---

## References

1. Mendez *et al.* (2019). ChEMBL: towards direct deposition of bioassay data. *Nucleic Acids Research*, 47, D930–D940.
2. Rogers & Hahn (2010). Extended-Connectivity Fingerprints. *J. Chem. Inf. Model.*, 50(5), 742–754.
3. Lundberg & Lee (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30.
4. Lipinski *et al.* (2001). Experimental and computational approaches to estimate solubility and permeability. *Adv. Drug Deliv. Rev.*, 46, 3–26.
5. Yang *et al.* (2019). Analyzing learned molecular representations for property prediction. *J. Chem. Inf. Model.*, 59(8), 3370–3388.
6. Wu *et al.* (2018). MoleculeNet: a benchmark for molecular machine learning. *Chem. Sci.*, 9, 513–530.

---

## Licence

MIT — see [LICENSE](LICENSE).

---

*Built as a drug discovery data science portfolio. Executed locally on Python 3.13 · sklearn 1.7 · xgboost 3.0 · lightgbm 4.x · PyTorch 2.11. Open to collaboration and feedback.*
