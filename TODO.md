# TODO — EGFR Bioactivity Prediction Project

Drug discovery ML portfolio project. All results below are from real execution.

---

## Completed

### Data & Infrastructure
- [x] Set up Google Colab notebook structure (30 sections, §1–§30)
- [x] Install all dependencies (RDKit, ChEMBL client, SHAP, XGBoost, LightGBM)
- [x] Connect to ChEMBL REST API and retrieve EGFR IC50 data
- [x] Curate raw bioactivity data: 25,758 raw → 10,546 curated (filter, dedup, validate SMILES)
- [x] Convert IC50 → pIC50 with physiological range filter (3–12)
- [x] Save curated dataset to `data/egfr_bioactivity_curated.csv`
- [x] `fetch_data.py` — standalone downloader with CSV caching and multi-target support
- [x] `run_notebook.py` — Windows asyncio + UTF-8 executor for headless notebook runs

### Feature Engineering
- [x] Morgan fingerprints (ECFP4, radius=2, 2048 bits)
- [x] Lipinski physicochemical descriptors (MW, LogP, HBD, HBA, TPSA)
- [x] Lipinski Rule of Five compliance check

### Exploratory Data Analysis
- [x] pIC50 distribution histogram + activity class breakdown
- [x] Lipinski descriptor distributions by activity class
- [x] Bit density analysis of ECFP4 fingerprints

### Modelling (Phase 1)
- [x] Train/test split (80/20, stratified by activity class)
- [x] Random Forest Regressor — R²=0.695, RMSE=0.728
- [x] XGBoost Regressor — R²=0.664, RMSE=0.764
- [x] LightGBM Regressor — R²=0.715, RMSE=0.704 (best individual)
- [x] SVR with RBF kernel (Pipeline with StandardScaler) — R²=0.692, RMSE=0.732
- [x] 5-fold cross-validation for all models
- [x] Test set evaluation: R², RMSE, MAE

### Visualisation & Interpretability
- [x] Predicted vs actual scatter + residual plot (best model)
- [x] Model comparison bar chart (R² and RMSE)
- [x] SHAP TreeExplainer — global summary bar plot + beeswarm plot
- [x] t-SNE chemical space visualisation (PCA 2048→50 → t-SNE 50→2)
- [x] Virtual screening: score 5 approved EGFR inhibitors with Ro5 compliance

### Phase 2 — Advanced Analyses
- [x] §14 Fingerprint comparison: RDKit FP 0.696 > ECFP4 0.690 > MACCS 0.621 (CV R²)
- [x] §15 Murcko scaffold analysis: 3,857 unique scaffolds, 36.6 % diversity
- [x] §16 Applicability domain (k-NN, k=5): 96.1 % in-domain, R²_in=0.714
- [x] §17 Ensemble stacking (Ridge meta-learner): R²=0.735, RMSE=0.678, MAE=0.503

### Phase 3 — Extended Methods
- [x] §18 Optuna HPO (50 trials, TPE, LightGBM): tuned R²=0.731, RMSE=0.684, MAE=0.507
- [x] §19 RDKit 2D descriptors (207 after cleaning): ECFP4+2D R²=0.718 (+0.003 vs ECFP4 only)
- [x] §20 Conformal prediction: 90.9 % empirical coverage at 90 % target, ±1.251 pIC50 interval
- [x] §21 3D conformer shape descriptors (NPR1/2, asphericity, RoG) for 200 sampled molecules
- [x] §22 Literature benchmark figure vs MoleculeNet, Chemprop, Mayr, Lenselink
- [x] §23 ErbB family selectivity — EGFR vs ErbB2 scatter on shared compounds; `data/multitarget_bioactivity.csv` (EGFR=10,546 · ErbB2=2,494 · ErbB3=83 · ErbB4=225 from ChEMBL 33)
- [x] §24 ChemBERTa-zinc-base-v1 embeddings: R²=0.539 (ECFP4 R²=0.715 — fingerprints win)
- [x] §26 GNN baseline (3×GCNConv → meanpool → MLP): R²=0.487, RMSE=0.939

### Application & Deployment
- [x] `app.py` — Streamlit 6-tab dashboard (Predict · Dataset · Models · Chemistry · Advanced · Methodology)
- [x] `deployment/huggingface/` — deploy instructions for Hugging Face Spaces
- [x] Model saved to `outputs/best_model.joblib` (auto-trained on first app launch)

### Documentation & Content
- [x] `README.md` — pipeline overview, all actual results, methods, references
- [x] `model_summary.json` — all model metrics (Phases 1–3 incl. GCN)
- [x] `content/blog_post.md` — Medium/Towards Data Science article draft
- [x] `content/linkedin_article.md` — LinkedIn article for UK pharma/biotech audience
- [x] `content/portfolio_entry.md` — portfolio website entry
- [x] `content/rsc_abstract.md` — RSC Chemical Biology abstract draft
- [x] `requirements.txt` with pinned package versions
- [x] `LICENSE` (MIT)
- [x] Replace placeholder `YOUR_USERNAME` with real GitHub username (OleksandrUK911)

### Repository Hygiene & Engineering
- [x] `.gitignore` — excludes `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, model `.joblib`, logs
- [x] `environment.yml` — conda environment (rdkit installed via conda-forge, not pip)
- [x] `Makefile` — `make fetch`, `make run`, `make test`, `make lint`, `make notebook`
- [x] `.github/workflows/ci.yml` — GitHub Actions CI: lint (flake8) + pytest on push/PR
- [x] `tests/test_features.py` — 10 pytest unit tests (fingerprints, Lipinski, pIC50, curation)
- [x] `CITATION.cff` — formal citation metadata (CFF 1.2.0 standard)

### Phase 4 — Rigour & Engineering (notebook §27–30)
- [x] §27 Scaffold-based train/test split — LGBM R²=0.607 scaffold vs 0.715 random (gap = −0.108, expected)
- [x] §28 Bootstrap CI (1 000 resamples) — LightGBM 95% CI: R²=[0.691, 0.737] RMSE=[0.676, 0.731]
- [x] §29 Activity cliff analysis — 241 cliffs (4.4% of 5,422 similar pairs with Tanimoto ≥ 0.6)
- [x] §30 Learning curves — LGBM R² still rising at 100% data (0.365→0.705); RF similar trend

---

## Remaining

| # | Priority | Task | Notes |
|---|---|---|---|
| 1 | High | **Upload repo to GitHub + push all files** | ✅ Ready — all files prepared, .gitignore verified |
| 2 | Low | **Deploy Streamlit app to Hugging Face Spaces** | Instructions in `deployment/huggingface/DEPLOY.md` |

### Notes on skipped sections

- **§25 ADMET (PyTDC)** — PyTDC pins `scikit-learn==1.2.2`; no pre-built wheel for Python 3.13. Runs fully in Google Colab (Python 3.10). Cell gracefully skips when PyTDC is absent.

---

## Known Issues — Fixed

| # | Issue | Fix |
|---|---|---|
| 1 | ChEMBL API slow / timeout for 25k records | `fetch_data.py` caches CSV; notebook loads from file |
| 2 | `standard_relation == "'='"` filtering all records | Changed to `== '='` (ChEMBL API returns plain `=`) |
| 3 | `sklearn.TSNE n_iter` deprecation (≥1.5) | Version-aware `tsne_kwargs` dict with `max_iter` |
| 4 | t-SNE slow on >2000 compounds | PCA(50) pre-reduction before t-SNE |
| 5 | SHAP slow for SVR | Skipped SVR SHAP; TreeExplainer on tree models only |
| 6 | `Pipeline` clone error in stacking | `sklearn.base.clone(model_proto)` instead of `type(model)(**params)` |
| 7 | Windows asyncio + zmq `RuntimeError` | `asyncio.WindowsSelectorEventLoopPolicy()` in `run_notebook.py` |
| 8 | `UnicodeDecodeError` reading notebook | `open(..., encoding='utf-8')` |
| 9 | Cell 55 source corrupted with `Й   1'` prefix | `_fix_corruption.py` — `re.sub(r'^[^\x00-\x7F\s\d\'\"]+[\s\d\'\"]*', '', src)` |

---

## Results Summary

| Section | Method | R² | RMSE | Notes |
|---|---|---|---|---|
| §7 | Random Forest | 0.695 | 0.728 | |
| §7 | XGBoost | 0.664 | 0.764 | |
| §7 | SVR (RBF) | 0.692 | 0.732 | |
| §7 | **LightGBM** | **0.715** | **0.704** | Best individual |
| §17 | Stacking (Ridge) | 0.735 | 0.678 | Best classical |
| §18 | LightGBM (Optuna) | 0.731 | 0.684 | Tuned |
| §19 | RF (ECFP4+2D) | 0.718 | — | +0.003 vs fingerprint-only |
| §24 | ChemBERTa+LightGBM | 0.539 | — | Transformer worse |
| §26 | GCN (PyTorch Geometric) | 0.487 | 0.939 | GNN baseline |
| §27 | LightGBM (scaffold split) | 0.607 | 0.878 | vs 0.715 random — gap shows scaffold memorisation |
| §28 | Bootstrap CI (LightGBM) | [0.691, 0.737] | — | 95% CI, 1 000 resamples |
| §29 | Activity cliffs | — | — | 241 cliffs / 5,422 similar pairs (4.4%) |
| §30 | Learning curves | 0.705 at 100% | — | Still rising → more data would help |

---

*Last updated: May 2026 · Python 3.13 · sklearn 1.7 · xgboost 3.0 · lightgbm 4.x · PyTorch 2.11*
