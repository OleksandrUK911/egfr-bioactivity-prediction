# From ChEMBL to Chemical Space: Building an End-to-End EGFR Bioactivity Predictor

*A 30-section ML pipeline that goes from raw ChEMBL bioactivity records to a
deployable Streamlit app — covering fingerprints, SHAP, applicability domain,
conformal prediction, stacking, ChemBERTa, ADMET multi-objective scoring,
a PyTorch Geometric GNN, and Phase 4 rigour analyses (scaffold splits, bootstrap
CIs, activity cliffs, learning curves).*

---

## TL;DR

| | |
|---|---|
| **Target** | EGFR (CHEMBL203) — Tagrisso/Osimertinib's clinical target |
| **Data** | 10,546 curated ChEMBL 33 IC50 records, pIC50 ∈ [3, 12] |
| **Best fingerprint model** | LightGBM, **R² = 0.715, RMSE = 0.704** |
| **Stacked ensemble** | RF + XGB + LGBM + SVR → Ridge meta-learner, **R² = 0.735** |
| **Interpretability** | SHAP (TreeExplainer), Murcko scaffolds, k-NN applicability domain |
| **Uncertainty** | Split conformal prediction (90 % coverage) |
| **Phase 3** | ChemBERTa, GCN (PyTorch Geometric), ADMET (PyTDC), ErbB selectivity |
| **App** | Streamlit + Hugging Face Spaces — paste SMILES, get pIC50 + Ro5 |
| **Repo** | <https://github.com/OleksandrUK911/egfr-bioactivity-prediction> |

---

## Why EGFR?

EGFR is one of the most clinically validated oncology targets in the world.
Five generations of inhibitors — from Gefitinib (2003) through Erlotinib,
Lapatinib, Afatinib, to Osimertinib (Tagrisso, 2015) — have reshaped non-small
cell lung cancer (NSCLC) treatment. Osimertinib was developed at AstraZeneca's
Cambridge (UK) site, which made it a natural target choice for a portfolio
project aimed at the Cambridge life-sciences ecosystem.

It is also a gift for ML: ChEMBL alone has tens of thousands of bioactivity
records, and the chemical diversity is high enough that the dataset behaves
like a real-world drug-discovery setting rather than a toy benchmark.

## The pipeline at a glance

```
ChEMBL API → curation → ECFP4 + Lipinski Ro5 → 4 baselines + SHAP →
t-SNE chemical space → virtual screening → Phase 2 (FP comparison,
scaffolds, AD, Optuna, stacking, conformal, RDKit 2D, 3D conformers)
→ Phase 3 (ChemBERTa, multi-task ErbB, ADMET, GCN) → Streamlit app
```

The whole thing is a single notebook with **30 sections**, each saving its
figures to `outputs/` and writing metrics to `model_summary.json`. A separate
`app.py` exposes the best model as a Streamlit web app.

## Section-by-section highlights

### 1–7 — The boring (but important) parts

ChEMBL REST → IC50/nM filter → SMILES validation → median-aggregate by
compound → pIC50 ∈ [3, 12] window. Then ECFP4 (radius 2, 2048 bits) +
Lipinski descriptors. **80/20 stratified split** by activity class.

A pitfall worth knowing: in current `chembl_webresource_client`, the
`standard_relation` field is the **literal string `=`**, not the older
``"'='"``. One incorrect filter and you silently drop your entire dataset.

### 8–9 — Diagnostics + SHAP

Predicted-vs-actual scatter + residual histogram. SHAP with TreeExplainer on
LightGBM produces a clean beeswarm where the most important features are
quinazoline-core fingerprint bits — the classical Gefitinib/Erlotinib
chemotype. Sanity check passed.

### 10 — t-SNE chemical space

PCA(50) → t-SNE on Tanimoto-friendly bit vectors. The map shows clean
separation between the active and inactive clouds, with the known approved
drugs landing where you'd expect.

### 14–17 — Phase 2 modelling

| Section | What it does | Result |
|---|---|---|
| 14 | MACCS / RDKit FP / ECFP4 head-to-head | RDKit FP narrowly wins (CV R² 0.696) |
| 15 | Murcko scaffolds | 3,857 unique scaffolds → 36.6 % diversity, healthy |
| 16 | Applicability domain (k-NN distance) | 95th-percentile flag for "out of domain" predictions |
| 17 | Stacking (Ridge meta-learner) | **R² = 0.735** — small but real lift |

### 18–22 — Optuna, RDKit 2D, conformal prediction, 3D shape, benchmarks

Optuna (50 trials Bayesian) tuned LightGBM. RDKit's ~200 2D descriptors did
*not* beat ECFP4 alone but stacked with it. Split conformal gives empirical
90 % coverage with median interval width ≈ 1.4 pIC50 units — interpretable
uncertainty for downstream triage.

### 23–25 — Phase 3 expanded scope

* **§23 Multi-task ErbB.** EGFR + ErbB2/3/4 from ChEMBL, Pareto front on
  shared molecules, identifies selectivity outliers.
* **§24 ChemBERTa.** Pre-trained transformer embeddings on SMILES; R² = 0.539
  vs LightGBM's 0.715 — significantly underperforms fingerprints at this scale.
  Pre-training on generic ZINC chemistry doesn't transfer well to the narrow
  kinase-inhibitor chemical series. Fine-tuning or a larger dataset would close
  the gap.
* **§25 ADMET (PyTDC).** Predict hERG, CYP3A4, aqueous solubility for the
  approved inhibitors and overlay on EGFR potency for a multi-objective
  drug-profile chart.

### 27–30 — Phase 4: Rigour engineering

| Section | What it does | Key result |
|---|---|---|
| 27 | Scaffold-based split | R² = 0.607 (vs 0.715 random) — gap of −0.108 |
| 28 | Bootstrap CI (1 000 resamples) | 95% CI: R² ∈ [0.691, 0.737] |
| 29 | Activity cliff analysis | 241 cliffs / 5,422 similar pairs (4.4%) |
| 30 | Learning curves | R² still rising at 100% data → more data would help |

The scaffold split is the most important of the four: it shows the model's
*realistic* performance on novel chemotypes, which is what drug discovery actually
requires. The −0.108 gap is expected and healthy — not a failure.

### 26 — Graph Neural Network

A vanilla 3-layer GCN (PyTorch Geometric) on the same train/test split.
Comparable to fingerprints out-of-the-box — exactly what the literature
predicts at this dataset size. The point isn't to win the leaderboard; it's
to demonstrate you can drop a learned representation into the same evaluation
harness.

## The lessons that aren't in the README

1. **Curation eats most of the time.** I rewrote the IC50 filter three times
   before the dataset stopped silently shrinking. Always log row counts after
   every filter step.
2. **Stacking gives more than tuning.** Optuna's tuned LightGBM bought ~+0.005
   R². Stacking four diverse base learners bought ~+0.020.
3. **Applicability domain is the most under-rated cheminformatics tool.**
   The model that confidently predicts pIC50 = 8 for a steroid scaffold it
   has never seen before is the model that ends up in a paper retraction.
4. **GNNs aren't free wins.** On ~10 k single-target datasets, ECFP4 +
   gradient boosting is *still* the right baseline.

## Try it

* **Notebook (Colab):** click the "Open in Colab" badge in the repo —
  Runtime → Run all.
* **Streamlit app:** `streamlit run app.py` after `pip install -r requirements.txt`.
* **Hugging Face Space:** see [`deployment/huggingface/DEPLOY.md`](https://github.com/OleksandrUK911/egfr-bioactivity-prediction/blob/main/deployment/huggingface/DEPLOY.md).

## What I'd do next

* **D-MPNN / Chemprop** for the GNN — closer to current SOTA than vanilla GCN.
* **Active learning loop** on the applicability-domain edges — minimise the
  number of new assays needed to expand the model's safe operating range.
* **Cross-target transfer** from the ErbB family to less-studied kinases.

---

*If this is the kind of work that resonates with your team — particularly in
Cambridge, UK — I'd love to chat. Comments, questions, or "you got X wrong"
emails all welcome.*

**Code:** <https://github.com/OleksandrUK911/egfr-bioactivity-prediction>
