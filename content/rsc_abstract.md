# RSC Chemical Biology / Next Generation Researchers — abstract

Cambridge, UK · Poster / short-talk submission · ~250 words.

---

## Title

**An interpretable, uncertainty-aware QSAR pipeline for EGFR inhibitors:
from ChEMBL to a deployable web app**

## Author

Oleksandr — independent researcher (portfolio project)
GitHub: <https://github.com/OleksandrUK911/egfr-bioactivity-prediction>

## Abstract (≈ 250 words)

The Epidermal Growth Factor Receptor (EGFR) is one of the most extensively
profiled oncology targets, with several generations of clinically approved
small-molecule inhibitors developed in the Cambridge (UK) life-sciences
ecosystem. We present an open-source, end-to-end QSAR pipeline that turns
10,546 curated ChEMBL 33 IC50 records (pIC50 ∈ [3, 12]) into a deployable
predictor of EGFR bioactivity from a SMILES string.

Four baseline regressors — Random Forest, XGBoost, LightGBM and SVR — were
trained on Morgan / ECFP4 fingerprints (radius 2, 2048 bits) using an 80/20
activity-stratified split. LightGBM was the best single model
(test R² = 0.715, RMSE = 0.704); a Ridge-meta-learner stack of all four
base models reached **R² = 0.735**. Hyperparameters were tuned via 50-trial
Bayesian optimisation (Optuna). Fingerprint comparison (MACCS / RDKit FP /
ECFP4) and Murcko scaffold analysis (3,857 unique scaffolds, 36.6 %
diversity) are reported alongside.

To address trust and deployability, the pipeline implements (i) SHAP
TreeExplainer attributions on the best tree model, (ii) a k-nearest-
neighbour applicability domain in fingerprint space, and (iii) split
conformal prediction giving empirical 90 % coverage (median interval
width ≈ 1.4 pIC50). Phase 3 extensions add ChemBERTa transformer embeddings (R² = 0.539; ECFP4
fingerprints outperform at this dataset scale), a 3-layer GCN (PyTorch
Geometric, R² = 0.487), multi-task ErbB family selectivity modelling across
EGFR/ErbB2/3/4, and an ADMET profile via PyTDC.

A Phase 4 rigour suite quantifies evaluation reliability: scaffold-based
train/test splitting (Murcko grouping, MoleculeNet protocol) yields
R² = 0.607, revealing a generalisation gap of −0.108 vs. random splitting
that reflects realistic prospective screening performance. Bootstrap confidence
intervals (1 000 resamples) give R² ∈ [0.691, 0.737] (95% CI). Activity cliff
analysis identifies 241 structurally similar but potency-divergent pairs (4.4%
of Tanimoto ≥ 0.6 pairs), and learning curves confirm performance is still
improving at full training size, motivating future data augmentation.

The full pipeline is reproducible from a single 30-section notebook, deployed
as a Streamlit web application, and released under MIT licence to support open,
benchmark-comparable QSAR research in the UK academic community.

**Keywords:** QSAR, EGFR, applicability domain, conformal prediction,
SHAP, GNN, ChemBERTa, ADMET, open source.
