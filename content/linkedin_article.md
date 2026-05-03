# LinkedIn article — UK pharma / biotech audience

**Title:** *I built an end-to-end EGFR bioactivity predictor — here's what
30 ML sections taught me about real drug-discovery workflows*

**Suggested cover image:** the t-SNE chemical-space plot
(`outputs/chemical_space_tsne.png`) or the SHAP beeswarm
(`outputs/shap_analysis.png`).

---

EGFR is the target behind AstraZeneca's Tagrisso — developed right here in
Cambridge — and one of the most studied kinases in oncology. Over the last
few weeks I built a complete machine-learning pipeline around it, going from
raw ChEMBL bioactivity records to a deployable Streamlit app, to sharpen the
skills most relevant to UK pharma and biotech data-science roles.

**The headline numbers:**

📊 **10,546 curated EGFR IC50 records** from ChEMBL 33
🎯 **LightGBM R² = 0.715, RMSE = 0.704** on a held-out 20 % test set
🧩 **Stacked ensemble (RF + XGBoost + LightGBM + SVR) → R² = 0.735**
🧪 SHAP, Murcko scaffolds, k-NN applicability domain, split conformal
   prediction (90 % empirical coverage)
🤖 ChemBERTa embeddings + a 3-layer GCN (PyTorch Geometric) for comparison
💊 ADMET multi-objective profile (hERG, CYP3A4, solubility) of the approved
   EGFR inhibitors via PyTDC
🌐 Streamlit app + Hugging Face Spaces deployment

**Phase 4 — Rigour engineering (§27–30):**

🏗️ **Scaffold split R² = 0.607** (vs 0.715 random) — the realistic performance
   on novel chemotypes; a −0.108 gap that disappears in careless evaluations
📊 **Bootstrap 95% CI: R² ∈ [0.691, 0.737]** — statistically defensible model ranking
⚡ **241 activity cliffs** (4.4% of similar pairs) — the model's hardest cases
📈 **Learning curves still rising** at 100% data — more data = better model

**Three things I underestimated going in:**

1️⃣ **Curation matters more than the model.** A single wrong filter on
   `standard_relation` silently dropped the entire dataset. Logging row
   counts after every step is non-negotiable in cheminformatics.

2️⃣ **Stacking beats tuning.** 50 trials of Optuna on LightGBM bought ~+0.005
   R². Stacking four diverse base learners bought ~+0.020. The lesson: model
   diversity is cheaper than hyperparameter perfectionism.

3️⃣ **GNNs aren't free wins on small datasets.** On ~10 k single-target
   bioactivity records, ECFP4 + gradient boosting is still the right
   baseline. GNNs come into their own at multi-task / multi-target scale,
   or with self-supervised pre-training.

**Why this project, why Cambridge:** computational drug discovery in the UK
is heavily concentrated in the Cambridge/AstraZeneca/Babraham/MRC-LMB
ecosystem. A portfolio that mirrors the actual day-to-day of a
cheminformatics team — curation, multiple baselines, interpretability,
applicability domain, calibrated uncertainty, deployment — should be more
useful than a Kaggle leaderboard solo.

**Code (open source, MIT):**
🔗 https://github.com/OleksandrUK911/egfr-bioactivity-prediction

I'd genuinely value feedback from anyone working in QSAR, computational
chemistry, or ML for drug discovery — particularly on what would make this
more useful as a hiring signal vs a portfolio piece. Drop a comment or
DM me 👇

---

#DrugDiscovery #Cheminformatics #MachineLearning #QSAR #ComputationalChemistry
#PharmaAI #LifeSciences #CambridgeUK #PortfolioProject #OpenSource
