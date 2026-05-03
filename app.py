"""
EGFR Bioactivity Predictor — Streamlit dashboard.

Run locally:
    pip install streamlit joblib
    streamlit run app.py

Tabs:
  1. Predict      — SMILES → pIC50, IC50, activity class, Lipinski, AD check
  2. Dataset      — EDA plots, dataset stats, data preview, curation pipeline
  3. Models       — performance tables, plots, SHAP, benchmarks, advanced models
  4. Chemistry    — t-SNE, scaffolds, virtual screening, ErbB family selectivity
  5. Advanced     — AD, conformal intervals, activity cliffs, 3D shape, learning curves
  6. Methodology  — full pipeline explanation, data source, features, validation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, Lipinski

ROOT          = Path(__file__).parent
DATA_CSV      = ROOT / "data" / "egfr_bioactivity_curated.csv"
MODEL_PATH    = ROOT / "outputs" / "best_model.joblib"
OUTPUTS       = ROOT / "outputs"
N_BITS        = 2048
MORGAN_RADIUS = 2
RANDOM_SEED   = 42
AD_K          = 5
CONFORMAL_HALF_WIDTH = 1.251


# ---------------------------------------------------------------------------
# Featurisation helpers
# ---------------------------------------------------------------------------
def smiles_to_morgan(smiles: str, radius: int = MORGAN_RADIUS,
                     n_bits: int = N_BITS) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def lipinski_report(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw         = Descriptors.MolWt(mol)
    logp       = Crippen.MolLogP(mol)
    hbd        = Lipinski.NumHDonors(mol)
    hba        = Lipinski.NumHAcceptors(mol)
    tpsa       = Descriptors.TPSA(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {
        "MW (≤500)":        round(mw, 2),
        "LogP (≤5)":        round(logp, 2),
        "HBD (≤5)":         hbd,
        "HBA (≤10)":        hba,
        "TPSA (Å²)":        round(tpsa, 2),
        "Violations":       violations,
        "Ro5 pass":         "✅ Yes" if violations <= 1 else "❌ No",
    }


def activity_class(pic50: float) -> str:
    if pic50 >= 6.0:
        return "Active"
    if pic50 >= 5.0:
        return "Moderate"
    return "Inactive"


def tanimoto_knn_distance(query_fp: np.ndarray,
                           train_fps: np.ndarray,
                           k: int = AD_K) -> float:
    inter  = train_fps @ query_fp
    sum_q  = float(query_fp.sum())
    sum_t  = train_fps.sum(axis=1)
    union  = sum_q + sum_t - inter
    sim    = np.where(union > 0, inter / union, 0.0)
    top_k  = np.partition(sim, -k)[-k:]
    return float(1.0 - top_k.mean())


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def get_model():
    import joblib
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    if not DATA_CSV.exists():
        st.error(
            f"Neither model ({MODEL_PATH}) nor data ({DATA_CSV}) found. "
            "Run the notebook first."
        )
        st.stop()
    import lightgbm as lgb
    st.warning(f"No saved model at `{MODEL_PATH}`. Training LightGBM — one-off, ~30 s.")
    df = pd.read_csv(DATA_CSV).dropna(subset=["canonical_smiles", "pIC50"])
    X, y = [], []
    for smi, p in zip(df["canonical_smiles"], df["pIC50"]):
        fp = smiles_to_morgan(smi)
        if fp is not None:
            X.append(fp); y.append(p)
    X, y = np.vstack(X), np.array(y)
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, random_state=RANDOM_SEED, verbose=-1,
    )
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


@st.cache_resource(show_spinner="Computing applicability domain…")
def get_train_fps() -> tuple[np.ndarray, float] | tuple[None, None]:
    if not DATA_CSV.exists():
        return None, None
    df  = pd.read_csv(DATA_CSV).dropna(subset=["canonical_smiles", "pIC50"])
    fps = [smiles_to_morgan(s) for s in df["canonical_smiles"]]
    fps = [f for f in fps if f is not None]
    X   = np.vstack(fps).astype(np.uint8)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X), size=min(500, len(X)), replace=False)
    dists = []
    for i in idx:
        q      = X[i].astype(float)
        others = np.delete(X[idx], np.where(idx == i)[0], axis=0).astype(float)
        if len(others) >= AD_K:
            dists.append(tanimoto_knn_distance(q, others, k=AD_K))
    threshold = float(np.percentile(dists, 95)) if dists else 0.6
    return X.astype(float), threshold


@st.cache_data
def load_model_summary() -> dict:
    import json
    p = OUTPUTS / "model_summary.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    if DATA_CSV.exists():
        return pd.read_csv(DATA_CSV)
    return None


def plot_img(filename: str, caption: str = ""):
    p = OUTPUTS / filename
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.info(f"Plot not yet generated: `outputs/{filename}`  \nRun the notebook to produce it.")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EGFR Bioactivity Predictor",
    page_icon="🧬",
    layout="wide",
)

summary = load_model_summary()
best    = summary.get("best_model", {})
ds      = summary.get("dataset", {})
p1      = summary.get("phase1_baseline", {})
p2      = summary.get("phase2_advanced", {})
p3      = summary.get("phase3_extended", {})
p4      = summary.get("phase4_rigour", {})

st.title("🧬 EGFR Bioactivity Predictor")
st.caption(
    "Machine-learning QSAR pipeline for EGFR inhibitors · "
    "LightGBM + ECFP4 · 10,546 ChEMBL 33 compounds · "
    f"R² = {p1.get('LightGBM', {}).get('R²_test', 0.715)} (random split) · "
    f"{p4.get('scaffold_split_LightGBM', {}).get('R²_test', 0.607)} (scaffold split)"
)

EXAMPLES = {
    "Erlotinib (1st gen)":   "COCCOC1=C(OCCO)C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C",
    "Gefitinib (1st gen)":   "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Osimertinib (3rd gen)": "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1",
    "Lapatinib (2nd gen)":   "CS(=O)(=O)CCNCc1oc(-c2ccc(Nc3ncnc4cc(OCc5cccc(F)c5)c(OCC)cc34)cc2Cl)cc1",
    "Afatinib (2nd gen)":    "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1",
}

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_eda, tab_models, tab_chem, tab_adv, tab_method = st.tabs([
    "🔬 Predict",
    "📊 Dataset & EDA",
    "🤖 Model Performance",
    "🗺️ Chemical Space",
    "🔩 Advanced Analysis",
    "📖 Methodology",
])


# ============================================================
# TAB 1 — Predict
# ============================================================
with tab_predict:
    with st.sidebar:
        st.header("Input")
        example_pick = st.selectbox(
            "Load example drug",
            options=["— none —"] + list(EXAMPLES.keys()),
            index=0,
        )
        default_smiles = EXAMPLES.get(example_pick, "")
        smiles = st.text_area(
            "SMILES string",
            value=default_smiles,
            height=120,
            placeholder="Paste SMILES here…",
        )
        submit = st.button("Predict", type="primary", use_container_width=True)
        st.markdown("---")
        st.markdown(
            "**Model:** LightGBM + ECFP4 (r=2, 2048 bits)  \n"
            "**R² test:** 0.715 (random) · 0.607 (scaffold)  \n"
            "**Uncertainty:** ±1.25 pIC50 @ 90% confidence  \n\n"
            "⚠️ Portfolio demo only.  \n"
            "**Not for clinical use.**"
        )

    if submit and smiles.strip():
        smi = smiles.strip()
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            st.error("❌ Invalid SMILES — RDKit could not parse the input.")
            st.stop()

        fp      = smiles_to_morgan(smi)
        model   = get_model()
        pic50   = float(model.predict(fp.reshape(1, -1))[0])
        ic50_nm = 10 ** (9 - pic50)
        ci_lo   = pic50 - CONFORMAL_HALF_WIDTH
        ci_hi   = pic50 + CONFORMAL_HALF_WIDTH
        ro5     = lipinski_report(smi)

        train_fps, ad_threshold = get_train_fps()
        if train_fps is not None:
            ad_dist   = tanimoto_knn_distance(fp.astype(float), train_fps, k=AD_K)
            in_domain = ad_dist <= ad_threshold
        else:
            ad_dist, in_domain = None, None

        col_left, col_right = st.columns([1, 1.5])

        with col_left:
            st.subheader("Molecular Structure")
            img = Draw.MolToImage(mol, size=(340, 340))
            st.image(img)
            st.code(Chem.MolToSmiles(mol), language=None)

        with col_right:
            st.subheader("Bioactivity Prediction")

            cls   = activity_class(pic50)
            badge = {"Active": "🟢", "Moderate": "🟡", "Inactive": "🔴"}[cls]
            m1, m2, m3 = st.columns(3)
            m1.metric("pIC50",         f"{pic50:.2f}")
            m2.metric("IC50",          f"{ic50_nm:,.0f} nM")
            m3.metric("Activity",      f"{badge} {cls}")

            st.markdown(
                f"**90% Prediction Interval:** pIC50 ∈ **[{ci_lo:.2f}, {ci_hi:.2f}]**  "
                f"→  IC50 ∈ [{10**(9-ci_hi)/1000:.2f} – {10**(9-ci_lo)/1000:.2f}] µM  "
                f"*(empirical coverage 90.9%, split-conformal)*"
            )

            with st.expander("ℹ️ What does pIC50 mean?"):
                st.markdown(
                    "**pIC50** = −log₁₀(IC50 in mol/L) — a logarithmic measure of potency.\n\n"
                    "| pIC50 | IC50 | Interpretation |\n"
                    "|-------|------|----------------|\n"
                    "| ≥ 9   | ≤ 1 nM | Extremely potent |\n"
                    "| 8     | 10 nM | Very potent |\n"
                    "| 7     | 100 nM | Potent |\n"
                    "| 6     | 1 µM | Active threshold |\n"
                    "| 5     | 10 µM | Weakly active |\n"
                    "| < 5   | > 10 µM | Inactive |\n\n"
                    "Most approved kinase inhibitors have pIC50 ≥ 8 against their primary target."
                )

            st.markdown("---")
            st.subheader("Applicability Domain")
            if ad_dist is not None:
                sim_pct = max(0.0, 1.0 - ad_dist) * 100
                st.progress(min(1.0, sim_pct / 100),
                            text=f"Structural similarity to training set: {sim_pct:.1f}%")
                if in_domain:
                    st.success(
                        f"✅ **In domain** — mean Tanimoto to 5 nearest training neighbours: "
                        f"{sim_pct:.1f}%.  Prediction is within the model's reliable range."
                    )
                else:
                    st.warning(
                        f"⚠️ **Out of domain** — similarity: {sim_pct:.1f}%.  "
                        "This compound is structurally distant from the training set. "
                        "Treat the prediction as a rough estimate only."
                    )
            else:
                st.info("AD check unavailable (training CSV not found).")

            st.markdown("---")
            st.subheader("Lipinski Rule of Five")
            ro5_df = pd.DataFrame([ro5])
            st.dataframe(ro5_df, hide_index=True, use_container_width=True)

            with st.expander("ℹ️ What is the Rule of Five?"):
                st.markdown(
                    "Lipinski's Rule of Five (Ro5) estimates **oral bioavailability**.  \n"
                    "A molecule is likely orally bioavailable if it satisfies ≤1 violation of:\n\n"
                    "- **MW ≤ 500 Da** — larger molecules are poorly absorbed\n"
                    "- **LogP ≤ 5** — too lipophilic → poor solubility\n"
                    "- **HBD ≤ 5** — hydrogen bond donors limit membrane crossing\n"
                    "- **HBA ≤ 10** — hydrogen bond acceptors limit membrane crossing\n\n"
                    "Note: some drugs (e.g. macrolides, natural products) intentionally break Ro5 "
                    "and use active transport mechanisms."
                )

    elif submit:
        st.warning("Please enter a SMILES string.")

    else:
        st.info("⬅️  Select an example drug or paste a SMILES string in the sidebar, then click **Predict**.")
        st.markdown("---")

        # Project overview on landing screen
        st.subheader("Project Overview")
        st.markdown(
            "This app demonstrates a full **Quantitative Structure-Activity Relationship (QSAR)** pipeline "
            "for predicting the bioactivity of small molecules against **EGFR** — a key target in oncology "
            "responsible for driving the growth of non-small-cell lung cancer (NSCLC), breast, pancreatic, "
            "and colorectal cancers.\n\n"
            "The pipeline covers data retrieval from **ChEMBL 33**, curation, molecular featurisation, "
            "classical ML modelling, uncertainty quantification, and virtual screening — "
            "mirroring workflows used in real drug discovery settings."
        )

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Training compounds", f"{ds.get('train_size', 8437):,}")
        c2.metric("Test compounds",     f"{ds.get('test_size', 2109):,}")
        c3.metric("Best R² (random)",   best.get("test_R²", 0.735))
        c4.metric("Best R² (scaffold)", p4.get("scaffold_split_LightGBM", {}).get("R²_test", 0.607))
        c5.metric("Best model",         best.get("name", "Stacking"))
        c6.metric("Active compounds",   f"{ds.get('active_pct', 72.4)}%")

        st.markdown("---")
        st.subheader("Key Findings")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "**What works:**\n"
                "- LightGBM + ECFP4 is the best single model (R²=0.715)\n"
                "- Stacking all four models reaches R²=0.735\n"
                "- 96.1% of test compounds fall within applicability domain\n"
                "- Conformal intervals achieve 90.9% empirical coverage\n"
                "- All 5 approved EGFR drugs scored as *Active* by virtual screening\n"
            )
        with col2:
            st.markdown(
                "**Limitations & honest caveats:**\n"
                "- Scaffold-split R²=0.607 — realistic estimate for novel scaffolds\n"
                "- Activity cliffs (4.4% of similar pairs) remain hard to predict\n"
                "- Model still rising on learning curve — more data would help\n"
                "- GCN and ChemBERTa underperform classical ML at this dataset size\n"
                "- Predictions are *in vitro* IC50, not in vivo efficacy\n"
            )


# ============================================================
# TAB 2 — Dataset & EDA
# ============================================================
with tab_eda:
    st.header("Dataset & Exploratory Data Analysis")
    st.markdown(
        "All bioactivity data was retrieved from **[ChEMBL 33](https://www.ebi.ac.uk/chembl/)** "
        "— the world's largest open-access database of bioactive molecules, maintained by the EMBL-EBI.  \n"
        "Target: **EGFR kinase** ([CHEMBL203](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL203/)), "
        "standard type **IC50** (half-maximal inhibitory concentration)."
    )

    # --- Metrics row ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw records fetched",    f"{ds.get('n_raw', 25758):,}")
    c2.metric("After curation",         f"{ds.get('n_compounds', 10546):,}")
    c3.metric("Removed by curation",    f"{ds.get('n_raw', 25758) - ds.get('n_compounds', 10546):,}")
    c4.metric("Retention rate",         f"{ds.get('n_compounds', 10546)/ds.get('n_raw', 25758)*100:.1f}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("pIC50 mean ± std",  f"{ds.get('pIC50_mean', 6.85):.2f} ± {ds.get('pIC50_std', 1.34):.2f}")
    c2.metric("Active (pIC50 ≥ 6)", f"{ds.get('active_pct', 72.4)}%")
    c3.metric("Ro5 compliant",      f"{ds.get('ro5_pass_pct', 72.8)}%")
    c4.metric("Train / Test split", f"{ds.get('train_size', 8437):,} / {ds.get('test_size', 2109):,}")

    st.markdown("---")

    # --- Curation pipeline ---
    st.subheader("Data Curation Pipeline")
    st.markdown(
        "Raw ChEMBL data contains noise: mixed units, inequality relations, duplicates, and invalid structures. "
        "The curation pipeline removes each source of noise systematically:"
    )
    curation_steps = pd.DataFrame([
        {"Step": "1. Unit filter",        "Condition": "standard_units == 'nM'",         "Rationale": "Ensures all IC50 values are on the same scale"},
        {"Step": "2. Relation filter",    "Condition": "standard_relation == '='",        "Rationale": "Removes '>' and '<' (censored values) — unusable for regression"},
        {"Step": "3. Remove nulls",       "Condition": "SMILES and IC50 not null/empty", "Rationale": "Drops compounds with missing structure or activity"},
        {"Step": "4. Numeric coercion",   "Condition": "IC50 parseable as float",        "Rationale": "Drops corrupted numeric fields"},
        {"Step": "5. Deduplication",      "Condition": "Median IC50 per ChEMBL ID",      "Rationale": "Multiple assays per compound → single robust estimate"},
        {"Step": "6. pIC50 range",        "Condition": "pIC50 ∈ [3, 12]",               "Rationale": "Removes extreme outliers (likely assay artefacts)"},
        {"Step": "7. SMILES validation",  "Condition": "RDKit.MolFromSmiles != None",    "Rationale": "Discards malformed SMILES that cannot be processed"},
    ])
    st.dataframe(curation_steps, hide_index=True, use_container_width=True)

    st.markdown("---")

    # --- EDA plots ---
    st.subheader("pIC50 Distribution & Activity Classes")
    st.markdown(
        "The pIC50 distribution is approximately **bell-shaped** centred around 6.8 — "
        "meaning the dataset is dominated by moderately-to-highly active compounds. "
        "This is expected: ChEMBL is enriched with active compounds that passed hit/lead filters. "
        "The **72.4% active rate** (pIC50 ≥ 6) reflects this bias toward potent compounds."
    )
    plot_img("eda_overview.png")

    st.subheader("Lipinski Drug-Likeness Descriptors by Activity Class")
    st.markdown(
        "Lipinski histograms separated by activity class reveal subtle but consistent trends:  \n"
        "- **MW:** Active compounds tend toward slightly higher MW (kinase inhibitors use "
        "bulkier groups to fill the ATP binding pocket)  \n"
        "- **LogP:** Active compounds show broader LogP distribution — lipophilicity drives "
        "membrane permeability but also toxicity  \n"
        "- **TPSA:** More active compounds can have higher TPSA due to multiple H-bond acceptors "
        "that interact with the kinase hinge region  \n"
        "The **72.8% Ro5 compliance rate** means ~27% of EGFR inhibitors are borderline "
        "drug-like — reflecting the trend toward larger, more complex kinase inhibitors."
    )
    plot_img("lipinski_descriptors.png")

    st.subheader("Morgan Fingerprint (ECFP4) Sparsity Analysis")
    st.markdown(
        "ECFP4 fingerprints with 2,048 bits are **extremely sparse**: only ~3% of bits are "
        "set on average per compound. This sparsity is actually advantageous for tree-based models "
        "(Random Forest, LightGBM, XGBoost) — they efficiently find the rare informative bits "
        "without being confused by uninformative zeros."
    )
    plot_img("fingerprint_analysis.png")

    # --- Data preview ---
    st.markdown("---")
    st.subheader("Dataset Preview")
    df_prev = load_dataset()
    if df_prev is not None:
        search = st.text_input("Filter by SMILES substring (optional)", "")
        df_show = df_prev if not search else df_prev[
            df_prev["canonical_smiles"].str.contains(search, case=False, na=False)
        ]
        cols_show = [c for c in ["molecule_chembl_id", "canonical_smiles", "IC50_nM",
                                  "pIC50", "activity_class"] if c in df_show.columns]
        st.dataframe(
            df_show[cols_show].head(50).reset_index(drop=True),
            use_container_width=True,
        )
        st.caption(f"Showing {min(50, len(df_show))} of {len(df_show):,} rows · "
                   f"Full dataset: {len(df_prev):,} compounds")


# ============================================================
# TAB 3 — Model Performance
# ============================================================
with tab_models:
    st.header("Model Training & Evaluation")
    st.markdown(
        "The pipeline benchmarks four classical ML algorithms, then extends to "
        "ensemble stacking, Bayesian hyperparameter optimisation, and deep learning baselines. "
        "All models use **ECFP4 Morgan fingerprints** (radius 2, 2,048 bits) as features, "
        "with an **80/20 stratified train-test split** and **5-fold cross-validation**."
    )

    # --- Phase 1: baseline table from JSON ---
    st.subheader("Phase 1 — Baseline Model Comparison")
    st.markdown(
        "**Interpreting regression metrics for QSAR:**\n"
        "- **R²** (coefficient of determination): 1.0 = perfect prediction. "
        "In QSAR, R² ≥ 0.6 is considered acceptable, ≥ 0.7 is good, ≥ 0.8 is excellent.\n"
        "- **RMSE** (root mean squared error) in pIC50 units — roughly the average prediction "
        "error on the logarithmic scale. RMSE = 0.7 means errors of ~5× in IC50.\n"
        "- **MAE** (mean absolute error) — similar to RMSE but less sensitive to outliers.\n"
        "- **CV R²** — cross-validation estimate (more reliable than single test-set R²)."
    )

    if p1:
        rows = []
        for name, m in p1.items():
            rows.append({
                "Model":    name,
                "R² test":  m.get("R²_test", "—"),
                "RMSE":     m.get("RMSE", "—"),
                "MAE":      m.get("MAE", "—"),
                "CV R²":    m.get("CV_R²", "—"),
            })
        df_p1 = pd.DataFrame(rows).sort_values("R² test", ascending=False)
        st.dataframe(df_p1, hide_index=True, use_container_width=True)

    plot_img("model_comparison.png", "Bar chart: R², RMSE, MAE across all four baseline models.")

    # --- Predicted vs actual ---
    st.subheader("Predicted vs Actual pIC50 — Best Model (LightGBM)")
    st.markdown(
        "Points on the diagonal = perfect predictions. Colour encodes residual magnitude: "
        "**green** = low error (< 0.5 pIC50), **red** = high error (> 1 pIC50).  \n\n"
        "The scatter follows the diagonal well across the full pIC50 range [4, 10], "
        "with slightly larger errors near the extremes — a known QSAR phenomenon "
        "caused by fewer training examples at the tails of the distribution."
    )
    plot_img("predicted_vs_actual.png")

    # --- Phase 2: advanced models ---
    st.subheader("Phase 2 — Advanced Models")

    r2_key = next((k for k in (p2.get("stacking_Ridge_meta") or {}) if "test" in k.lower()), "R²_test")

    adv_rows = []
    if p2.get("stacking_Ridge_meta"):
        m = p2["stacking_Ridge_meta"]
        adv_rows.append({"Model": "Stacking (RF+XGB+LGB+SVR → Ridge)",
                          "R² test": m.get(r2_key, m.get("R2_test", "—")),
                          "RMSE": m.get("RMSE", "—"), "MAE": m.get("MAE", "—"),
                          "Note": "Best overall"})
    if p3.get("optuna_lgbm_50trials_TPE"):
        m = p3["optuna_lgbm_50trials_TPE"]
        r2_o = next((v for k, v in m.items() if "test" in k.lower()), "—")
        adv_rows.append({"Model": f"LightGBM + Optuna ({m.get('n_trials','?')} trials, {m.get('sampler','TPE')})",
                          "R² test": r2_o,
                          "RMSE": m.get("RMSE", "—"), "MAE": m.get("MAE", "—"),
                          "Note": "Bayesian HPO"})
    if p3.get("chembert_zinc_base_v1"):
        m = p3["chembert_zinc_base_v1"]
        r2_c = next((v for k, v in m.items() if "test" in k.lower()), "—")
        adv_rows.append({"Model": "ChemBERTa (frozen embeddings + LightGBM)",
                          "R² test": r2_c, "RMSE": "—", "MAE": "—",
                          "Note": m.get("note", "")})
    if p3.get("gcn_3layer_pytorch_geometric"):
        m = p3["gcn_3layer_pytorch_geometric"]
        r2_g = next((v for k, v in m.items() if "test" in k.lower()), "—")
        adv_rows.append({"Model": "GCN 3-layer (PyTorch Geometric)",
                          "R² test": r2_g,
                          "RMSE": m.get("RMSE", "—"), "MAE": m.get("MAE", "—"),
                          "Note": m.get("architecture", "")})
    if adv_rows:
        st.dataframe(pd.DataFrame(adv_rows), hide_index=True, use_container_width=True)
        st.markdown(
            "> **Why does stacking win?** Ensembling combines the strengths of each model: "
            "Random Forest handles outliers well, LightGBM captures complex non-linear patterns, "
            "SVR provides smooth interpolation in dense regions, XGBoost adds regularisation. "
            "The Ridge meta-learner learns optimal weights without overfitting."
        )
        st.markdown(
            "> **Why do GCN and ChemBERTa underperform?** Deep learning methods need more data "
            "to beat well-tuned classical ML. At N ≈ 10k, the inductive bias of Morgan fingerprints "
            "is more useful than the expressiveness of graph networks or transformers. "
            "Fine-tuning ChemBERTa (rather than frozen embeddings) would likely close the gap."
        )

    # --- Fingerprint comparison ---
    fp_cmp = p2.get("fingerprint_comparison_CV_R2", {})
    if fp_cmp:
        st.subheader("Fingerprint Type Comparison (CV R²)")
        st.markdown(
            "Different molecular fingerprints encode different aspects of chemical structure. "
            "We compared three common types on the same LightGBM model:"
        )
        fp_rows = [{"Fingerprint": k, "CV R²": v,
                    "Description": {
                        "RDKit FP":       "Path-based, captures linear bond paths",
                        "ECFP4 (Morgan)": "Circular, captures local chemical environments (our primary choice)",
                        "MACCS Keys":     "166 predefined structural keys, interpretable but limited",
                    }.get(k, "")}
                   for k, v in fp_cmp.items()]
        st.dataframe(pd.DataFrame(fp_rows).sort_values("CV R²", ascending=False),
                     hide_index=True, use_container_width=True)

    # --- SHAP ---
    st.subheader("SHAP Feature Importance — Model Interpretability")
    st.markdown(
        "**SHAP** (SHapley Additive exPlanations) attributes each prediction to individual "
        "fingerprint bits, quantifying which structural substructures drive activity.  \n\n"
        "- **Left plot:** Top 20 bits ranked by mean |SHAP| — the most globally influential bits\n"
        "- **Right plot:** SHAP values for top 10 bits, split by bit presence (red) vs absence (green)\n\n"
        "In practice, high-SHAP bits correspond to substructures like the **quinazoline core**, "
        "**aniline substituents**, and **methoxy groups** — all well-known EGFR pharmacophore elements."
    )
    plot_img("shap_analysis.png")

    # --- Benchmark + Bootstrap ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Literature Benchmark")
        st.markdown(
            "Our pipeline is competitive with published EGFR QSAR studies "
            "despite using only freely available data and open-source tools.  \n"
            "The gold bar represents the stacking ensemble (R²=0.735)."
        )
        plot_img("benchmark_comparison.png")
    with col2:
        st.subheader("Bootstrap Confidence Intervals (1,000 resamples)")
        bci = p4.get("bootstrap_CI_1000_resamples_LightGBM", {})
        if bci:
            st.markdown(
                f"Single test-set R² = **{bci.get('R²_point', 0.715)}**  \n"
                f"95% CI: **[{bci.get('R²_95CI_lower', 0.691)}, {bci.get('R²_95CI_upper', 0.737)}]**  \n"
                f"RMSE 95% CI: [{bci.get('RMSE_95CI_lower', 0.676)} – {bci.get('RMSE_95CI_upper', 0.731)}]  \n\n"
                "The narrow CI confirms that the R²=0.715 estimate is stable and not a lucky "
                "train-test split artefact."
            )
        plot_img("bootstrap_ci.png")

    # --- Scaffold split ---
    st.subheader("Scaffold-Based Train/Test Split — Realistic Generalisation Estimate")
    sc = p4.get("scaffold_split_LightGBM", {})
    st.markdown(
        "A random 80/20 split can be *optimistic* — the test set may share scaffolds "
        "with the training set, inflating R². The **Murcko scaffold split** "
        "(standard in cheminformatics since MoleculeNet/Chemprop) ensures that all compounds "
        "sharing a scaffold core go to the same partition.  \n\n"
        f"| Split type | R² | RMSE | Comment |\n"
        f"|---|---|---|---|\n"
        f"| Random 80/20 | **{sc.get('R2_random_split', 0.715)}** | — | May overestimate real-world performance |\n"
        f"| Scaffold split | **{sc.get('R²_test', 0.607)}** | {sc.get('RMSE', 0.878)} | "
        f"Realistic estimate for novel scaffolds |\n"
        f"| Gap | **{sc.get('generalisation_gap', -0.108)}** | — | Expected in QSAR |\n"
    )
    plot_img("scaffold_split_comparison.png")


# ============================================================
# TAB 4 — Chemical Space
# ============================================================
with tab_chem:
    st.header("Chemical Space & Structural Analysis")

    # --- t-SNE ---
    st.subheader("t-SNE Chemical Space Map")
    st.markdown(
        "**t-SNE** (t-distributed Stochastic Neighbour Embedding) reduces the 2,048-dimensional "
        "fingerprint space to 2D while preserving local structure. We first apply **PCA → 50 dims** "
        "to denoise, then t-SNE on 1,500 randomly sampled compounds (full dataset takes ~5 min).  \n\n"
        "- **Left:** Continuous pIC50 colour scale — warm colours = more active compounds\n"
        "- **Right:** Discrete activity class — reveals whether active/inactive compounds form "
        "distinct clusters or are interleaved (the latter indicates activity cliffs)\n\n"
        "The map shows that EGFR inhibitors form **several distinct chemical clusters** "
        "corresponding to different kinase inhibitor scaffolds (quinazolines, pyrimidines, "
        "imidazoles, etc.) rather than a uniform cloud — confirming high scaffold diversity."
    )
    plot_img("chemical_space_tsne.png")

    # --- Scaffolds ---
    st.subheader("Murcko Scaffold Analysis")
    scf = p2.get("murcko_scaffold_analysis", {})
    if scf:
        c1, c2, c3 = st.columns(3)
        c1.metric("Unique scaffolds",       f"{scf.get('unique_scaffolds', 3857):,}")
        c2.metric("Scaffold diversity",     f"{scf.get('diversity_index_pct', 36.6)}%")
        c3.metric("Singleton scaffolds",    f"{scf.get('singleton_scaffold_pct', 60)}%")
    st.markdown(
        "A **Murcko scaffold** is the ring system + linkers of a molecule, stripped of "
        "side chains. High scaffold diversity (36.6% of compounds have a unique scaffold) "
        "means the dataset covers a broad chemical space — making the ML task harder but "
        "the trained model more transferable.  \n\n"
        "The **60% singleton rate** (scaffolds appearing only once) confirms that the dataset "
        "is not dominated by a small number of chemotypes — important for unbiased modelling."
    )
    plot_img("scaffold_analysis.png")

    # --- Virtual screening ---
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.subheader("Virtual Screening — Known EGFR Drugs")
        st.markdown(
            "Model validation: the pipeline should correctly rank 5 approved EGFR inhibitors "
            "as highly active. All were predicted as **Active** (pIC50 > 6):  \n\n"
            "| Drug | Generation | Mechanism | FDA approval |\n"
            "|------|-----------|-----------|-------------|\n"
            "| Erlotinib | 1st | Reversible, EGFR WT | 2004 |\n"
            "| Gefitinib | 1st | Reversible, EGFR WT | 2003 |\n"
            "| Afatinib | 2nd | Irreversible, pan-ErbB | 2013 |\n"
            "| Lapatinib | 2nd | Reversible, EGFR+ErbB2 | 2007 |\n"
            "| Osimertinib | 3rd | Irreversible, T790M mutant | 2017 |\n\n"
            "Osimertinib (*Tagrisso*, AstraZeneca, Cambridge UK) is the current SoC for "
            "EGFR-mutant NSCLC — its correct scoring validates the clinical relevance of this pipeline."
        )
        plot_img("virtual_screening.png")
    with col2:
        st.subheader("ErbB Family Selectivity")
        erbb = p3.get("erbb_family_compounds", {})
        if erbb:
            st.markdown("**ChEMBL 33 compound counts per target:**")
            erbb_df = pd.DataFrame([
                {"Target": "EGFR (CHEMBL203)",         "Compounds": erbb.get("EGFR_CHEMBL203", 10546),  "Data density": "Dense"},
                {"Target": "ErbB2 (CHEMBL1824)",       "Compounds": erbb.get("ErbB2_CHEMBL1824", 2494), "Data density": "Moderate"},
                {"Target": "ErbB4 (CHEMBL3009)",       "Compounds": erbb.get("ErbB4_CHEMBL3009", 225),  "Data density": "Sparse"},
                {"Target": "ErbB3 (CHEMBL2363049)",    "Compounds": erbb.get("ErbB3_CHEMBL2363049", 83),"Data density": "Very sparse"},
            ])
            st.dataframe(erbb_df, hide_index=True, use_container_width=True)
        st.markdown(
            "The selectivity scatter plot reveals that many EGFR inhibitors also hit ErbB2 "
            "(HER2) — explaining why drugs like lapatinib and afatinib are dual EGFR/ErbB2 inhibitors. "
            "True EGFR-selective compounds cluster above the diagonal."
        )
        plot_img("erbb_selectivity.png")


# ============================================================
# TAB 5 — Advanced Analysis
# ============================================================
with tab_adv:
    st.header("Advanced Analysis")
    st.markdown(
        "This section covers four rigorous analyses that go beyond standard QSAR: "
        "applicability domain, calibrated uncertainty, activity cliff characterisation, "
        "3D molecular shape, and data efficiency."
    )

    # --- AD ---
    st.subheader("Applicability Domain (k-NN Tanimoto)")
    ad = p2.get("applicability_domain_kNN_k5", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("In-domain compounds",     f"{ad.get('in_domain_pct', 96.1)}%")
    col2.metric("R² (in-domain only)",     ad.get("R2_in_domain", 0.714))
    col3.metric("AD threshold (percentile)", f"{ad.get('threshold_percentile', 95)}th")
    st.markdown(
        "The **applicability domain** identifies compounds the model is likely to predict reliably. "
        "We define it via mean Tanimoto distance to the **k=5 nearest training neighbours**: "
        "compounds closer than the 95th percentile of within-training distances are *in domain*.  \n\n"
        "**96.1% in-domain rate** on the test set means the model sees almost no truly novel "
        "chemical matter in this held-out set — expected when using random splits. "
        "The scaffold split (Tab 3) gives a more honest picture of generalisation."
    )
    plot_img("applicability_domain.png")

    # --- Conformal prediction ---
    st.subheader("Conformal Prediction — Calibrated Uncertainty Intervals")
    cp = p3.get("conformal_prediction_split", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Target coverage",     f"{cp.get('target_coverage', 0.90)*100:.0f}%")
    c2.metric("Empirical coverage",  f"{cp.get('empirical_coverage', 0.909)*100:.1f}%")
    c3.metric("Interval half-width", f"±{cp.get('interval_width_pIC50', 1.251)} pIC50")
    st.markdown(
        "**Split-conformal prediction** provides *distribution-free* coverage guarantees: "
        "if you request 90% intervals, the method *provably* covers ≥90% of future test points "
        "regardless of the model or data distribution.  \n\n"
        "Our empirical coverage is **90.9%** (target: 90%) — confirming the calibration is accurate. "
        "The interval width of **±1.251 pIC50** corresponds to roughly ±17× uncertainty in IC50 — "
        "wide, but honest. Classical QSAR models rarely report calibrated uncertainty at all."
    )
    plot_img("conformal_prediction.png")

    # --- Activity cliffs ---
    st.subheader("Activity Cliff Analysis")
    ac = p4.get("activity_cliff_analysis", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Similar pairs (Tan ≥ 0.6)",  f"{ac.get('similar_pairs', 5422):,}")
    c2.metric("Cliff pairs (ΔpIC50 ≥ 2)",   f"{ac.get('cliff_pairs', 241):,}")
    c3.metric("Cliff density",              f"{ac.get('cliff_density_pct', 4.4)}%")
    c4.metric("Structural similarity cut",   ac.get("tanimoto_threshold", 0.6))
    st.markdown(
        "**Activity cliffs** are pairs of structurally similar compounds with large differences in "
        "potency — the hardest-to-predict cases in QSAR.  \n\n"
        "With 4.4% cliff density (241 cliff pairs among 5,422 similar pairs), EGFR inhibitors "
        "show a *moderate* cliff landscape. This explains the residual scatter in the predicted "
        "vs actual plot: cliff compounds systematically challenge the model because a single "
        "substituent change can shift IC50 by 100× or more.  \n\n"
        "Activity cliffs are caused by subtle electronic effects, binding-pose switches, "
        "or induced-fit changes in the EGFR binding pocket that fingerprint-based models cannot capture."
    )
    plot_img("activity_cliffs.png")

    # --- 3D shape ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("3D Conformer Shape Descriptors")
        st.markdown(
            "Beyond 2D fingerprints, **3D shape descriptors** capture the spatial geometry "
            "of molecules. We compute **Principal Moments of Inertia (PMI)** for 200 sampled "
            "compounds using RDKit's ETKDG conformer generator.  \n\n"
            "The **PMI triangle** has three corners:  \n"
            "- **Rod-like** (linear, e.g. alkynes)\n"
            "- **Disc-like** (flat, e.g. aromatic rings)\n"
            "- **Sphere-like** (globular, e.g. adamantane)\n\n"
            "Active EGFR inhibitors cluster toward the **disc-rod** region — consistent with "
            "the flat quinazoline/pyrimidine scaffolds that bind in the planar ATP pocket."
        )
        plot_img("shape_descriptors_3d.png")
    with col2:
        st.subheader("Learning Curves — Data Efficiency")
        lc = p4.get("learning_curves_LightGBM", {})
        c1, c2 = st.columns(2)
        c1.metric("R² at 5% data",   lc.get("R2_at_5pct", 0.365))
        c2.metric("R² at 100% data", lc.get("R2_at_100pct", 0.705))
        st.markdown(
            "Learning curves show how model performance scales with training set size.  \n\n"
            f"- At **5% data** (≈420 compounds): R² = {lc.get('R2_at_5pct', 0.365)} — already predictive\n"
            f"- At **100% data** (≈8,400 compounds): R² = {lc.get('R2_at_100pct', 0.705)}\n"
            f"- The curve is **still rising** at 100% — more ChEMBL data or external datasets "
            "(e.g. BindingDB, patent databases) would improve the model further\n\n"
            "This contrasts with deep learning, which typically needs 100k+ compounds to plateau."
        )
        plot_img("learning_curves.png")


# ============================================================
# TAB 6 — Methodology
# ============================================================
with tab_method:
    st.header("Methodology & Pipeline")
    st.markdown(
        "This section explains the full scientific pipeline — from raw database records "
        "to calibrated predictions — as it would appear in a Methods section of a paper."
    )

    # --- Pipeline overview ---
    st.subheader("Pipeline Overview")
    st.markdown(
        """
```
ChEMBL 33 API
    ↓  IC50 records for EGFR (CHEMBL203)
Data Curation
    ↓  nM filter → exact relation → deduplication → SMILES validation → pIC50 [3,12]
Feature Engineering
    ↓  ECFP4 Morgan Fingerprints (radius 2, 2048 bits) via RDKit
    ↓  [Optional: Lipinski descriptors, RDKit 2D descriptors, 3D PMI descriptors]
Model Training
    ↓  Random Forest · XGBoost · LightGBM · SVR (RBF) — 5-fold CV + 80/20 holdout
    ↓  Stacking ensemble (Ridge meta-learner)
    ↓  Hyperparameter optimisation (Optuna TPE, 50 trials)
Validation
    ↓  Random split R² / Scaffold split R² / Bootstrap CI / Conformal intervals
Interpretability
    ↓  SHAP feature importance · t-SNE chemical space · Activity cliff analysis
Deployment
    ↓  Streamlit web app + Hugging Face Spaces
```
        """
    )

    # --- Target biology ---
    st.subheader("Target: EGFR Kinase")
    st.markdown(
        "**Epidermal Growth Factor Receptor (EGFR / ErbB1 / HER1)** is a receptor tyrosine kinase "
        "encoded by the *EGFR* gene on chromosome 7p12.  \n\n"
        "**Oncological relevance:**\n"
        "- Overexpressed or mutated in 25–30% of non-small-cell lung cancers (NSCLC)\n"
        "- Activating mutations (exon 19 deletions, L858R) drive tumour growth\n"
        "- Resistance mutations (T790M, C797S) emerge under treatment pressure\n\n"
        "**Drug target rationale:** EGFR inhibitors competitively block the ATP binding site "
        "in the kinase domain, preventing autophosphorylation and downstream MAPK/PI3K signalling. "
        "Three generations of inhibitors have been developed to overcome successive resistance mutations.\n\n"
        "**ChEMBL target ID:** CHEMBL203 · UniProt: P00533 · PDB: 1IVO (erlotinib co-crystal)"
    )

    # --- Features ---
    st.subheader("Molecular Featurisation")
    feat_data = pd.DataFrame([
        {"Feature type": "ECFP4 Morgan Fingerprints", "Dimension": "2,048 bits",
         "Pros": "Encodes local chemical environments; sparse; fast; interpretable via SHAP",
         "Cons": "No 3D info; no stereochemistry by default"},
        {"Feature type": "Lipinski Descriptors", "Dimension": "5 scalars",
         "Pros": "Drug-likeness; human-interpretable; fast",
         "Cons": "Very low dimensionality; limited predictive power alone"},
        {"Feature type": "RDKit 2D Descriptors", "Dimension": "207 scalars",
         "Pros": "Captures diverse physicochemical properties",
         "Cons": "Many correlated; needs feature selection"},
        {"Feature type": "3D PMI/NPR Descriptors", "Dimension": "4 scalars",
         "Pros": "Captures 3D molecular shape",
         "Cons": "Requires conformer generation (slow); single conformer ≠ bioactive conformer"},
        {"Feature type": "ChemBERTa Embeddings", "Dimension": "384",
         "Pros": "Pretrained on 77M SMILES; captures global chemical context",
         "Cons": "Frozen embeddings underperform at N=10k; fine-tuning needed"},
    ])
    st.dataframe(feat_data, hide_index=True, use_container_width=True)

    # --- Validation strategy ---
    st.subheader("Validation Strategy")
    st.markdown(
        "Robust QSAR validation requires multiple complementary approaches:"
    )
    val_data = pd.DataFrame([
        {"Method": "5-fold cross-validation",       "R²": "0.694 ± 0.018 (LightGBM)",
         "Purpose": "Unbiased performance estimate; model selection"},
        {"Method": "Random 80/20 holdout",           "R²": "0.715 (LightGBM)",
         "Purpose": "Comparable to published benchmarks"},
        {"Method": "Scaffold-based split",            "R²": "0.607 (LightGBM)",
         "Purpose": "Realistic estimate for novel scaffolds"},
        {"Method": "Bootstrap CI (1,000 resamples)", "R²": "0.715 [0.691 – 0.737]",
         "Purpose": "Quantifies uncertainty in the R² estimate itself"},
        {"Method": "Conformal prediction",           "R²": "Coverage 90.9% @ 90% target",
         "Purpose": "Calibrated per-compound prediction intervals"},
        {"Method": "Virtual screening (known drugs)", "R²": "5/5 Active (qualitative)",
         "Purpose": "Clinical relevance check"},
    ])
    st.dataframe(val_data, hide_index=True, use_container_width=True)

    # --- Software ---
    st.subheader("Software & Reproducibility")
    st.markdown(
        "All code is open-source. The full pipeline can be reproduced from scratch:\n\n"
        "| Package | Version | Role |\n"
        "|---------|---------|------|\n"
        "| RDKit | ≥2022.9 | SMILES parsing, fingerprints, Lipinski, conformers |\n"
        "| chembl-webresource-client | ≥0.10 | ChEMBL API data retrieval |\n"
        "| scikit-learn | ≥1.3 | Random Forest, SVR, pipelines, metrics |\n"
        "| LightGBM | ≥4.0 | Best single model |\n"
        "| XGBoost | ≥2.0 | Baseline gradient boosting |\n"
        "| SHAP | ≥0.44 | Model interpretability |\n"
        "| Optuna | ≥3.5 | Bayesian hyperparameter optimisation |\n"
        "| PyTorch Geometric | ≥2.4 | Graph Neural Network baseline |\n"
        "| HuggingFace Transformers | ≥4.40 | ChemBERTa embeddings |\n"
        "| Streamlit | ≥1.30 | This web application |\n\n"
        "**Data:** ChEMBL 33 (EMBL-EBI, 2023). CC BY-SA 3.0 licence.  \n"
        "**Model:** MIT licence. Predictions are for research purposes only."
    )

    # --- References ---
    st.subheader("Key References")
    st.markdown(
        "1. Gaulton et al. (2017). *ChEMBL: a large-scale bioactivity database for drug discovery.* "
        "Nucleic Acids Research.\n"
        "2. Rogers & Hahn (2010). *Extended-Connectivity Fingerprints.* J. Chem. Inf. Model.\n"
        "3. Lundberg & Lee (2017). *A unified approach to interpreting model predictions (SHAP).* NeurIPS.\n"
        "4. Vovk et al. (2005). *Algorithmic Learning in a Random World.* Springer. (Conformal prediction)\n"
        "5. Hu et al. (2019). *Strategies for Pre-training Graph Neural Networks.* ICLR.\n"
        "6. Lipinski et al. (2001). *Experimental and computational approaches to estimate "
        "solubility and permeability.* Adv. Drug Deliv. Rev.\n"
        "7. Bemis & Murcko (1996). *The properties of known drugs: Molecular frameworks.* "
        "J. Med. Chem. (Murcko scaffolds)\n"
    )
