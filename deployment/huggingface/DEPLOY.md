# Deploy to Hugging Face Spaces

This folder contains the metadata needed to publish [`app.py`](../../app.py)
on [Hugging Face Spaces](https://huggingface.co/spaces) as a **Streamlit**
Space.

## What gets deployed

| File / folder                          | Purpose                                       |
|----------------------------------------|-----------------------------------------------|
| `app.py`                               | Streamlit UI (root of repo)                   |
| `requirements.txt`                     | Python deps (root of repo)                    |
| `data/egfr_bioactivity_curated.csv`    | Training data — model trains on first launch |
| `outputs/best_model.joblib` (optional) | Pre-trained model (skips first-launch retrain)|
| `deployment/huggingface/README.md`     | Space card / YAML frontmatter (copied to root)|

## One-time setup

1. Create a new Space at <https://huggingface.co/new-space>:
   - **Space SDK:** *Streamlit*
   - **Hardware:** *CPU basic* is sufficient (model is small, < 200 MB RAM)
   - **License:** MIT
2. Install the Hugging Face CLI and log in:
   ```bash
   pip install -U huggingface_hub
   huggingface-cli login   # paste a write-scoped token from hf.co/settings/tokens
   ```

## Pushing the code

The Space is a regular git repository — push from the project root.

```bash
# Replace USER and SPACE_NAME with your handle / chosen name.
git remote add space https://huggingface.co/spaces/USER/SPACE_NAME

# Use this folder's README.md as the Space landing page (it has the YAML
# frontmatter Spaces requires). The copy below is overwritten on every
# deploy push; keep the canonical project README in the repo root.
cp deployment/huggingface/README.md README.hf.md

# Push only what the Space needs (everything else is fine to include too,
# but this keeps the Space lean and the cold-start fast):
git push space HEAD:main
```

If you would rather keep the repo's normal `README.md` untouched, create a
**dedicated branch** for the Space:

```bash
git checkout -b hf-space
cp deployment/huggingface/README.md README.md   # overwrites for this branch only
git add README.md
git commit -m "Hugging Face Space card"
git push space hf-space:main
git checkout main
```

## Optional: ship a pre-trained model

By default, `app.py` retrains a small LightGBM model on first launch (≈30 s on
free Spaces hardware) and caches it to `outputs/best_model.joblib`. If you
prefer instant cold-start, train it locally and commit the joblib:

```bash
python -c "import app; app.get_model_and_data()"     # warms the cache
git add outputs/best_model.joblib
git commit -m "Pre-trained model for HF Space"
```

> ⚠️ The joblib will be ~10–50 MB depending on `n_estimators`. Hugging Face
> Spaces accepts files up to 5 GB via Git LFS, so this is safe.

## Verifying locally

The Streamlit Space environment can be approximated with:

```bash
python -m venv .venv-hf && source .venv-hf/bin/activate   # PowerShell: .\.venv-hf\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501> — the interface is identical to what users see
on the Space.

## Updating the Space

Subsequent deploys are just `git push space HEAD:main`. The Space rebuilds
automatically on every push.
