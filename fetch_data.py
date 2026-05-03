"""Fetch IC50 bioactivity data from ChEMBL (and optionally PubChem / BindingDB
/ ExCAPE-DB) for one or more targets.

Examples:
    # Default: EGFR only from ChEMBL (backwards-compatible)
    python fetch_data.py

    # ErbB family for selectivity / multi-task work (Phase 3)
    python fetch_data.py --targets EGFR,ERBB2,ERBB3,ERBB4

    # Custom CHEMBL IDs
    python fetch_data.py --targets CHEMBL203,CHEMBL1824

    # Phase 3 dataset expansion — pull EGFR data from additional sources and
    # build a unified CSV (deduplicated on canonical SMILES, ChEMBL preferred):
    python fetch_data.py --extra-sources pubchem,bindingdb,excape

    # Use a locally-downloaded ExCAPE / BindingDB dump instead of fetching:
    python fetch_data.py --extra-sources excape \
        --excape-tsv path/to/pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv.xz
    python fetch_data.py --extra-sources bindingdb --bindingdb-tsv path/to/BindingDB_All.tsv

Notes on extra sources:
  * PubChem  — BioAssay AID 1851 (NCATS EGFR kinase panel) via the public REST
               API. Returns IC50 (uM) for the EGFR sub-assay.
  * BindingDB — large pre-downloaded TSV is required (BindingDB_All.tsv from
                bindingdb.org/rwd/bind/downloads/BindingDB_All_*.tsv.zip). The
                dump is filtered for EGFR target name + IC50 (nM).
  * ExCAPE-DB — large TSV/TSV.XZ dump required (Sun et al. 2017, J Cheminform
                9:17, downloadable from Zenodo). Filtered to gene symbol EGFR.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Known kinase targets relevant to this project
# ----------------------------------------------------------------------------
TARGETS = {
    'EGFR':  'CHEMBL203',      # primary target
    'ERBB2': 'CHEMBL1824',     # HER2
    'ERBB3': 'CHEMBL2363049',  # HER3
    'ERBB4': 'CHEMBL3009',     # HER4
}


def resolve_target(token: str) -> tuple[str, str]:
    """Return (alias, chembl_id) for a user token (alias or raw CHEMBL id)."""
    token = token.strip().upper()
    if token in TARGETS:
        return token, TARGETS[token]
    if token.startswith('CHEMBL'):
        alias = next((a for a, cid in TARGETS.items() if cid == token), token)
        return alias, token
    raise ValueError(f'Unknown target: {token!r}. Known aliases: {list(TARGETS)}')


def fetch_target(alias: str, chembl_id: str, out_dir: str = 'data') -> pd.DataFrame:
    """Fetch + curate IC50 data for one ChEMBL target. Returns curated DataFrame."""
    from chembl_webresource_client.new_client import new_client

    out_csv = os.path.join(out_dir, f'{alias.lower()}_bioactivity_curated.csv')
    print(f'\n=== {alias} ({chembl_id}) ===')
    print('Connecting to ChEMBL...')

    res = new_client.activity.filter(
        target_chembl_id=chembl_id,
        standard_type='IC50',
    ).only([
        'molecule_chembl_id', 'canonical_smiles',
        'standard_value', 'standard_units', 'standard_relation',
    ])

    rows = []
    for i, r in enumerate(res):
        rows.append(r)
        if (i + 1) % 1000 == 0:
            print(f'  ...{i+1:,} records', flush=True)
    print(f'Raw records: {len(rows):,}')

    if not rows:
        print(f'No records found for {alias}; skipping.')
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Curation: same filters as the original single-target script
    df = df[df['standard_units'] == 'nM'].copy()
    df = df[df['standard_relation'] == '='].copy()
    df.dropna(subset=['canonical_smiles', 'standard_value'], inplace=True)
    df = df[df['canonical_smiles'].str.strip() != ''].copy()
    df['IC50_nM'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df.dropna(subset=['IC50_nM'], inplace=True)

    df = (df.groupby('molecule_chembl_id')
            .agg({'canonical_smiles': 'first', 'IC50_nM': 'median'})
            .reset_index())

    df['pIC50'] = -np.log10(df['IC50_nM'] * 1e-9)
    df = df[(df['pIC50'] >= 3) & (df['pIC50'] <= 12)].reset_index(drop=True)
    df['target'] = alias

    df.to_csv(out_csv, index=False)
    print(f'Saved: {out_csv}')
    print(f'  Compounds : {len(df):,}')
    print(f'  pIC50     : {df["pIC50"].min():.2f} - {df["pIC50"].max():.2f}')
    print(f'  Active >6 : {(df["pIC50"] > 6).sum():,} '
          f'({(df["pIC50"] > 6).mean() * 100:.1f}%)')
    return df


def build_multi_target_table(per_target: dict[str, pd.DataFrame],
                             out_dir: str = 'data') -> pd.DataFrame:
    """Build a wide table: one row per molecule, one pIC50 column per target.

    Useful for selectivity analysis and multi-task model training (Phase 3).
    """
    if len(per_target) < 2:
        return pd.DataFrame()

    parts = []
    smiles_lookup: dict[str, str] = {}
    for alias, df in per_target.items():
        if df.empty:
            continue
        sub = df[['molecule_chembl_id', 'canonical_smiles', 'pIC50']].copy()
        for cid, smi in zip(sub['molecule_chembl_id'], sub['canonical_smiles']):
            smiles_lookup.setdefault(cid, smi)
        parts.append(
            sub[['molecule_chembl_id', 'pIC50']]
            .rename(columns={'pIC50': f'pIC50_{alias}'})
        )

    if not parts:
        return pd.DataFrame()

    wide = parts[0]
    for p in parts[1:]:
        wide = wide.merge(p, on='molecule_chembl_id', how='outer')
    wide.insert(1, 'canonical_smiles',
                wide['molecule_chembl_id'].map(smiles_lookup))

    out_csv = os.path.join(out_dir, 'multitarget_bioactivity.csv')
    wide.to_csv(out_csv, index=False)

    pic_cols = [c for c in wide.columns if c.startswith('pIC50_')]
    n_full = wide[pic_cols].notna().all(axis=1).sum()
    print('\n=== Multi-target wide table ===')
    print(f'Saved: {out_csv}')
    print(f'  Unique molecules     : {len(wide):,}')
    print(f'  Compounds with all   : {n_full:,} '
          f'({n_full / max(len(wide), 1) * 100:.1f}%) — usable for joint training')
    print('  Per-target coverage  :')
    for c in pic_cols:
        print(f'    {c:<15} {wide[c].notna().sum():>6,} compounds')
    return wide


# ---------------------------------------------------------------------------
# Phase 3 — Dataset expansion: PubChem, BindingDB, ExCAPE-DB
# ---------------------------------------------------------------------------
def _canonicalise_smiles(smi: str) -> str | None:
    """RDKit canonical SMILES; returns None on parse failure."""
    try:
        from rdkit import Chem  # type: ignore
    except ImportError:
        return smi  # fall back to as-is if RDKit unavailable
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _finalise_pic50(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Common post-processing: canonicalise SMILES, compute pIC50, dedup, filter."""
    df = df.dropna(subset=['canonical_smiles', 'IC50_nM']).copy()
    df['IC50_nM'] = pd.to_numeric(df['IC50_nM'], errors='coerce')
    df = df.dropna(subset=['IC50_nM'])
    df = df[df['IC50_nM'] > 0].copy()
    df['canonical_smiles'] = df['canonical_smiles'].map(_canonicalise_smiles)
    df = df.dropna(subset=['canonical_smiles'])
    df = (df.groupby('canonical_smiles', as_index=False)
            .agg({'IC50_nM': 'median'}))
    df['pIC50'] = -np.log10(df['IC50_nM'] * 1e-9)
    df = df[(df['pIC50'] >= 3) & (df['pIC50'] <= 12)].reset_index(drop=True)
    df['source'] = source
    return df


def fetch_pubchem_aid(aid: int = 1851, out_dir: str = 'data') -> pd.DataFrame:
    """Fetch IC50 data for a PubChem BioAssay AID (default 1851 = NCATS EGFR).

    Uses the public PubChem PUG REST API. EGFR sub-assay activity column is
    autodetected. Returns a DataFrame with canonical_smiles + IC50_nM + pIC50.
    """
    import io

    try:
        import requests  # type: ignore
    except ImportError:
        print('warning: `requests` not installed; cannot query PubChem.',
              file=sys.stderr)
        return pd.DataFrame()

    print(f'\n=== PubChem AID {aid} ===')
    base = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
    try:
        # 1. Concise CSV of (CID, Activity_*, Phenotype) for the assay
        r = requests.get(f'{base}/assay/aid/{aid}/CSV', timeout=120)
        r.raise_for_status()
        assay = pd.read_csv(io.StringIO(r.text), low_memory=False)
    except Exception as e:
        print(f'warning: PubChem assay download failed: {e}', file=sys.stderr)
        return pd.DataFrame()

    # Locate IC50 column (first column whose name contains "IC50")
    ic50_cols = [c for c in assay.columns if 'IC50' in c.upper()]
    cid_col = next((c for c in assay.columns if c.upper() == 'PUBCHEM_CID'), None)
    if cid_col is None or not ic50_cols:
        print('warning: could not find CID/IC50 columns in PubChem CSV.',
              file=sys.stderr)
        return pd.DataFrame()
    ic50_col = ic50_cols[0]
    egfr = assay[[cid_col, ic50_col]].rename(
        columns={cid_col: 'cid', ic50_col: 'IC50_uM'})
    egfr = egfr.dropna(subset=['cid', 'IC50_uM'])
    egfr['cid'] = egfr['cid'].astype('Int64')
    print(f'  Records with IC50: {len(egfr):,}')

    if egfr.empty:
        return pd.DataFrame()

    # 2. Fetch SMILES for these CIDs in batches of 200
    smiles_map: dict[int, str] = {}
    cids = egfr['cid'].dropna().unique().tolist()
    for i in range(0, len(cids), 200):
        chunk = cids[i:i + 200]
        cid_str = ','.join(str(int(c)) for c in chunk)
        try:
            rr = requests.get(
                f'{base}/compound/cid/{cid_str}/property/CanonicalSMILES/CSV',
                timeout=60,
            )
            rr.raise_for_status()
            sub = pd.read_csv(io.StringIO(rr.text))
            for cid, smi in zip(sub['CID'], sub['CanonicalSMILES']):
                smiles_map[int(cid)] = smi
        except Exception as e:
            print(f'  ...batch {i}-{i+200} failed: {e}', file=sys.stderr)
        if (i + 200) % 2000 == 0:
            print(f'  ...resolved {len(smiles_map):,}/{len(cids):,} SMILES',
                  flush=True)

    egfr['canonical_smiles'] = egfr['cid'].map(smiles_map)
    egfr['IC50_nM'] = pd.to_numeric(egfr['IC50_uM'], errors='coerce') * 1000.0

    out = _finalise_pic50(
        egfr[['canonical_smiles', 'IC50_nM']].copy(), source=f'PubChem-AID{aid}')
    out_csv = os.path.join(out_dir, f'egfr_pubchem_aid{aid}.csv')
    out.to_csv(out_csv, index=False)
    print(f'Saved: {out_csv}  ({len(out):,} compounds)')
    return out


def load_bindingdb(tsv_path: str | None, out_dir: str = 'data',
                   target_keyword: str = 'EGFR') -> pd.DataFrame:
    """Filter a local BindingDB_All.tsv dump for EGFR IC50 entries.

    Download once from https://www.bindingdb.org/rwd/bind/downloads/ and pass
    the path via --bindingdb-tsv. Returns canonical_smiles + IC50_nM + pIC50.
    """
    print(f'\n=== BindingDB ({target_keyword}) ===')
    if not tsv_path or not os.path.exists(tsv_path):
        print(f'warning: BindingDB TSV not found at {tsv_path!r}; skipping. '
              'Download from https://www.bindingdb.org/rwd/bind/downloads/',
              file=sys.stderr)
        return pd.DataFrame()

    # Stream-read large TSV; only keep the columns we need
    cols_needed = ['Ligand SMILES', 'Target Name Assigned by Curator or DataSource',
                   'IC50 (nM)']
    try:
        chunks = pd.read_csv(tsv_path, sep='\t', usecols=cols_needed,
                             chunksize=200_000, low_memory=False,
                             on_bad_lines='skip')
    except Exception as e:
        print(f'warning: failed to read BindingDB TSV: {e}', file=sys.stderr)
        return pd.DataFrame()

    keep = []
    total = 0
    for chunk in chunks:
        total += len(chunk)
        m = chunk['Target Name Assigned by Curator or DataSource'] \
            .astype(str).str.contains(target_keyword, case=False, na=False)
        keep.append(chunk[m])
    if not keep:
        return pd.DataFrame()
    df = pd.concat(keep, ignore_index=True)
    print(f'  Scanned {total:,} rows; matched {target_keyword}: {len(df):,}')

    # Some BindingDB IC50 cells contain ">", "<" etc. — strip and coerce
    ic = df['IC50 (nM)'].astype(str).str.replace(r'[<>~=]', '', regex=True)
    df = df.assign(
        canonical_smiles=df['Ligand SMILES'],
        IC50_nM=pd.to_numeric(ic, errors='coerce'),
    )[['canonical_smiles', 'IC50_nM']]

    out = _finalise_pic50(df, source='BindingDB')
    out_csv = os.path.join(out_dir, 'egfr_bindingdb.csv')
    out.to_csv(out_csv, index=False)
    print(f'Saved: {out_csv}  ({len(out):,} compounds)')
    return out


def load_excape(tsv_path: str | None, out_dir: str = 'data',
                gene_symbol: str = 'EGFR') -> pd.DataFrame:
    """Filter a local ExCAPE-DB dump for EGFR pXC50 records.

    The dump (Sun et al. 2017) is distributed as a large TSV(.xz). Pass the
    path via --excape-tsv. Columns of interest: Gene_Symbol, SMILES, pXC50,
    Activity_Flag.
    """
    print(f'\n=== ExCAPE-DB ({gene_symbol}) ===')
    if not tsv_path or not os.path.exists(tsv_path):
        print(f'warning: ExCAPE TSV not found at {tsv_path!r}; skipping. '
              'Download from https://zenodo.org/record/173258', file=sys.stderr)
        return pd.DataFrame()

    cols_needed = ['Gene_Symbol', 'SMILES', 'pXC50', 'Activity_Flag']
    try:
        chunks = pd.read_csv(tsv_path, sep='\t', usecols=cols_needed,
                             chunksize=500_000, low_memory=False,
                             on_bad_lines='skip')
    except Exception as e:
        print(f'warning: failed to read ExCAPE TSV: {e}', file=sys.stderr)
        return pd.DataFrame()

    keep = []
    total = 0
    for chunk in chunks:
        total += len(chunk)
        m = chunk['Gene_Symbol'].astype(str).str.upper() == gene_symbol.upper()
        keep.append(chunk[m])
    df = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
    print(f'  Scanned {total:,} rows; matched {gene_symbol}: {len(df):,}')
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=['SMILES', 'pXC50']).copy()
    df['pXC50'] = pd.to_numeric(df['pXC50'], errors='coerce')
    df = df.dropna(subset=['pXC50'])
    df['IC50_nM'] = (10 ** -df['pXC50']) * 1e9

    out = _finalise_pic50(
        df[['SMILES', 'IC50_nM']].rename(columns={'SMILES': 'canonical_smiles'}),
        source='ExCAPE-DB')
    out_csv = os.path.join(out_dir, 'egfr_excape.csv')
    out.to_csv(out_csv, index=False)
    print(f'Saved: {out_csv}  ({len(out):,} compounds)')
    return out


def merge_egfr_sources(chembl: pd.DataFrame,
                       extras: list[pd.DataFrame],
                       out_dir: str = 'data',
                       out_name: str = 'egfr_unified.csv') -> pd.DataFrame:
    """Merge ChEMBL EGFR with extra-source DataFrames on canonical SMILES.

    ChEMBL is preferred when a SMILES appears in multiple sources (per-source
    pIC50 columns are kept for provenance audits). Returns the unified table.
    """
    print('\n=== Unified EGFR table ===')
    chembl_sub = chembl[['canonical_smiles', 'IC50_nM', 'pIC50']].copy()
    chembl_sub['canonical_smiles'] = chembl_sub['canonical_smiles'].map(
        _canonicalise_smiles)
    chembl_sub = chembl_sub.dropna(subset=['canonical_smiles'])
    chembl_sub['source'] = 'ChEMBL'

    all_parts = [chembl_sub] + [e for e in extras if e is not None and not e.empty]
    if len(all_parts) == 1:
        print('  No extra sources merged.')
        return chembl_sub

    long = pd.concat(all_parts, ignore_index=True)
    pivot = (long.pivot_table(index='canonical_smiles',
                              columns='source', values='pIC50',
                              aggfunc='median'))
    pivot.columns = [f'pIC50_{c}' for c in pivot.columns]

    # Consensus pIC50: prefer ChEMBL, else mean across remaining sources
    if 'pIC50_ChEMBL' in pivot.columns:
        consensus = pivot['pIC50_ChEMBL'].copy()
        other_cols = [c for c in pivot.columns if c != 'pIC50_ChEMBL']
        consensus = consensus.fillna(pivot[other_cols].mean(axis=1))
    else:
        consensus = pivot.mean(axis=1)
    pivot.insert(0, 'pIC50', consensus)
    pivot.insert(1, 'n_sources',
                 pivot[[c for c in pivot.columns if c.startswith('pIC50_') and
                        c != 'pIC50']].notna().sum(axis=1))

    pivot = pivot.reset_index()
    out_csv = os.path.join(out_dir, out_name)
    pivot.to_csv(out_csv, index=False)
    print(f'Saved: {out_csv}')
    print(f'  Compounds (union)   : {len(pivot):,}')
    for c in pivot.columns:
        if c.startswith('pIC50_'):
            print(f'    {c:<22} {pivot[c].notna().sum():>6,}')
    print(f'  In >=2 sources      : {(pivot["n_sources"] >= 2).sum():,}')
    return pivot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--targets', default='EGFR',
                        help='Comma-separated target aliases or CHEMBL ids '
                             f'(default: EGFR). Known aliases: {list(TARGETS)}')
    parser.add_argument('--out-dir', default='data',
                        help='Directory to write CSV files into (default: data)')
    parser.add_argument('--legacy-egfr-name', action='store_true',
                        help='When fetching EGFR, also write the legacy filename '
                             'egfr_bioactivity_curated.csv to keep notebooks/app '
                             'working unchanged.')
    parser.add_argument('--extra-sources', default='',
                        help='Comma-separated extra sources for EGFR dataset '
                             'expansion. Choices: pubchem, bindingdb, excape.')
    parser.add_argument('--pubchem-aid', type=int, default=1851,
                        help='PubChem BioAssay AID (default: 1851 — NCATS EGFR '
                             'kinase panel).')
    parser.add_argument('--bindingdb-tsv', default=None,
                        help='Path to local BindingDB_All.tsv dump.')
    parser.add_argument('--excape-tsv', default=None,
                        help='Path to local ExCAPE-DB TSV (.tsv or .tsv.xz).')
    parser.add_argument('--unified-out', default='egfr_unified.csv',
                        help='Filename (in --out-dir) for the merged EGFR '
                             'dataset across sources.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokens = [t for t in args.targets.split(',') if t.strip()]
    try:
        resolved = [resolve_target(t) for t in tokens]
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2

    per_target: dict[str, pd.DataFrame] = {}
    for alias, chembl_id in resolved:
        per_target[alias] = fetch_target(alias, chembl_id, args.out_dir)

    # Backwards-compat: keep the original EGFR filename
    if 'EGFR' in per_target and not per_target['EGFR'].empty:
        legacy = os.path.join(args.out_dir, 'egfr_bioactivity_curated.csv')
        if args.legacy_egfr_name or len(per_target) == 1:
            cols = ['molecule_chembl_id', 'canonical_smiles', 'IC50_nM', 'pIC50']
            per_target['EGFR'][cols].to_csv(legacy, index=False)
            print(f'(legacy) Also wrote {legacy}')

    if len(per_target) > 1:
        build_multi_target_table(per_target, args.out_dir)

    # ----- Phase 3: dataset expansion via extra sources --------------------
    extra = [s.strip().lower() for s in args.extra_sources.split(',') if s.strip()]
    if extra:
        if 'EGFR' not in per_target or per_target['EGFR'].empty:
            print('warning: --extra-sources requested but ChEMBL EGFR data is '
                  'empty; cannot merge.', file=sys.stderr)
        else:
            extras: list[pd.DataFrame] = []
            if 'pubchem' in extra:
                extras.append(fetch_pubchem_aid(args.pubchem_aid, args.out_dir))
            if 'bindingdb' in extra:
                extras.append(load_bindingdb(args.bindingdb_tsv, args.out_dir))
            if 'excape' in extra:
                extras.append(load_excape(args.excape_tsv, args.out_dir))
            merge_egfr_sources(per_target['EGFR'], extras,
                               args.out_dir, args.unified_out)

    print('\nDone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
