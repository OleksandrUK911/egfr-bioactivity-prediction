"""Unit tests for featurisation and curation utilities."""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

# ---------------------------------------------------------------------------
# Helpers reused across tests
# ---------------------------------------------------------------------------

ERLOTINIB = "Cn1cnc2c1ncnc2Nc1ccc(OCCOC)c(OCCOC)c1"
INVALID_SMILES = "this-is-not-a-smiles"

# ---------------------------------------------------------------------------
# Morgan fingerprint tests
# ---------------------------------------------------------------------------

def smiles_to_morgan(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def test_fingerprint_valid_smiles():
    fp = smiles_to_morgan(ERLOTINIB)
    assert fp is not None
    assert fp.shape == (2048,)
    assert fp.dtype == np.uint8
    assert fp.sum() > 0


def test_fingerprint_invalid_smiles():
    fp = smiles_to_morgan(INVALID_SMILES)
    assert fp is None


def test_fingerprint_radius_affects_output():
    fp2 = smiles_to_morgan(ERLOTINIB, radius=2)
    fp3 = smiles_to_morgan(ERLOTINIB, radius=3)
    assert fp2 is not None and fp3 is not None
    assert not np.array_equal(fp2, fp3), "Different radii should produce different fingerprints"


def test_fingerprint_bit_length():
    for n in (512, 1024, 2048):
        fp = smiles_to_morgan(ERLOTINIB, n_bits=n)
        assert fp is not None
        assert fp.shape == (n,)


# ---------------------------------------------------------------------------
# Lipinski descriptor tests
# ---------------------------------------------------------------------------

def lipinski_report(smiles: str) -> dict | None:
    from rdkit.Chem import Crippen, Descriptors, Lipinski
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw   = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {
        "MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba, "TPSA": tpsa,
        "Violations": violations, "Ro5 pass": violations <= 1,
    }


def test_lipinski_erlotinib():
    report = lipinski_report(ERLOTINIB)
    assert report is not None
    assert report["MW"] == pytest.approx(373.4, abs=1.0)
    assert report["Ro5 pass"] is True
    assert report["Violations"] == 0


def test_lipinski_invalid_smiles():
    report = lipinski_report(INVALID_SMILES)
    assert report is None


def test_lipinski_keys_present():
    report = lipinski_report(ERLOTINIB)
    for key in ("MW", "LogP", "HBD", "HBA", "TPSA", "Violations", "Ro5 pass"):
        assert key in report


# ---------------------------------------------------------------------------
# pIC50 conversion tests
# ---------------------------------------------------------------------------

def test_pic50_conversion():
    ic50_nm = 1.0       # 1 nM → pIC50 = 9.0
    pic50 = -np.log10(ic50_nm * 1e-9)
    assert pic50 == pytest.approx(9.0, abs=1e-6)


def test_pic50_range_filter():
    values = [1.5, 3.0, 6.5, 12.0, 14.0, -1.0]
    filtered = [v for v in values if 3.0 <= v <= 12.0]
    assert filtered == [3.0, 6.5, 12.0]


# ---------------------------------------------------------------------------
# Data curation smoke test
# ---------------------------------------------------------------------------

def test_curation_drops_invalid_smiles():
    df = pd.DataFrame({
        "canonical_smiles": [ERLOTINIB, INVALID_SMILES, ERLOTINIB],
        "pIC50": [8.0, 7.0, 6.5],
    })
    valid_mask = df["canonical_smiles"].apply(
        lambda s: Chem.MolFromSmiles(s) is not None
    )
    result = df[valid_mask].reset_index(drop=True)
    assert len(result) == 2
    assert INVALID_SMILES not in result["canonical_smiles"].values
