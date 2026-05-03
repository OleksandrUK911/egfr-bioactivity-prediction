"""Fix mojibake (UTF-8 bytes stored as cp1252/latin-1) in notebook markdown cells."""
import json
import sys
from pathlib import Path

REPLACEMENT_CHAR = '�'

# Build a reverse map: Unicode char -> original byte (covering all cp1252 + Latin-1)
# cp1252 special range for 0x80-0x9F (the rest of 0x00-0xFF is Latin-1 identical)
_CP1252_SPECIAL = {
    0x80: '€', 0x82: '‚', 0x83: 'ƒ', 0x84: '„',
    0x85: '…', 0x86: '†', 0x87: '‡', 0x88: 'ˆ',
    0x89: '‰', 0x8a: 'Š', 0x8b: '‹', 0x8c: 'Œ',
    0x8e: 'Ž', 0x91: '‘', 0x92: '’', 0x93: '“',
    0x94: '”', 0x95: '•', 0x96: '–', 0x97: '—',
    0x98: '˜', 0x99: '™', 0x9a: 'š', 0x9b: '›',
    0x9c: 'œ', 0x9e: 'ž', 0x9f: 'Ÿ',
}
# Char -> byte lookup (cp1252 wins for the special range; Latin-1 for everything else)
_CHAR_TO_BYTE: dict[str, int] = {}
for _b in range(0x100):
    _c = _CP1252_SPECIAL.get(_b, chr(_b))
    if _c not in _CHAR_TO_BYTE:  # cp1252 special chars take priority
        _CHAR_TO_BYTE[_c] = _b
# Also allow Latin-1 C1 controls (0x80-0x9F as chr(b)) that cp1252 doesn't assign
for _b in range(0x80, 0xa0):
    _c = chr(_b)
    if _c not in _CHAR_TO_BYTE:
        _CHAR_TO_BYTE[_c] = _b


def _try_mojibake(chunk: str) -> str | None:
    """Return UTF-8 decoded string if chunk looks like mojibake, else None."""
    try:
        raw = bytes(_CHAR_TO_BYTE[c] for c in chunk)
    except KeyError:
        return None  # contains chars that can't be a cp1252/latin-1 byte
    try:
        decoded = raw.decode('utf-8')
    except UnicodeDecodeError:
        return None
    if decoded == chunk or REPLACEMENT_CHAR in decoded:
        return None
    if len(decoded) >= len(chunk):
        return None  # must get shorter
    return decoded


def fix_mojibake_greedy(text: str) -> str:
    """Greedy left-to-right 4→3→2 char window mojibake fixer."""
    result = []
    i = 0
    while i < len(text):
        fixed = False
        for length in [4, 3, 2]:
            if i + length > len(text):
                continue
            decoded = _try_mojibake(text[i:i + length])
            if decoded is not None:
                result.append(decoded)
                i += length
                fixed = True
                break
        if not fixed:
            result.append(text[i])
            i += 1
    return ''.join(result)


def fix_source(source_lines):
    # Join everything first (handles char-by-char storage as well as line storage)
    full = ''.join(source_lines)
    fixed = fix_mojibake_greedy(full)
    if fixed == full:
        return source_lines  # nothing changed, preserve original structure
    # Re-split into lines (keep trailing \n with each line — standard nbformat)
    return fixed.splitlines(keepends=True)


def fix_notebook(path):
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)

    changed_cells = 0
    for cell in nb['cells']:
        src = cell.get('source', [])
        if not src:
            continue
        fixed = fix_source(src)
        if fixed != src:
            cell['source'] = fixed
            changed_cells += 1

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"{path.name}: fixed {changed_cells} cells", flush=True)
    return changed_cells


def _mojibake_of(utf8_bytes: bytes) -> str:
    """Re-encode UTF-8 bytes as cp1252 chars to produce the mojibake string."""
    result = []
    for b in utf8_bytes:
        result.append(_CP1252_SPECIAL.get(b, chr(b)))
    return ''.join(result)


# --- Verification using programmatically generated test cases ---
TEST_CASES = [
    # (utf8_bytes, expected_char)
    (b'\xc2\xa7',     '§'),  # § section sign
    (b'\xe2\x80\x94', '—'),  # — em-dash
    (b'\xe2\x80\x93', '–'),  # – en-dash
    (b'\xce\x94',     'Δ'),  # Δ delta
    (b'\xe2\x89\x88', '≈'),  # ≈ approx
    (b'\xe2\x89\xa5', '≥'),  # ≥ >=
    (b'\xc2\xb1',     '±'),  # ± plus-minus
    (b'\xc2\xb5',     'µ'),  # µ mu
    (b'\xc2\xb7',     '·'),  # · middle dot
    (b'\xc2\xb2',     '²'),  # ² superscript 2
    (b'\xe2\x80\x9c', '“'),  # " left double quote
    (b'\xe2\x80\x9d', '”'),  # " right double quote
    (b'\xe2\x80\x98', '‘'),  # ' left single quote
    (b'\xe2\x80\x99', '’'),  # ' right single quote
    (b'\xcf\x80',     'π'),  # π pi
    (b'\xce\xb1',     'α'),  # α alpha
]

# Chars that must NOT be changed (already correct Unicode)
PRESERVE_CASES = [
    '²',  # ²
    '±',  # ±
    'µ',  # µ
    '·',  # ·
    '§',  # §
    '—',  # —
    'Δ',  # Δ
    '≈',  # ≈
    'R²=0.735',  # R²=0.735 mixed line
    'RMSE±SD',   # RMSE±SD
]

print("Running verification tests...")
all_pass = True

for utf8_bytes, expected in TEST_CASES:
    mojibake = _mojibake_of(utf8_bytes)
    result = fix_mojibake_greedy(mojibake)
    ok = result == expected
    if not ok:
        all_pass = False
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}]  {ascii(mojibake):30s}  ->  {ascii(result):20s}  (expected {ascii(expected)})")

print()
for char in PRESERVE_CASES:
    result = fix_mojibake_greedy(char)
    ok = result == char
    if not ok:
        all_pass = False
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}]  preserve {ascii(char):30s}  ->  {ascii(result)}")

print()
if not all_pass:
    print("ERROR: Some tests failed. Aborting.")
    sys.exit(1)

print("All tests passed. Applying fix to notebooks...")
base = Path(__file__).parent

notebooks = [
    base / "bioactivity_prediction.ipynb",
    base / "bioactivity_prediction_executed.ipynb",
]

total = 0
for nb_path in notebooks:
    if nb_path.exists():
        total += fix_notebook(nb_path)
    else:
        print(f"SKIP (not found): {nb_path.name}")

print(f"\nDone. Total cells modified: {total}")
