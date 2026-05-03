"""Executes the notebook with Windows asyncio + UTF-8 fix."""
import asyncio
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('bioactivity_prediction.ipynb', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Allow individual cells to fail without stopping execution
nb.metadata.setdefault('execution', {})['allow_errors'] = True

client = NotebookClient(
    nb,
    timeout=7200,
    kernel_name='drugdisc',
    resources={'metadata': {'path': '.'}},
)

print('Executing notebook...')
errors = []
try:
    client.execute()
    print('SUCCESS - Execution complete.')
except CellExecutionError as e:
    msg = str(e).encode('utf-8', 'replace').decode('utf-8')[:200]
    print(f'Cell error (partial output saved): {msg}')
    errors.append(msg)
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {str(e).encode("utf-8", "replace").decode()[:300]}')
finally:
    with open('bioactivity_prediction_executed.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

total_code = sum(1 for c in nb.cells if c.cell_type == 'code')
executed = sum(1 for c in nb.cells if c.cell_type == 'code'
                 and c.get('execution_count') is not None)
cell_errors = sum(1 for c in nb.cells for o in c.get('outputs', [])
                  if o.get('output_type') == 'error')
print('Saved: bioactivity_prediction_executed.ipynb')
print(f'  Code cells: {total_code}, executed: {executed}, cell errors: {cell_errors}')
