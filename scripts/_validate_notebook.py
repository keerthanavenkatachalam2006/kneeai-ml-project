"""Validate the generated Colab notebook: nbformat schema plus per-cell syntax.

Also guards against the failure that actually bit us: identifiers arriving with
`*` where `_` belongs, which happens when cell text is round-tripped through
rendered Markdown.
"""

import ast
import re
import sys
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks/ml/01_train_oaizib_colab.ipynb"

nb = nbformat.read(str(NB), as_version=4)
nbformat.validate(nb)
print("nbformat schema: VALID")

code_cells = [c for c in nb.cells if c.cell_type == "code"]
print(f"cells: {len(nb.cells)} total, {len(code_cells)} code")

failures = 0

# `for *name in` / `train*tf` — an asterisk touching an identifier character.
MANGLED = re.compile(r"(?<![\s*=+\-/(,])\*(?=[A-Za-z_])|(?<=[A-Za-z_])\*(?=[A-Za-z_])")

for i, cell in enumerate(code_cells, 1):
    src = cell.source

    for lineno, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#") or '"' in line or "'" in line:
            continue
        if MANGLED.search(line):
            failures += 1
            print(f"  cell {i} line {lineno}: possible markdown-mangled underscore: {stripped[:80]}")

    cleaned_lines = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            indent = line[: len(line) - len(stripped)]
            cleaned_lines.append(f"{indent}pass")
        else:
            cleaned_lines.append(line)

    try:
        ast.parse("\n".join(cleaned_lines))
    except SyntaxError as exc:
        failures += 1
        print(f"  cell {i}: SYNTAX ERROR line {exc.lineno}: {exc.msg}")

print(f"checks: {'ALL PASS' if not failures else f'{failures} FAILED'}")
print(f"size: {NB.stat().st_size / 1024:.1f} KB")
sys.exit(1 if failures else 0)
