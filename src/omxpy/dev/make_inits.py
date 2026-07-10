# src/dev/make_inits.py
import ast
from pathlib import Path
from datetime import datetime

src = Path(__file__).parent.parent
log_file = Path(__file__).parent / "log" / "make_inits.log"
log_file.parent.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_lines = [f"\n[{timestamp}]"]

for pkg_dir in sorted(src.iterdir()):
    if not pkg_dir.is_dir():
        continue

    exports = []

    for py_file in sorted(pkg_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        tree = ast.parse(py_file.read_text())
        names = [
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and not node.name.startswith("_")
        ]

        if names:
            exports.append(f"from .{py_file.stem} import {', '.join(names)}")

    if exports:
        init = pkg_dir / "__init__.py"
        init.write_text("\n".join(exports) + "\n")
        print(f"wrote {init}")
        log_lines.append(f"  {pkg_dir.name}/__init__.py")
        for line in exports:
            log_lines.append(f"    {line}")

with open(log_file, "a") as f:
    f.write("\n".join(log_lines) + "\n")

print(f"logged to {log_file}")