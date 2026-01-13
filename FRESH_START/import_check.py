import ast, glob, os, importlib

files = [
    p
    for p in glob.glob("**/*.py", recursive=True)
    if ".venv" not in p and ".venv" not in p
]
modules = set()
for f in files:
    try:
        with open(f, "r", encoding="utf-8") as fh:
            node = ast.parse(fh.read(), filename=f)
    except Exception:
        continue
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                modules.add(n.module.split(".")[0])

ignore = {"__future__"}
results = []
for m in sorted(modules):
    if not m or m in ignore:
        continue
    try:
        importlib.import_module(m)
        results.append((m, True, ""))
    except Exception as e:
        results.append((m, False, str(e)))

ok = [r for r in results if r[1]]
fail = [r for r in results if not r[1]]
print(f"Checked {len(results)} modules: {len(ok)} ok, {len(fail)} failed")
for m, okv, err in fail:
    print(f"FAILED: {m} -> {err}")

with open("IMPORT_CHECK_RESULTS.txt", "w", encoding="utf-8") as out:
    out.write(f"Checked {len(results)} modules: {len(ok)} ok, {len(fail)} failed\n\n")
    for m, okv, err in results:
        out.write(f"{m}\t{okv}\t{err}\n")
