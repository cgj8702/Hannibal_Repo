import ast, glob, os

py_files = [p for p in glob.glob("**/*.py", recursive=True) if ".venv" not in p]
module_names = set()
for p in py_files:
    name = os.path.splitext(os.path.basename(p))[0]
    module_names.add(name)

matches = []
for p in py_files:
    try:
        with open(p, "r", encoding="utf-8") as fh:
            node = ast.parse(fh.read())
    except Exception:
        continue
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                mod = alias.name.split(".")[0]
                if (
                    mod in module_names
                    and mod != os.path.splitext(os.path.basename(p))[0]
                ):
                    matches.append((p, "import", mod))
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                mod = n.module.split(".")[0]
                if (
                    mod in module_names
                    and mod != os.path.splitext(os.path.basename(p))[0]
                ):
                    matches.append((p, "from", mod))

for p, typ, mod in sorted(matches):
    print(f"{p}: {typ} {mod}")

with open("LOCAL_IMPORTS_FOUND.txt", "w", encoding="utf-8") as out:
    for p, typ, mod in sorted(matches):
        out.write(f"{p}: {typ} {mod}\n")
