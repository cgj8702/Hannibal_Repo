import ast
import os
import sys


# Files to check (excluding _archive)
FILES_TO_CHECK = [
    "auto_editor_parallel.py",
    "cleanup_workspace.py",
    "hannibal_server.py",
    "ingest.py",
    "list_models_debug.py",
    "parse_screenplays.py"
]

# Mapping import names to pip package names (common overrides)
IMPORT_MAP = {
    "google.genai": "google-genai",
    "google.generativeai": "google-generativeai", # Legacy but still used in list_models_debug
    "google.cloud": "google-cloud-aiplatform", # Approximation, often part of google-cloud-*
    "google.auth": "google-auth",
    "langchain_community": "langchain-community",
    "langchain_google_genai": "langchain-google-genai",
    "langchain_core": "langchain-core",
    "chromadb": "chromadb",
    "fastapi": "fastapi",
    "pydantic": "pydantic",
    "uvicorn": "uvicorn",
    "pdfplumber": "pdfplumber",
    "pypdf": "pypdf",
    "tqdm": "tqdm",
    "PIL": "Pillow",
    "dotenv": "python-dotenv",
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "cv2": "opencv-python"
}

def get_requirements():
    if not os.path.exists("requirements.txt"):
        return set()
    with open("requirements.txt") as f:
        # Simple parsing: remove version specs, comments, whitespace
        lines = f.readlines()
        cleaned = set()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip version specifiers like ==, >=, <
            pkg = line.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0]
            cleaned.add(pkg.lower())
        return cleaned

def get_imports(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            print(f"Skipping {filepath} (Syntax Error)")
            return set()
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

def is_std_lib(module_name):
    # Basic check for standard library (incomplete but functional for most)
    # Using a hardcoded list of common std libs to avoid external dependency for this script if possible
    # Or using sys.stdlib_module_names if available (Python 3.10+)
    if sys.version_info >= (3, 10):
        return module_name in sys.stdlib_module_names
    
    # Fallback/Supplemental list
    std_libs = {
        "os", "sys", "time", "json", "re", "math", "random", "datetime", 
        "collections", "itertools", "functools", "logging", "typing", 
        "pathlib", "shutil", "subprocess", "glob", "pickle", "copy",
        "concurrent", "threading", "io", "csv", "argparse", "ast", "textwrap", "enum"
    }
    return module_name in std_libs

def main():
    requirements = get_requirements()
    print(f"Loaded {len(requirements)} packages from requirements.txt")
    
    all_imports = set()
    missing_packages = set()
    
    print("-" * 40)
    for file in FILES_TO_CHECK:
        if not os.path.exists(file):
            print(f"Warning: {file} not found.")
            continue
            
        imports = get_imports(file)
        # Filter std lib
        third_party = {imp for imp in imports if not is_std_lib(imp)}
        
        # Check against requirements
        for imp in third_party:
            # Check map first
            pkg_name = IMPORT_MAP.get(imp, imp) # Default to import name if not mapped
            
            # Check if pkg_name or the original import is in requirements
            # (Case insensitive check)
            if pkg_name.lower() not in requirements and imp.lower() not in requirements:
                # Special logic for google namespace packages which are messy
                if imp == "google" and any(r.startswith("google-") for r in requirements):
                    continue 
                    
                print(f"[MISSING] {file}: imports '{imp}' (Package: '{pkg_name}'?)")
                missing_packages.add(pkg_name)
    
    print("-" * 40)
    if missing_packages:
        print("Recommended additions to requirements.txt:")
        for pkg in sorted(missing_packages):
            print(pkg)
    else:
        print("All imports valid and accounted for in requirements.txt (or standard library).")

if __name__ == "__main__":
    main()
