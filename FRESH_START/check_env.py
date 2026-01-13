import os

print("GOOGLE_API_KEY set:", bool(os.environ.get("GOOGLE_API_KEY")))
print(
    "GOOGLE_API_KEY value sample:",
    (
        os.environ.get("GOOGLE_API_KEY")[:4] + "..."
        if os.environ.get("GOOGLE_API_KEY")
        else None
    ),
)
print("faiss_index exists:", os.path.exists("faiss_index"))
print("cwd:", os.getcwd())
