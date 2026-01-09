import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm


# 1. Define your Cleaning Function
def clean_screenplay_text(text):

    # Removes "BIG FISH - FINAL [any number]."
    text = re.sub(r"BIG FISH - FINAL  \d+\.", "", text)

    # 1. Kill numbers at the START of the header (e.g., "2 INT.")
    text = re.sub(r"^\s*\d+\s+(?=INT\.|EXT\.)", "", text, flags=re.MULTILINE)

    # The "Squish" Killer
    # Logic: If a line ends with a closing parenthesis ')' followed by digits,
    # keep the parenthesis but delete the digits.
    # Checks for: "(1987)4", "(1987) 4", "(1987)24"
    text = re.sub(r"(INT\.|EXT\..*?)\)\s*\d+\s*$", r"\1)", text, flags=re.MULTILINE)

    # 3. Kill "Ghost" numbers floating in the MIDDLE (e.g., "FRONT 4 HALL")
    # This removes standalone digits (1-3 length) inside header lines, but ignores years (1998).
    # We run it twice just in case there are two ghost numbers.
    for _ in range(2):
        text = re.sub(
            r"(^(?:INT\.|EXT\.).*?)\s+\b\d{1,3}\b\s+(.*?$)",
            r"\1 \2",
            text,
            flags=re.MULTILINE,
        )

    # Matches (cont...) followed by ANYTHING until the closing parenthesis
    text = re.sub(r"\s*\(cont.*?\)", "", text, flags=re.IGNORECASE)

    # Remove Transition markings (Optional, but often good for RAG)
    # CUT TO:, DISSOLVE TO:, etc.
    text = re.sub(
        r"^\s*(CUT|DISSOLVE|FADE|TRANSITION|MATCH CUT|COME|CROSSFADE)\s+TO:\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    return text


# --- MAIN PIPELINE ---

loader = PyPDFLoader("C:/Users/carly/Documents/Coding/movie_scripts/big_fish.pdf")
documents = loader.load()

# 3. CLEAN (The New Step)
print("Cleaning text...")
for doc in documents:
    doc.page_content = clean_screenplay_text(doc.page_content)

    # Check Page 5 (usually a busy page)
# print("--- DEBUG: PREVIEW OF PAGE 5 AFTER CLEANING ---")
# print(documents[5].page_content)
# print("-----------------------------------------------")

# 4. NOW run your Semantic Chunker
# (Insert your previous SemanticChunker code here)
# ...

# Initialize Gemini's Brain (The Embeddings)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# The Magic: Semantic Chunking
# 'percentile' means: "If the topic changes drastically (top 95%), cut the chunk."
text_splitter = SemanticChunker(
    embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95
)

print(f"Loaded {len(documents)} pages.")

# TEST MODE: Only try the first 10 pages to see if it works at all
# If this finishes fast, you know the code is good, just slow.
subset_documents = documents[:10]

print("Semantic Chunking (Test Run)...")
# ... setup chunker ...
chunks = text_splitter.split_documents(subset_documents)
print(f"Created {len(chunks)} chunks from the first 10 pages.")

# 1. SETUP THE LIST TO HOLD RESULTS
final_chunks = []

# 2. DEFINE BATCH SIZE
# We process 5 pages at a time so we can see progress
batch_size = 5

# 3. THE LOOP WITH THE BAR
# This splits your list of 100 pages into groups of 5
# tqdm wraps the loop to create the visual bar
batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]

print("Starting Semantic Chunking...")
for batch in tqdm(batches, desc="Processing Script"):
    # Chunk just these 5 pages
    new_chunks = text_splitter.split_documents(batch)

    # Add them to the main pile
    final_chunks.extend(new_chunks)

print(f"\nDone! Created {len(final_chunks)} total chunks.")

# Let's peek at a random chunk to see if it captured a full scene
sample_chunk = chunks[5]
# Print the full text
print(sample_chunk.page_content)

# Print how many characters it is
print(f"\nTotal characters in this chunk: {len(sample_chunk.page_content)}")
