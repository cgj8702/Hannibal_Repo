import os
import glob
import re
import time
from google import genai
from google.genai import types
import tqdm

def add_line_numbers(text, start_line=1):
    """Adds line numbers to the text for easier reference."""
    lines = text.split('\n')
    numbered_lines = [f"{start_line+i}: {line}" for i, line in enumerate(lines)]
    return '\n'.join(numbered_lines), start_line + len(lines)

def split_into_chunks(text, max_chars=7000):
    """Splits the text into chunks respecting max chars.
    Using 7000 chars (approx 1750 tokens) to stay well within 15k TPM limit when combined with delays.
    """
    scene_pattern = re.compile(r'(=== \d+ .*? ===)')
    parts = scene_pattern.split(text)
    chunks = []
    current_chunk = ""
    if parts:
         current_chunk = parts[0]
         for i in range(1, len(parts), 2):
             header = parts[i]
             content = parts[i+1] if i+1 < len(parts) else ""
             full_scene = header + content
             if len(current_chunk) + len(full_scene) > max_chars:
                 chunks.append(current_chunk)
                 current_chunk = full_scene
             else:
                 current_chunk += full_scene
         if current_chunk:
             chunks.append(current_chunk)
    else:
        chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    return chunks

def proofread_chunk(client, text, filename, chunk_index, retries=5):
    """Sends a chunk to the model."""
    prompt = (
        "You are a professional proofreader. Proofread the following screenplay text segment, "
        "identifying typos, formatting inconsistencies, or logical errors. "
        "For each error you find:\n"
        "1. Cite the Episode (from filename), Scene (from headers), and Line Number.\n"
        "2. Explain the error.\n"
        "3. Provide the corrected text for that section.\n\n"
        "After listing the errors, provide the full validated and corrected version of the text segment. "
        "Do not leave out any part of the original text in your corrected version.\n\n"
        f"Filename: {filename} (Chunk {chunk_index})\n\n"
        "Screenplay Content (with line numbers):\n"
        f"{text}"
    )

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model='gemma-3-27b-it',
                contents=prompt,
                 config=types.GenerateContentConfig(
                    response_modalities=["TEXT"],
                 )
            )
            return response.text
        except types.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait_time = (2 ** attempt) * 30 # Aggressive backoff starting at 30s
                print(f"\n[Chunk {chunk_index}] Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n[Chunk {chunk_index}] Client Error: {e}")
                return f"[Error processing chunk {chunk_index}: {str(e)}]"
        except Exception as e:
            print(f"\n[Chunk {chunk_index}] Exception: {e}")
            if attempt < retries - 1:
                time.sleep(10)
            else:
                return f"[Error processing chunk {chunk_index}: {str(e)}]"
    
    return f"[Failed to process chunk {chunk_index} after {retries} retries]"

def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return

    client = genai.Client(api_key=api_key)

    input_dir = "ParsedScreenplays"
    output_dir = "ProofreadScreenplays"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    
    if not files:
        print(f"No text files found in {input_dir}")
        return

    print(f"Found {len(files)} screenplays to process.")
    print("Using strict rate limiting: Max 7000 chars/chunk, 25s delay between chunks.")

    for file_path in tqdm.tqdm(files, desc="Proofreading Screenplays"):
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, f"Proofread_{filename}")
        
        if os.path.exists(output_path):
             print(f"Skipping {filename} (already exists)")
             continue
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Proofreading Output for {filename}\n")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        numbered_content, _ = add_line_numbers(content)
        
        chunks = split_into_chunks(numbered_content, max_chars=7000)
        
        print(f"\nProcessing {filename} in {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            result = proofread_chunk(client, chunk, filename, i+1)
            
            output_text = f"\n=== Chunk {i+1} Output ===\n{result}\n"
            
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(output_text)

            print(f"Finished Chunk {i+1}/{len(chunks)}")
            
            # 25s delay to handle 15k token/min limit
            # 7k chars ~ 1750 tokens input + output overhead
            # 15000 / (1750*2) approx 4 chunks/min -> 15s delay strict minimum.
            # Using 25s to be extremely safe against quota blocks.
            time.sleep(25) 

    print("Proofreading complete. Check the 'ProofreadScreenplays' directory.")

if __name__ == "__main__":
    main()
