
import json
import os
import glob
import re

# Configuration
INPUT_PATTERN = "output_v2/*.jsonl" # Path where we will download results
OUTPUT_DIR = "Final_Proofread_Screenplays_V2"

def process_v2_results():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    result_files = glob.glob(INPUT_PATTERN)
    if not result_files:
        print("No result files found in output_v2/")
        return

    print(f"Processing {len(result_files)} result files...")
    
    for file_path in result_files:
        print(f"Reading {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    
                    # 1. Identify File
                    # Vertex AI Batch Input URI is in entry['instance']['content'] or request info
                    # We might need to map it back if the output doesn't preserve filename clearly
                    # Usually entry['instance'] contains the original input.
                    if 'instance' in entry and 'content' in entry['instance']:
                         # 'content' might be the prompt itself if we successfully mapped it?
                         # Or it's the GCS URI of the input file if we used file-based input?
                         # In create_batch_request_v2.py, we put the PROMPT in "contents".
                         # We didn't allow passing a filename in metadata easily via the standard structure.
                         # CHECK: Did we lose the filename mapping?
                         
                         # In V1, we used "request" field to match?
                         # Actually, in create_batch_request_v2.py, we wrote:
                         # request_entry = { "request": { "contents": ... } }
                         # We did NOT include a custom ID.
                         # This is a potential issue! We won't know which file is which unless we infer it from content.
                         # OR we rely on the order? (Risky)
                         pass
                    
                    # Alternative: We infer the episode from the text content.
                    # The prompt included the whole screenplay.
                    # The response includes the "corrected_text" or "corrected_segments".
                    # We can look at the first scene header or "HANNIBAL" title card equivalent.
                    
                    # 2. Extract Prediction
                    # prediction['candidates'][0]['content']['parts'][0]['text']
                    # This text should be valid JSON.
                    if 'prediction' in entry:
                        candidates = entry['prediction'].get('candidates', [])
                        if candidates:
                            raw_text = candidates[0]['content']['parts'][0]['text']
                            
                            # Parse the Model's JSON Output
                            try:
                                model_output = json.loads(raw_text)
                                
                                # Reconstruct Content
                                full_text = ""
                                
                                # Check if we have 'full_reconstructed_text' (if model provided it)
                                if 'full_reconstructed_text' in model_output:
                                    full_text = model_output['full_reconstructed_text']
                                elif 'corrected_segments' in model_output:
                                    segments = model_output['corrected_segments']
                                    for seg in segments:
                                        # Prefer corrected_text
                                        full_text += seg.get('corrected_text', '') + "\n\n"
                                else:
                                    # Fallback
                                    full_text = str(model_output)
                                
                                # 3. Determine Filename
                                filename = "Unknown_Screenplay.txt"
                                
                                # Strategy A: Check strict headers in output (if present)
                                match = re.search(r'Ep\. #(\d+)', full_text)
                                if match:
                                    ep_num = match.group(1)
                                    filename = f"Proofread_Hannibal_{ep_num}_V2.txt"
                                
                                # Strategy B: Check the INPUT prompt for the original text artifacts
                                elif 'instance' in entry:
                                    try:
                                        # Drill down: instance -> [request] -> contents -> parts -> text
                                        instance_data = entry['instance']
                                        if 'request' in instance_data:
                                            input_prompt = instance_data['request']['contents'][0]['parts'][0]['text']
                                        else:
                                            input_prompt = instance_data['contents'][0]['parts'][0]['text']
                                        
                                        # Look for "Hannibal_1x01" style patterns or title cards in input
                                        # Example input text: "Screenplay Content:\n... HANNIBAL ... Ep. #101"
                                        
                                        # Regex for "Ep. #101" or "Prod. #101"
                                        ep_match = re.search(r'(?:Ep\.|Prod\.) #(\d+)', input_prompt, re.IGNORECASE)
                                        if ep_match:
                                            ep_num = ep_match.group(1)
                                            # Convert 101 to 1x01 logic?
                                            if len(ep_num) == 3:
                                                season = ep_num[0]
                                                episode = ep_num[1:]
                                                filename = f"Proofread_Hannibal_{season}x{episode}_V2.txt"
                                            else:
                                                filename = f"Proofread_Hannibal_{ep_num}_V2.txt"
                                        
                                        # Fallback: Look for "Aperitif", "Entree" etc if we had a map (we don't easily)
                                    except Exception as e:
                                        # print(f"Filename inference failed: {e}")
                                        pass

                                # Fallback Strategy C: Use Hash or Order (Last Resort)
                                if filename == "Unknown_Screenplay.txt":
                                     # Use existing count
                                    existing_count = len(glob.glob(os.path.join(OUTPUT_DIR, "Unknown*.txt")))
                                    filename = f"Unknown_Screenplay_{existing_count + 1}.txt"

                                # Save
                                out_path = os.path.join(OUTPUT_DIR, filename)
                                with open(out_path, 'w', encoding='utf-8') as out_f:
                                    out_f.write(full_text)
                                print(f"Saved {filename}")
                                
                            except json.JSONDecodeError:
                                print("Model output was not valid JSON.")
                                # Save raw text for debugging
                                # ...
                                pass

                except Exception as e:
                    print(f"Error processing line: {e}")

if __name__ == "__main__":
    process_v2_results()
