import os

def combine_screenplays():
    parsed_folder = "ParsedScreenplays"
    output_filename = "Hannibal_All_Screenplays.txt"
    output_path = os.path.join(os.getcwd(), output_filename)
    
    parsed_folder_abs = os.path.join(os.getcwd(), parsed_folder)
    
    if not os.path.exists(parsed_folder_abs):
        print(f"Error: The folder '{parsed_folder}' does not exist.")
        return

    # List files and sort them to ensure episode order
    files = [f for f in os.listdir(parsed_folder_abs) if f.casefold().endswith(".txt")]
    files.sort() # Alphabetical sort handles 1x01, 1x02 correctly

    if not files:
        print("No text files found to combine.")
        return

    print(f"Found {len(files)} screenplays to combine.")
    
    with open(output_path, "w", encoding="utf-8") as outfile:
        for filename in files:
            file_path = os.path.join(parsed_folder_abs, filename)
            print(f"Adding: {filename}")
            
            # Add a clear separator/header for each screenplay
            outfile.write(f"\n{'='*50}\n")
            outfile.write(f"START OF SCREENPLAY: {filename}\n")
            outfile.write(f"{'='*50}\n\n")
            
            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                    outfile.write("\n") # Ensure separation between files
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
    print(f"\nSuccessfully combined all screenplays into: {output_path}")

if __name__ == "__main__":
    combine_screenplays()
