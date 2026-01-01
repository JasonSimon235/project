import os
import pandas as pd
from backend import backend_app
import time

# Configuration
IMAGE_FOLDER = "benchmark_images"
OUTPUT_CSV = "evaluation_results.csv"
TEST_CONTEXT = "This image is from a general knowledge dataset used for accessibility testing."

def run_batch_test():
    results = []
    
    # 1. Check Image Folder
    if not os.path.exists(IMAGE_FOLDER):
        print(f"FATAL ERROR: The folder '{IMAGE_FOLDER}' does not exist.")
        return

    image_files = os.listdir(IMAGE_FOLDER)
    # Filter for actual images
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"DEBUG: Found {len(image_files)} images in '{IMAGE_FOLDER}'.")
    
    if len(image_files) == 0:
        print("FATAL ERROR: No images found. Run extract_images.py first!")
        return

    # 2. Loop through images
    print("Starting generation... (This might take a minute)")
    
    for i, filename in enumerate(image_files):
        # Stop after 50 to save time/quota
        if i >= 50: break
            
        file_path = os.path.join(IMAGE_FOLDER, filename)
        description = "ERROR: Script crashed before API call" # Default value
        
        try:
            with open(file_path, "rb") as f:
                image_bytes = f.read()
                
                # Construct Prompt
                prompt = app.construct_prompt(TEST_CONTEXT)
                
                # Query API
                # We expect app.query_blip2_api to return a string, even if it's an error message
                description = app.query_blip2_api(image_bytes, prompt) 
                
        except Exception as e:
            print(f"CRITICAL ERROR on {filename}: {e}")
            description = f"ERROR: {str(e)}"

        # Clean up description
        clean_description = str(description).replace("\n", " ").strip()
        
        # Print progress
        print(f"[{i+1}/{len(image_files)}] {filename} -> {clean_description[:40]}...")
        
        # Add to results list
        results.append({
            "image_id": filename,
            "generated_description": clean_description
        })
        
        # Optional: Sleep briefly to avoid hitting API rate limits too hard
        time.sleep(0.5)

    # 3. Force Save Results
    print(f"Saving {len(results)} rows to CSV...")
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"DONE! Check {OUTPUT_CSV} now.")

if __name__ == "__main__":
    run_batch_test()