import os
import json
import pandas as pd
from collections import defaultdict

def find_duplicates(folder_path):
    csv_files_content = defaultdict(list)
    json_csv_mapping = {}

    # Read all JSON files and map to their corresponding CSV files
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            json_path = os.path.join(folder_path, filename)
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                csv_filename = data.get('csv_filename')
                if csv_filename:
                    csv_path = os.path.join(folder_path, csv_filename)
                    if os.path.exists(csv_path):
                        json_csv_mapping[filename] = csv_filename
                        try:
                            # Read CSV content and convert to string
                            csv_content = pd.read_csv(csv_path).to_string(index=False)
                            csv_files_content[csv_content].append((filename, csv_filename))
                        except Exception as e:
                            print(f"Error reading CSV file {csv_path}: {e}")
    
    # Find duplicates
    duplicates = []
    for content, files in csv_files_content.items():
        if len(files) > 1:
            duplicates.extend(files[1:])  # Keep one, mark others as duplicates
    
    return duplicates

def delete_duplicates(folder_path, duplicates):
    for json_file, csv_file in duplicates:
        json_path = os.path.join(folder_path, json_file)
        csv_path = os.path.join(folder_path, csv_file)
        
        try:
            os.remove(json_path)
            print(f"Deleted JSON file: {json_path}")
        except Exception as e:
            print(f"Error deleting JSON file {json_path}: {e}")
        
        try:
            os.remove(csv_path)
            print(f"Deleted CSV file: {csv_path}")
        except Exception as e:
            print(f"Error deleting CSV file {csv_path}: {e}")

def main(folder_path):
    duplicates = find_duplicates(folder_path)
    if duplicates:
        print("The following files are duplicates and will be deleted:")
        for json_file, csv_file in duplicates:
            print(f"JSON: {json_file}, CSV: {csv_file}")
        
        confirm = input("Are you sure you want to delete these files? Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            delete_duplicates(folder_path, duplicates)
        else:
            print("Deletion canceled.")
    else:
        print("No duplicates found.")

if __name__ == "__main__":
    folder_path = "evaluation/" 
    main(folder_path)
