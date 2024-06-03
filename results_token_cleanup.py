import glob

def process_content(content):
    # Remove <start> and <end> substrings
    cleaned_content = content.replace("<start> ", "").replace(" <end>", "")
    # Capitalize the first letter of the remaining sentence
    cleaned_content = cleaned_content.strip()  # Remove leading and trailing whitespaces
    cleaned_content = cleaned_content[0].upper() + cleaned_content[1:] + "."
    return cleaned_content

# Path to the directory where you want to search for .txt files
directory_path = 'occlusion_study_sat_results'

# Use glob to find all paths ending with .txt in the specified directory and all its subdirectories
txt_files = glob.glob(f'{directory_path}/**/*.txt', recursive=True)
print(txt_files)

# Process and overwrite the found files
for file_path in txt_files:
    with open(file_path, 'r') as file:
        content = file.read()
        processed_content = process_content(content)

    # Overwrite the file with the cleaned-up content
    with open(file_path, 'w') as file:
        file.write(processed_content)
