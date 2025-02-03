import re
import os

# Define paths
INPUT_FILE = "wind_in_the_willows.txt"  # Replace with your dataset file
OUTPUT_FILE = "processed_wind_in_the_willows.txt"

def clean_text(text):
    """Cleans and normalizes text for better training performance."""
    # Remove extra spaces, newlines, and special characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)  # Ensure space after punctuation
    text = re.sub(r'\n{2,}', '\n\n', text)  # Keep paragraph breaks but remove extra newlines
    text = re.sub(r'\s+', ' ', text).strip()  # Remove excessive spaces

    return text

def preprocess_file(input_path, output_path):
    """Reads, cleans, and preprocesses the dataset."""
    with open(input_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

    cleaned_text = clean_text(raw_text)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(cleaned_text)

    print(f"✅ Processed text saved to {output_path}")

# Run the preprocessing
if __name__ == "__main__":
    preprocess_file(INPUT_FILE, OUTPUT_FILE)