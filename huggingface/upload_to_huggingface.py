# pip install huggingface_hub transformers
# huggingface-cli login

import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from huggingface_hub import login

def upload_model(username, model_name, local_path, hf_token=None):
    """Uploads a trained model to Hugging Face Model Hub"""

    # Log in to Hugging Face (if a token is provided)
    if hf_token:
        login(token=hf_token)

    # Define repo name
    repo_name = f"{username}/{model_name}"

    print(f"🚀 Uploading model to {repo_name} on Hugging Face...")

    # Load trained model & tokenizer
    model = GPT2LMHeadModel.from_pretrained(local_path)
    tokenizer = GPT2Tokenizer.from_pretrained(local_path)

    # Push to Hugging Face Hub
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)

    print(f"✅ Model uploaded successfully to: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a trained model to Hugging Face")

    parser.add_argument("--username", type=str, required=True, help="Your Hugging Face username")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model repository on Hugging Face")
    parser.add_argument("--local_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token (optional)")

    args = parser.parse_args()

    upload_model(args.username, args.model_name, args.local_path, args.hf_token)