import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_text(model_name, prompt=None, max_length=100):
    """Loads a model from Hugging Face and generates text. If no prompt is provided, generates random text."""

    print(f"🚀 Loading model: {model_name} from Hugging Face...")

    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Handle case where no prompt is provided
    if not prompt or prompt.strip() == "":
        print("⚡ No prompt provided. Generating random text...")
        input_text = "<|endoftext|>"  # Special token to generate text from scratch
    else:
        input_text = prompt

    # Encode input prompt
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate text
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.9, top_k=50, top_p=0.95)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a model from Hugging Face.")

    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model repository name (e.g., 'your-username/wind-in-the-willows-gpt2')")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt to generate text from (leave empty for random text)")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")

    args = parser.parse_args()

    output_text = generate_text(args.model_name, args.prompt, args.max_length)
    print("\n📝 Generated Text:\n", output_text)
    