import os
import argparse
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# Fix Windows symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Distill a fine-tuned GPT-2 model into a smaller version.")
parser.add_argument("--model_dir", type=str, required=True, help="Path to the fine-tuned GPT-2 model.")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training dataset (text file).")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the distilled model.")
args = parser.parse_args()

# Validate inputs
if not os.path.exists(args.model_dir):
    raise FileNotFoundError(f"❌ Error: Model directory '{args.model_dir}' does not exist!")

if not os.path.exists(args.dataset_path):
    raise FileNotFoundError(f"❌ Error: Dataset file '{args.dataset_path}' not found!")

# Load teacher (fine-tuned) model
print(f"🔍 Loading teacher model from: {args.model_dir}")
teacher_model = GPT2LMHeadModel.from_pretrained(args.model_dir)
tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)

# Load student model (smaller GPT-2)
print("🔍 Loading student model (distilgpt2)...")
student_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
student_model.resize_token_embeddings(len(tokenizer))

# Detect and move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device).eval()
student_model.to(device).train()

# Load dataset
print(f"📚 Loading dataset from: {args.dataset_path}")
dataset = load_dataset("text", data_files={"train": args.dataset_path})["train"]

# Tokenization function with debugging
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

    # Debugging: Ensure inputs are not empty
    if len(tokens["input_ids"]) == 0:
        print("❌ Warning: Empty tokenized input found!")

    return tokens

# Apply tokenization
print("⚙️ Tokenizing dataset...")
dataset = dataset.map(tokenize_function, batched=True)

# Define distillation loss function
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
Compute the distillation loss based on KL Divergence.
    """
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean"
    ) * (temperature ** 2)

# Training function
def train_student():
    print("🚀 Starting distillation training...")
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)

    for epoch in range(3):  # Train for 3 epochs
        total_loss = 0

        for batch in dataset:
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(device)

            # Skip empty inputs
            if inputs["input_ids"].numel() == 0:
                print("⚠️ Skipping empty input batch...")
                continue

                # Compute teacher outputs (no gradient update needed)
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)

            # Compute student outputs
            student_outputs = student_model(**inputs)

            # Compute distillation loss
            loss = distillation_loss(student_outputs.logits, teacher_outputs.logits)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"✅ Epoch {epoch+1}: Average Loss {total_loss / len(dataset)}")

    # Save the distilled student model
    print(f"💾 Saving distilled student model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    student_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Distilled model saved successfully!")

# Run the training function
train_student()