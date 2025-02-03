# Install: https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe
# Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip install datasets transformers
# Restart computer & run script.

import os
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback

# Define the directory where the trained model will be saved/loaded
MODEL_DIR = "./wind_in_the_willows_model"

def model_exists():
    """Checks if a trained model already exists."""
    return os.path.exists(MODEL_DIR) and os.path.isdir(MODEL_DIR)

if __name__ == "__main__":
    print(f"Info: PyTorch is using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Check if a trained model exists
    if model_exists():
        print("Info: ✅ Found existing trained model, loading...")
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    else:
        print("Info: No trained model found, training from scratch...")

        # Load dataset
        dataset_path = "wind_in_the_willows.txt"
        dataset = load_dataset("text", data_files={"train": dataset_path})["train"]

        # Split dataset into training and evaluation sets (80% train, 20% test)
        split_dataset = dataset.train_test_split(test_size=0.2)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        # Function to tokenize dataset
        def tokenize_function(examples):
            """Tokenizes input text data."""
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=256
            )

        # Apply tokenization to datasets
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)

        # Load pre-trained GPT-2 model
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Free up GPU memory
        torch.cuda.empty_cache()

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",  # Directory for saving model checkpoints
            eval_strategy="epoch",   # Evaluate model at the end of each epoch
            save_strategy="epoch",   # Save model at each epoch
            num_train_epochs=3,       # Number of training epochs
            per_device_train_batch_size=1,  # Batch size per GPU
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,  # Simulate a larger batch size
            save_steps=500,           # Save model checkpoint every 500 steps
            save_total_limit=2,       # Keep only the latest 2 checkpoints
            logging_dir="./logs",    # Directory for logging
            fp16=True,                # Use mixed precision for efficiency
            gradient_checkpointing=True,  # Save memory at cost of extra computation
            dataloader_num_workers=0, # Avoid multiprocessing overhead
            optim="adamw_torch_fused", # Optimizer choice
            report_to="none",        # Disable automatic logging to external platforms
            load_best_model_at_end=True, # Load best model based on evaluation loss
            metric_for_best_model="eval_loss",
            greater_is_better=False,  # Lower evaluation loss is better
            logging_steps=50          # Log every 50 steps
        )

        # Define data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # No masked language modeling
        )

        # Set up Trainer class for training
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop training if no improvement for 3 epochs
        )

        # Train the model
        print("Info: Training about to start...")
        trainer.train()

        # Save trained model for future use
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        print("Info: ✅ Model training complete and saved.")

    # Function to generate text using the trained model
    def generate_text(prompt=None, max_length=100):
        """Generates text based on a given prompt using the trained model."""
        if prompt is None or prompt.strip() == "":
            input_text = "<|endoftext|>"
        else:
            input_text = prompt

        # Tokenize input and move to the same device as the model
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.9,  # Controls randomness
                top_k=50,         # Limits sampling to top 50 words
                top_p=0.95,       # Nucleus sampling
                use_cache=True,
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)

    # Generate and print example text
    random_text = generate_text()
    print("Randomly Generated Text:\n", random_text)