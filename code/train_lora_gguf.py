import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Llama 3 with LoRA and export to GGUF")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Base HF model id")
    parser.add_argument("--data", required=True, help="Path to training data in JSONL format")
    parser.add_argument("--output", default="./lora-gguf-model", help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=1)
    return parser.parse_args()


def load_data(path):
    return load_dataset("json", data_files=path)["train"]


def main():
    args = parse_args()

    dataset = load_data(args.data)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_steps=100,
        bf16=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=training_args,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    try:
        from llama_cpp import convert
        gguf_path = os.path.join(args.output, "model.gguf")
        convert(args.output, gguf_path)
        print(f"Saved GGUF model to {gguf_path}")
    except Exception as e:
        print(f"Failed to convert to GGUF: {e}. Install llama_cpp to enable conversion.")


if __name__ == "__main__":
    main()
