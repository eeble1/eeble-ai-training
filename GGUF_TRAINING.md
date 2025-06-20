# Low VRAM Llama3 Fine-tuning Guide

This document outlines a minimal workflow for fine-tuning Llama 3 using LoRA with 4-bit quantization and exporting the result as a `.gguf` file.

## 1. Prepare the Dataset

Create a JSON Lines (`.jsonl`) file. Each line should contain a single JSON object with two fields:

```json
{"text": "<prompt and answer in Llama3 chat format>"}
```

The `text` field should already be formatted using the [Llama3 chat template](code/chat_template/llama-3-instruct.jinja). For example:

```json
{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe capital of France is Paris.<|eot_id|>"}
```

## 2. Install Requirements

```bash
pip install -r requirements.txt
pip install llama_cpp  # required for GGUF conversion
```

## 3. Run Training

Use the provided script `train_lora_gguf.py`:

```bash
python code/train_lora_gguf.py --data /path/to/data.jsonl --epochs 3 --output /project/models/my-llama3
```

Training uses 4‑bit QLoRA to reduce VRAM requirements. Adjust `--epochs` as desired.

## 4. Resulting Files

After training completes you will find the Hugging Face model weights under the chosen output directory and a compressed `model.gguf` file ready for use with `llama.cpp`-compatible runtimes.
