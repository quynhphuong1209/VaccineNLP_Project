# -*- coding: utf-8 -*-
"""
💡 02. Gemma-4 4B QLoRA Training - VaccineNLP
Fine-tuning Gemma-4 for Explainable AI (XAI) in Public Health.
Supports local execution and environment-agnostic paths.
"""

import os
import sys

# Thêm thư mục gốc của dự án vào sys.path để nhận diện module 'src'
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
from huggingface_hub import login
from datasets import load_dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

# Centralized Path Management
from src.common import paths

# ==============================
# ⚙️ CONFIGURATION
# ==============================
MODEL_NAME = "unsloth/gemma-4-E4B-it"
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True
OUTPUT_DIR = paths.MODEL_DIR / "gemma_qlora_xai"
TRAIN_PATH = paths.MODEL_READY_DIR / "train_v2_seg.jsonl"
TEST_PATH = paths.GOLD_DATA_DIR / "benchmark_test_set.jsonl"

# Mapping for labels
MISINFO_MAP = {0: "Không liên quan", 1: "Tin giả", 2: "Chính xác"}
STANCE_MAP = {0: "Ủng hộ", 1: "Phản đối", 2: "Trung lập", 3: "Không rõ"}
SENTIMENT_MAP = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}

def setup_huggingface():
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        print("✅ Logged in to HuggingFace using HF_TOKEN.")
    else:
        print("⚠️ HF_TOKEN environment variable not found. Ensure you are logged in or public models are accessible.")

def format_prompt(row, tokenizer):
    text = row.get('text_cleaned') or row.get('text') or ""
    reasoning = row.get('llm_reasoning', 'Phân tích dựa trên ngữ cảnh.')
    ids = row.get('standardized_ids') or [0, 3, 1]

    convo = [
        {"role": "user", "content": (
            f"You are an Explainable AI in Public Health. Analyze the text, "
            f"provide your reasoning first, then the structured labels.\n\nVăn bản: {text}"
        )},
        {"role": "model", "content": (
            f"Lý luận: {reasoning}\n"
            f"Kết quả: {MISINFO_MAP.get(ids[0], 'Không liên quan')} | "
            f"{STANCE_MAP.get(ids[1], 'Không rõ')} | "
            f"{SENTIMENT_MAP.get(ids[2], 'Trung tính')}"
        )}
    ]
    return {"text": tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix('<bos>')}

def main():
    paths.ensure_dirs()
    setup_huggingface()

    # 1. Load Model & Tokenizer
    print(f"📡 Loading model: {MODEL_NAME}")
    model, tokenizer = FastModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = LOAD_IN_4BIT,
    )

    # 2. Setup PEFT (LoRA)
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers = False,
        finetune_language_layers = True,
        finetune_attention_modules = True,
        finetune_mlp_modules = True,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    model.print_trainable_parameters()

    # 3. Prepare Data
    print(f"📂 Loading datasets...")
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    
    raw_train_ds = load_dataset("json", data_files=str(TRAIN_PATH), split="train")
    split_ds = raw_train_ds.train_test_split(test_size=0.1, seed=42)
    
    train_ds = split_ds["train"].map(lambda x: format_prompt(x, tokenizer))
    eval_ds = split_ds["test"].map(lambda x: format_prompt(x, tokenizer))
    test_ds = load_dataset("json", data_files=str(TEST_PATH), split="train").map(lambda x: format_prompt(x, tokenizer))

    print(f"✅ Data ready: Train={len(train_ds)}, Val={len(eval_ds)}, Test={len(test_ds)}")

    # 4. Trainer Setup
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_ds,
        eval_dataset = eval_ds,
        args = SFTConfig(
            output_dir = str(OUTPUT_DIR),
            dataset_text_field = "text",
            max_length = MAX_SEQ_LENGTH,
            per_device_train_batch_size = 2,
            per_device_eval_batch_size = 1,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            num_train_epochs = 2,
            learning_rate = 2e-4,
            logging_steps = 10,
            save_strategy = "epoch",
            eval_strategy = "steps",
            eval_steps = 50,
            optim = "adamw_8bit",
            bf16 = torch.cuda.is_bf16_supported(),
            fp16 = not torch.cuda.is_bf16_supported(),
            weight_decay = 0.01,
            report_to = "none",
            seed = 3407,
        ),
    )

    # 5. Response-only Training (Instruction Tuning Optimization)
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|turn>user\n",
        response_part = "<|turn>model\n",
    )

    # 6. Run Training
    print("🚀 Starting Gemma-4 fine-tuning...")
    trainer.train()

    # 7. Save Final Model
    final_path = OUTPUT_DIR / "final_model"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"💾 Model saved successfully at: {final_path}")

if __name__ == "__main__":
    main()