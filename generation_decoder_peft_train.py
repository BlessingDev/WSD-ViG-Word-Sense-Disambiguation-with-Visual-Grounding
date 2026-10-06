import argparse
import json
import os
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

MODEL_CHECKPOINT = "google/gemma-3-27b-it"
TRAIN_DATASET_PATH = "/workspace/data/dataset_construction_train/train_set_pos.csv"
VAL_DATASET_PATH = "/workspace/data/dataset_construction_train/val_set_pos.csv"

PROMPT_TEMPLATE = """Text Context: {context}
Ambiguous Word: {word}
---
Sense List:
{sense_list}
---
You are an expert linguistic annotator. An ambiguous word found within the context of a given text, and a list of potential senses for that word are provided. Your task is to determine the correct sense by selecting the one that most directly aligns with the context and its background event.
---
Decision Logic & Rules
Follow these rules in order of priority to make your decision:

First, take a look at the 'Text Context' and the image. Leverage both context to analyze any significant background context is provided in the image. If such information exists, prioritize the sense that best fits this background information, even if it is not the most direct match for the visual content.

Directness of Meaning: If no additional background information can be obtained from given context, choose the sense that provides the most direct and specific fit for the visual and linguistic context. Even if a perfect match does not exist, select the sense that is the closest indirect match.
---
After all your reasoning is finished, provide your final decision as the format 'A: [sense number]' and end your generation. In here, [sense number] is the index of your chosen sense from the provided list.
"""


def main(args):
    train_dataset = Dataset.from_csv(args.train_file)
    val_dataset = Dataset.from_csv(args.validation_file)

    processor = AutoProcessor.from_pretrained(args.model_checkpoint)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Gemma-3는 멀티모달 아키텍처이므로 AutoModelForImageTextToText를 권장합니다.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )

    # ----------------------------------------------------
    # LoRA 설정: 최근 표준 레시피 (Attention + MLP All-Linear)
    # ----------------------------------------------------
    peft_config = LoraConfig(
        r=args.lora_rank,  # LoRA 차원 수
        lora_alpha=args.lora_rank * 2,  # 일반적으로 alpha = 2 * r 로 스케일링 설정
        lora_dropout=args.dropout_rate,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # 최근 벤치마크(Llama/Gemma 계열)에서 성능 수렴도가 가장 높은 All Linear 프로젝션 레이어 타깃
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    def preprocess_function(sample):
        assistant_text = "A: " + str(int(sample["gold_sense"]))
        sense_list = json.loads(sample["senses"]).get(sample["gold_pos"], [])
        sense_str = "\n".join(
            [f" {idx + 1}. {sense}" for idx, sense in enumerate(sense_list)]
        )
        prompt = PROMPT_TEMPLATE.format(
            context=sample["word_phrase"],
            word=sample["word"],
            sense_list=sense_str,
        )

        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": os.path.join(
                                args.image_dir, sample["gold_image"]
                            ),
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "completion": [{"role": "assistant", "content": assistant_text}],
        }

    tokenized_train_dataset = train_dataset.map(preprocess_function)
    tokenized_val_dataset = val_dataset.map(preprocess_function)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        do_train=True,
        do_eval=True,
        packing=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": 2},
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=3,
        num_train_epochs=args.train_epochs,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        metric_for_best_model="eval_loss",
        bf16=True,
        push_to_hub=False,
        load_best_model_at_end=True,
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,  # SFTTrainer에 peft_config 전달 시 LoRA 자동 래핑 및 가중치 고정 처리
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        processing_class=processor.tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # LoRA 어댑터 가중치만 저장됩니다.
    trainer.save_model(os.path.join(args.output_dir, "final_lora_adapter"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default=TRAIN_DATASET_PATH)
    parser.add_argument("--validation_file", type=str, default=VAL_DATASET_PATH)
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1",
    )
    parser.add_argument("--model_checkpoint", type=str, default=MODEL_CHECKPOINT)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # LoRA는 완전 파인튜닝(1e-5)보다 다소 높은 학습률(1e-4 ~ 2e-4)이 안정적입니다.
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation to use (eager, flash_attention_2, etc.)",
    )

    args = parser.parse_args()
    print(args)
    main(args)