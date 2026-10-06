
python /workspace/generation_decoder_peft_train.py \
    --model_checkpoint google/gemma-3-27b-it \
    --attn_implementation eager \
    --train_file /workspace/data/train_set_process/wsd_set_entire_ambiguous_sentence_train.csv \
    --validation_file /workspace/data/train_set_process/wsd_set_entire_ambiguous_sentence_val.csv \
    --image_dir /workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1 \
    --output_dir /workspace/model_dir/gemma-3-27b-it/peft_iwsd2/ \
    --train_epochs 10 \
    --weight_decay 0.01 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --eval_steps 200 \
    --lora_rank 16 \
    --learning_rate 1e-4 \
    --warmup_steps 100 \
    --logging_steps 20