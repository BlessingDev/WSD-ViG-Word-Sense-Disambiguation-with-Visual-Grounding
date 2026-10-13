# google/gemma-3-4b-it
# lr 5e-6 decay 0.01
# Qwen/Qwen3-VL-4B-Instruct
# lr 6e-6 decay 0.03


python /workspace/generation_decoder_train.py \
    --model_checkpoint Qwen/Qwen3-VL-4B-Instruct \
    --attn_implementation eager \
    --train_file /workspace/data/train_set_process/wsd_set_entire_ambiguous_sentence_sense_search3_2_gemma3_k3_train.csv \
    --validation_file /workspace/data/train_set_process/wsd_set_entire_ambiguous_sentence_sense_search3_2_gemma3_k3_val.csv \
    --image_dir /workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1 \
    --output_dir /workspace/model_dir/Qwen3-VL-4B-Instruct/iwsd-rag-gemma3_v3_2_t1/ \
    --train_epochs 10 \
    --weight_decay 0.03 \
    --label_smoothing_factor 0.0 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 6e-6 \
    --warmup_steps 100 \
    --logging_steps 20 \
    --save_steps 200 \
    --prompt_template image_rag_sense

#bash /workspace/scripts/vllm_image_zeroshot_models_sense_search.sh