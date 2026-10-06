
# 하나의 파일에 대한 inference를 여러 모델로 수행할 때 사용하는 스크립트
models=("google/gemma-3-27b-it" "Qwen/Qwen3-VL-30B-A3B-Instruct" "mistralai/Mistral-Small-3.1-24B-Instruct-2503" "Qwen/Qwen2.5-VL-7B-Instruct")
#models=("google/gemma-3-4b-it" "Qwen/Qwen3-VL-4B-Instruct")
#models=("/workspace/model_dir/gemma-3-4b-it/iwsd-rag-gemma3/final_model" "/workspace/model_dir/Qwen3-VL-4B-Instruct/iwsd-rag-gemma3/final_model")

for model_checkpoint in "${models[@]}"
do
    model_name=$(echo "$model_checkpoint" | cut -d'/' -f2)
    echo "Running inference with model: $model_name"
    python /workspace/vllm_inference.py \
        --model_checkpoint ${model_checkpoint} \
        --inference_set_path /workspace/data/test_set_process/wsd_set_entire_labeled_sense_ambig_sentence_text_prompt.csv \
        --output_file_path /workspace/data/test_set_process/inference/sense_ambiguous_sentence/wsd_set_entire_labeled_ambiguous_sentence_sense_text_${model_name}.csv \
        --image_dir /workspace/data/semeval-2023-V-WSD-test/test_images/ \
        --image_number 1 \
        --seed 42
done