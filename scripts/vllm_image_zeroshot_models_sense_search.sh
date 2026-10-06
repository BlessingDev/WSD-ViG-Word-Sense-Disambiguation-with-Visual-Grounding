
# 하나의 파일에 대한 inference를 여러 모델로 수행할 때 사용하는 스크립트
#models=("google/gemma-3-27b-it" "Qwen/Qwen3-VL-30B-A3B-Instruct" "mistralai/Mistral-Small-3.1-24B-Instruct-2503" "Qwen/Qwen2.5-VL-7B-Instruct")
#models=("google/gemma-3-4b-it" "Qwen/Qwen3-VL-4B-Instruct")
models=("/workspace/model_dir/gemma-3-4b-it/iwsd-rag-gemma3/final_model" "/workspace/model_dir/Qwen3-VL-4B-Instruct/iwsd-rag-gemma3/final_model")

output_dir="/workspace/data/test_set_process/inference/sense_search_ambiguous_sentence_tune"

for model_checkpoint in "${models[@]}"
do
    model_name=$(echo "$model_checkpoint" | cut -d'/' -f2)
    echo "Running inference with model: $model_name"
    # gemma3 sense_search inference
    for k in 1 2 3
    do
        python /workspace/vllm_inference.py \
            --model_checkpoint ${model_checkpoint} \
            --inference_set_path /workspace/data/test_set_process/wsd_set_entire_labeled_ambiguous_sentence_sense_search3_gemma3_k${k}_prompt.csv \
            --output_file_path ${output_dir}/wsd_set_entire_ambiguous_sentence_sense_search3_gemma3_k${k}_${model_name}.csv \
            --image_dir /workspace/data/semeval-2023-V-WSD-test/test_images/ \
            --image_number 1 \
            --seed 42
    done
    # mistral3 sense_search inference
    for k in 1 2 3
    do
        python /workspace/vllm_inference.py \
            --model_checkpoint ${model_checkpoint} \
            --inference_set_path /workspace/data/test_set_process/wsd_set_entire_labeled_ambiguous_sentence_sense_search3_mistral3_k${k}_prompt.csv \
            --output_file_path ${output_dir}/wsd_set_entire_ambiguous_sentence_sense_search3_mistral3_k${k}_${model_name}.csv \
            --image_dir /workspace/data/semeval-2023-V-WSD-test/test_images/ \
            --image_number 1 \
            --seed 42
    done
    # qwen3 sense_search inference
    for k in 1 2 3
    do
        python /workspace/vllm_inference.py \
            --model_checkpoint ${model_checkpoint} \
            --inference_set_path /workspace/data/test_set_process/wsd_set_entire_labeled_ambiguous_sentence_sense_search3_qwen3_k${k}_prompt.csv \
            --output_file_path ${output_dir}/wsd_set_entire_ambiguous_sentence_sense_search3_qwen3_k${k}_${model_name}.csv \
            --image_dir /workspace/data/semeval-2023-V-WSD-test/test_images/ \
            --image_number 1 \
            --seed 42
    done
done