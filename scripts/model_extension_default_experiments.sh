
model_checkpoint="Qwen/Qwen3-VL-8B-Instruct"
model_directory="/workspace/data/test_set_process/inference/qwen3-8b-it"
model_save_name="qwen3-vl-8b-instruct"

# text-only inference
python /workspace/vllm_inference.py \
    --model_checkpoint ${model_checkpoint} \
    --inference_set_path /workspace/data/test_set_process/wsd_set_entire_labeled_sense_ambig_sentence_text_prompt.csv \
    --output_file_path ${model_directory}/wsd_set_entire_labeled_ambiguous_sentence_sense_text_${model_save_name}.csv

# image-text inference
python /workspace/vllm_inference.py \
    --model_checkpoint ${model_checkpoint} \
    --image_dir /workspace/data/semeval-2023-V-WSD-test/test_images/ \
    --inference_set_path /workspace/data/test_set_process/wsd_set_entire_labeled_ambiguous_sentence_sense_prompt.csv \
    --output_file_path ${model_directory}/wsd_set_entire_labeled_ambiguous_sentence_sense_${model_save_name}.csv

# image-text RAG inference
for i in $(seq 1 3)
do
    echo "Running image-text RAG inference for k ${i}..."
    # gemma3 summarization
    python /workspace/vllm_inference.py \
        --model_checkpoint ${model_checkpoint} \
        --image_dir /workspace/data/semeval-2023-V-WSD-test/test_images/ \
        --inference_set_path /workspace/data/test_set_process/wsd_set_entire_labeled_ambiguous_sentence_sense_search3_gemma3_k${i}_prompt.csv \
        --output_file_path ${model_directory}/wsd_set_entire_labeled_ambiguous_sentence_sense_search3_gemma3_k${i}_${model_save_name}.csv
    # qwen3 summarization
    python /workspace/vllm_inference.py \
        --model_checkpoint ${model_checkpoint} \
        --image_dir /workspace/data/semeval-2023-V-WSD-test/test_images/ \
        --inference_set_path /workspace/data/test_set_process/wsd_set_entire_labeled_ambiguous_sentence_sense_search3_qwen3_k${i}_prompt.csv \
        --output_file_path ${model_directory}/wsd_set_entire_labeled_ambiguous_sentence_sense_search3_qwen3_k${i}_${model_save_name}.csv
done