sample_num=3

cur_idx=1
# text-only unimodal t-test
for i in $(seq 1 $sample_num)
do
    echo "Running text-only unimodal t-test inference for sample ${cur_idx}..."
    python /workspace/vllm_inference.py \
        --model_checkpoint google/gemma-3-27b-it \
        --temperature 0.2 \
        --inference_set_path /workspace/data/test_set_process/wsd_set_entire_labeled_sense_ambig_sentence_text_prompt.csv \
        --output_file_path /workspace/data/test_set_process/inference/uni_modal_shortcut/wsd_set_entire_labeled_ambiguous_sentence_sense_text_${cur_idx}_gemma-3-27b-it.csv
    cur_idx=$((cur_idx + 1))
    sleep 1s
done

cur_idx=1
# image-only unimodal t-test
for i in $(seq 1 $sample_num)
do
    echo "Running image-only unimodal t-test inference for sample ${cur_idx}..."
    python /workspace/vllm_inference.py \
        --model_checkpoint google/gemma-3-27b-it \
        --image_dir /workspace/data/semeval-2023-V-WSD-test/test_images/ \
        --temperature 0.2 \
        --inference_set_path /workspace/data/test_set_process/wsd_set_entire_labeled_image_sense_prompt.csv \
        --output_file_path /workspace/data/test_set_process/inference/uni_modal_shortcut/wsd_set_entire_labeled_sense_image_${cur_idx}_gemma-3-27b-it.csv
    cur_idx=$((cur_idx + 1))
    sleep 1s
done

cur_idx=1
# image-text t-test
for i in $(seq 1 $sample_num)
do
    echo "Running image-text unimodal t-test inference for sample ${cur_idx}..."
    python /workspace/vllm_inference.py \
        --model_checkpoint google/gemma-3-27b-it \
        --image_dir /workspace/data/semeval-2023-V-WSD-test/test_images/ \
        --temperature 0.2 \
        --inference_set_path /workspace/data/test_set_process/wsd_set_entire_labeled_ambiguous_sentence_sense_prompt.csv \
        --output_file_path /workspace/data/test_set_process/inference/uni_modal_shortcut/wsd_set_entire_labeled_ambiguous_sentence_sense_${cur_idx}_gemma-3-27b-it.csv
    cur_idx=$((cur_idx + 1))
    sleep 1s
done