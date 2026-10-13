
# This script is for basic zero-shot CoT evaluation on the WSD-ViG dataset.
# This script will run the inference under c+asen+R setting.
# Note that we will assume that the Reverse Image Search (RIS) is already done and the results are stored in a CSV file.
# Refer to the construct2_sense_identification.sh script for more details on how to perform the RIS and sense identification steps.

# First, retrieve the web content from the web urls and add prompt to summarize them.
# We will assume that the 
python /workspace/web_search_summarize_prompt.py \
    --retrieval_result_path /workspace/data/WSD-ViG/test_google_vision_result.csv \
    --wsd_set_path /workspace/data/WSD-ViG/test.csv \
    --output_path /workspace/data/WSD-ViG/test_summarize_prompt.csv \
    --prompt_type ambig_sentence

# Then perform inference on the summarized web content to identify the sense of the ambiguous word.
# The summarization can be performed using [mistralai/Mistral-Small-3.1-24B-Instruct-2503, google/gemma-3-27b-it, Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen3-VL-30B-A3B-Instruct, LGAI-EXAONE/EXAONE-4.5-33B] models.
# In this script, we will use google/gemma-3-27b-it model for inference. You can change the model to other models that our script supports.
python /workspace/vllm_inference.py \
    --model_checkpoint google/gemma-3-27b-it \
    --inference_set_path /workspace/data/WSD-ViG/test_summarize_prompt.csv \
    --output_file_path /workspace/data/WSD-ViG/test_summarize_gemma3_inference.csv \
    --image_dir /workspace/data/semeval-2023-V-WSD-test/test_images/ \
    --image_number 1 \
    --seed 42

# Then, add prompt to the original WSD set to perform WSD on test set.
# The top_k is set to 2 here. You can change it to find the sweet spot.
python /workspace/add_prompt_for_generation.py \
    --task sense_identification \
    --dataset_path /workspace/data/WSD-ViG/test.csv \
    --output_path /workspace/data/WSD-ViG/test_wsd_gemma3_k2_prompt.csv \
    --summarized_set_path /workspace/data/WSD-ViG/test_summarize_gemma3_inference.csv \
    --top_k 2

python /workspace/vllm_inference.py \
    --model_checkpoint google/gemma-3-27b-it \
    --inference_set_path /workspace/data/WSD-ViG/test_wsd_gemma3_k2_prompt.csv \
    --output_file_path /workspace/data/WSD-ViG/test_wsd_gemma3_k2_inference.csv \
    --image_dir /workspace/data/semeval-2023-V-WSD-test/test_images/ \
    --image_number 1 \
    --seed 42

# Now, evaluate the performance of the model on the test set.
python /workspace/main.py \
    --task evaluate_wsd_accuracy \
    --input_path /workspace/data/WSD-ViG/test_wsd_gemma3_k2_inference.csv
