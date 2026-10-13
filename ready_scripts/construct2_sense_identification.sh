
# I will assume that the Reverse Image Search (RIS) is already done and the results are stored in a CSV file.
# The resulting csv file should contain "word_index", "web_urls" columns.
# * word_index: the index of the word in the original V-WSD set
# * web_urls: the json object string of the web urls found by RIS for the corresponding word. The example of the json object string is as follows:
# [{""title"": ""How grass grows. - YouTube"", ""url"": ""https://www.youtube.com/watch?v=MLm4QVhGspo""}]
# This script will read the CSV file containing the RIS results and the original WSD set, and add prompt to perform sense identification on them.

# First retrieve the web content from the web urls and add prompt to summarize them.
python /workspace/web_search_summarize_prompt.py \
    --retrieval_result_path /workspace/data/train_set_process/wsd_set_entire_google_vision_result.csv \
    --wsd_set_path /workspace/data/WSD-ViG/set_1_pos_classification.csv \
    --output_path /workspace/data/WSD-ViG/set_1_summarize_prompt.csv \
    --prompt_type phrase

# Then perform inference on the summarized web content to identify the sense of the ambiguous word.
python /workspace/vllm_inference.py \
    --model_checkpoint google/gemma-3-27b-it \
    --inference_set_path /workspace/data/WSD-ViG/set_1_summarize_prompt.csv \
    --output_file_path /workspace/data/WSD-ViG/set_1_summarize_inference.csv \
    --image_dir /workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/ \
    --image_number 1 \
    --seed 42

# Then, add prompt to the original WSD set to perform sense identification on them.
# k is set to 2 as default, following the setting in the paper. You can change it to other values if you want.
python /workspace/add_prompt_for_generation.py \
    --task sense_identification \
    --dataset_path /workspace/data/WSD-ViG/set_1_pos_classification.csv \
    --output_path /workspace/data/WSD-ViG/set_1_sense_identification_prompt.csv \
    --summarized_set_path /workspace/data/WSD-ViG/set_1_summarize_inference.csv \
    --top_k 2

# Finally, perform inference on the generated sense identification prompts to identify the sense of the ambiguous word.
python /workspace/vllm_inference.py \
    --model_checkpoint google/gemma-3-27b-it \
    --inference_set_path /workspace/data/WSD-ViG/set_1_sense_identification_prompt.csv \
    --output_file_path /workspace/data/WSD-ViG/set_1_sense_identification_inference.csv \
    --image_dir /workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/ \
    --image_number 1 \
    --seed 42

# Then, extract the predicted sense from the inference results and save them to a CSV file.
python /workspace/main.py \
    --task extract_predicted_senses \
    --input_path /workspace/data/WSD-ViG/set_1_sense_identification_inference.csv \
    --output_path /workspace/data/WSD-ViG/set_2_sense_identification.csv
