
python /workspace/add_prompt_for_generation.py \
    --task ambiguous_sentence_generation \
    --dataset_path /workspace/data/train_set_process/set_2_sense_identification.csv \
    --output_path /workspace/data/train_set_process/set_2_ambiguous_sentence_generation_prompt.csv

# Following the settings in the paper, we use gpt-5.4-mini model for ambiguous sentence generation. You can change the model to your own model checkpoint.
# .env file should contain the OPENAI_API_KEY for the model inference.
python /workspace/openai_inference.py \
    --gpt_model gpt-5.4-mini \
    --reasoning_effort medium \
    --wsd_set_path /workspace/data/train_set_process/set_2_ambiguous_sentence_generation_prompt.csv \
    --output_file_path /workspace/data/train_set_process/set_2_ambiguous_sentence_generation_inference.csv \

# Then, extract the generated ambiguous sentences from the inference results and save them to a CSV file.
python /workspace/main.py \
    --task extract_ambiguous_sentences \
    --inference_file_path /workspace/data/train_set_process/set_2_ambiguous_sentence_generation_inference.csv \
    --output_file_path /workspace/data/train_set_process/wsd_vig_train_set.csv

# Finally, The entire dataset is constructed automatically and saved to 'wsd_vig_train_set.csv'.