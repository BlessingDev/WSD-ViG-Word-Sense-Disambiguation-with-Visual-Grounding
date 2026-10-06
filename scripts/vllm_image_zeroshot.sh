model_checkpoint="google/gemma-3-27b-it"
#model_checkpoint="Qwen/Qwen2.5-VL-7B-Instruct"
#model_checkpoint="Qwen/Qwen3-VL-30B-A3B-Instruct"
#model_checkpoint="mistralai/Mistral-Small-3.1-24B-Instruct-2503"
#model_checkpoint="LGAI-EXAONE/EXAONE-4.5-33B"
#model_checkpoint="/workspace/model_dir/gemma-3-4b-it/iwsd-rag/final_model"
#    --example_set_path /workspace/data/dataset_construction_train/invalid1_examples.csv \
#    --image_dir /workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/
#    --image_dir /workspace/data/semeval-2023-V-WSD-test/test_images/
#    --inference_set_path /workspace/data/test_set_process/wsd_set_entire_sense_ambig_sentence_prompt.csv

python /workspace/vllm_inference.py \
    --model_checkpoint ${model_checkpoint} \
    --inference_set_path /workspace/data/train_set_process/wsd_set_entire_ambiguous_sentence_summarize_prompt.csv \
    --output_file_path /workspace/data/train_set_process/inference/summarize/wsd_set_entire_ambiguous_sentence_summarize_wiki_gemma-3-27b-it.csv \
    --image_dir /workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/ \
    --image_number 1 \
    --seed 42