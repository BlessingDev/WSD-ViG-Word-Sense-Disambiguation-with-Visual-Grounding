# modify the python file path to your own path

python /workspace/add_prompt_for_generation.py \
    --task pos_classification \
    --dataset_path /workspace/data/WSD-ViG/set_0.csv \
    --output_path /workspace/data/WSD-ViG/set_0_pos_prompt.csv

# modify the image directory path to your own path. The image directory should contain the images corresponding to the V-WSD dataset.
python /workspace/vllm_inference.py \
    --model_checkpoint google/gemma-3-27b-it \
    --inference_set_path /workspace/data/WSD-ViG/set_0_pos_prompt.csv \
    --output_file_path /workspace/data/WSD-ViG/set_0_pos_inference.csv \
    --image_dir /workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/ \
    --image_number 1 \
    --seed 42

python /workspace/main.py \
    --task extract_predicted_pos \
    --input_path /workspace/data/WSD-ViG/set_0_pos_inference.csv \
    --output_path /workspace/data/WSD-ViG/set_1_pos_classification.csv
