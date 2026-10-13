# modify the python file path to your own path
# V-WSD dataset should be in default form. the script automatically extracts the data and gold image list from the V-WSD dataset. The output will be saved in the specified output path.

python /workspace/main.py \
    --task extract_data_from_vwsd \
    --input_path /workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt \
    --output_path /workspace/data/WSD-ViG/set_0.csv