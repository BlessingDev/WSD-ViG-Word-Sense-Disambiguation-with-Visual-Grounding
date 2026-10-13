def rearrange_trial_to_text_wsd(output_file_path):
    """
    rearrange_trial_to_text_wsd의 Docstring
    
    :param output_file_path: 재배열된 데이터를 저장할 파일 경로
    """
    import data_process
    
    data_path = '/workspace/data/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial.data.v1.txt'
    gold_image_path = '/workspace/data/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial.gold.v1.txt'
    
    polysemy_words_df = data_process.filter_polysemy_words(data_path)
    word_index_list = polysemy_words_df['word_index'].tolist()
    gold_image_list = data_process.get_gold_image_list(gold_image_path, word_index_list)
    
    polysemy_words_df['gold_image'] = gold_image_list
    
    polysemy_words_df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"Rearranged data saved to {output_file_path}")

def sample_and_save_wsd_set(output_file_path, sample_size, check_duplicates=False,
                            data_path='/workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt',
                            gold_image_path='/workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt'):
    """
    sample_and_save_wsd_set의 Docstring
    
    :param output_file_path: 샘플링된 데이터를 저장할 파일 경로
    :param sample_size: 샘플링할 데이터의 크기 (-1인 경우 전체 데이터셋 사용)
    :param check_duplicates: 중복 단어를 허용할지 여부 (True인 경우 중복 단어 제거)
    """
    import data_process
    
    sampled_wsd_df = data_process.sample_data_from_train_set(data_path, sample_size, check_duplicates)
    
    word_index_list = sampled_wsd_df['word_index'].tolist()
    gold_image_list = data_process.get_gold_image_list(gold_image_path, word_index_list)
    
    sampled_wsd_df['gold_image'] = gold_image_list
    
    sampled_wsd_df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"Sampled WSD set saved to {output_file_path}")

def generate_ambiguous_sentence_for_wsd_set(wsd_set_path):
    """
    generate_ambiguous_sentence_for_wsd_set의 Docstring
    
    :param wsd_set_path: WSD 문제 세트가 저장된 txt 파일 경로
    :return: WSD 문제 세트가 포함된 DataFrame
    """
    from vision_model_process import generate_ambiguous_sentences_with_gemma3
    
    wsd_df = generate_ambiguous_sentences_with_gemma3(wsd_set_path)
    
    wsd_df.to_csv('/workspace/data/train_set_process/wsd_set_n400_ambiguous.csv', index=False, encoding='utf-8')

def caption_trail_images(word_df_path):
    """
    caption_trail_images의 Docstring
    """
    import pandas as pd
    from vision_model_process import caption_image_with_gemma3
    
    image_base_path = '/workspace/data/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1'
    
    polysemy_words_df = pd.read_csv(word_df_path)
    
    target_word_list = polysemy_words_df['word'].tolist()
    image_name_list = polysemy_words_df['gold_image'].tolist()
    
    captions = caption_image_with_gemma3(target_word_list, image_name_list, image_base_path)
    
    polysemy_words_df['generated_caption'] = captions
    
    output_captioned_data_path = '/workspace/data/captioned_polysemy_words.csv'
    polysemy_words_df.to_csv(output_captioned_data_path, index=False, encoding='utf-8')
    print(f"Captioned data saved to {output_captioned_data_path}")

def verify_sense_labels(wsd_set_path, output_file_path):
    """
    verify_sense_labels의 Docstring
    
    :param wsd_set_path: WSD 문제 세트가 저장된 txt 파일 경로
    :param output_file_path: 검증된 데이터를 저장할 파일 경로
    :return: WSD 문제 세트가 포함된 DataFrame
    """
    from vision_model_process import verify_sense_labeling_with_gemma3
    
    wsd_df = verify_sense_labeling_with_gemma3(wsd_set_path)
    
    wsd_df.to_csv(output_file_path, index=False, encoding='utf-8')

def make_ambiguous_sentence_generation_prompt(wsd_set_path, output_file_path):
    """
    make_ambiguous_sentence_generation_prompt의 Docstring
    
    :param wsd_set_path: WSD 문제 세트가 저장된 txt 파일 경로
    :param output_file_path: 프롬프트가 포함된 데이터를 저장할 파일 경로
    :return: WSD 문제 세트가 포함된 DataFrame
    """
    from data_process import ambiguous_sentence_generation_prompts_for_wsd_set
    
    wsd_df = ambiguous_sentence_generation_prompts_for_wsd_set(wsd_set_path)
    
    wsd_df.to_csv(output_file_path, index=False, encoding='utf-8')

def answer_wsd_set_and_save(wsd_df_path, model_path, method="image", output_file_path="/workspace/data/"):
    """
    answer_wsd_set_and_save의 Docstring
    
    :param wsd_df_path: WSD 문제 세트가 저장된 txt 파일 경로
    :param model_path: 사용할 모델의 경로
    :param method: 사용할 답변 방법 ("image", "text", "broad_caption", "specific_caption" 중 하나)
    :param output_file_path: 답변된 WSD 세트를 저장할 파일 경로
    """
    import pandas as pd
    from vision_model_process import answer_wsd_set_with_gemma3_image_provided, answer_wsd_set_with_gemma3_text_only, answer_wsd_set_with_gemma3_broad_caption, answer_wsd_set_with_gemma3_specific_caption
    
    wsd_df = pd.read_csv(wsd_df_path)
    
    if method == "image":
        answer_list = answer_wsd_set_with_gemma3_image_provided(wsd_df_path, model_path=model_path)
    elif method == "text":
        answer_list = answer_wsd_set_with_gemma3_text_only(wsd_df_path, model_path=model_path)
    elif method == "broad_caption":
        answer_list, caption_list = answer_wsd_set_with_gemma3_broad_caption(wsd_df_path, model_path=model_path)
        
        wsd_df["generated_caption"] = caption_list
    elif method == "specific_caption":
        answer_list, caption_list = answer_wsd_set_with_gemma3_specific_caption(wsd_df_path, model_path=model_path)
        
        wsd_df["generated_caption"] = caption_list
    
    wsd_df["predicted_answer"] = answer_list
    
    output_answered_data_path = output_file_path + f"{model_path.split('/')[-1]}/answered_ambiguous_sentence_wsd_set_{method}.csv"
    wsd_df.to_csv(output_answered_data_path, index=False, encoding='utf-8')
    print(f"Answered WSD set saved to {output_answered_data_path}")

def google_vision_search_for_wsd_set(wsd_df_path, output_file_path, image_dir="/workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/"):
    """
    google_vision_search_for_wsd_set의 Docstring
    
    :param wsd_df_path: WSD 문제 세트가 저장된 txt 파일 경로
    :param output_file_path: 이미지 검색 결과가 포함된 데이터를 저장할 파일 경로
    :return: WSD 문제 세트가 포함된 DataFrame
    """
    import os
    import json
    import pandas as pd
    from tqdm.auto import tqdm
    from image_search import perform_google_vision_search
    
    wsd_df = pd.read_csv(wsd_df_path)
    
    target_word_list = wsd_df['gold_image'].tolist()
    
    retriev_result_dict = {
        "word_index": wsd_df['word_index'].tolist(),
        "web_urls": [],
        "best_label": [],
        "entities": []
    }
    for idx in tqdm(range(len(target_word_list))):
        target_word_list[idx] = os.path.join(image_dir, target_word_list[idx])
    
        retrieved_result = perform_google_vision_search(target_word_list[idx])
        
        if retrieved_result is None:
            retriev_result_dict["web_urls"].append("None")
            retriev_result_dict["best_label"].append("No label")
            retriev_result_dict["entities"].append("None")
        else:
            retriev_result_dict["web_urls"].append(json.dumps(retrieved_result["web_urls"]))
            retriev_result_dict["best_label"].append(retrieved_result["best_label"])
            retriev_result_dict["entities"].append(json.dumps(retrieved_result["entities"]))

    retrieved_df = pd.DataFrame(retriev_result_dict)
    retrieved_df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"WSD set with retrieved image result saved to {output_file_path}")

def supplement_goolge_vision_search_result(wsd_df_path, exist_output_file_path, image_dir="/workspace/data/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/"):
    """
    supplement_goolge_vision_search_result의 Docstring
    
    :param wsd_df_path: WSD 문제 세트가 저장된 txt 파일 경로
    :param retriev_result_path: 이미지 검색 결과가 저장된 파일 경로
    :param output_file_path: 보완된 데이터를 저장할 파일 경로
    :return: 보완된 WSD 문제 세트가 포함된 DataFrame
    """
    import os
    import json
    import pandas as pd
    from image_search import perform_google_vision_search
    
    wsd_df = pd.read_csv(wsd_df_path)
    retrieved_df = pd.read_csv(exist_output_file_path)
    
    retriev_result_dict = {
        "word_index": wsd_df['word_index'].tolist(),
        "web_urls": [],
        "best_label": [],
        "entities": []
    }
    
    for word_idx in wsd_df['word_index'].tolist():
        if word_idx in retrieved_df['word_index'].tolist():
            retrieved_row = retrieved_df[retrieved_df['word_index'] == word_idx]
            retriev_result_dict["web_urls"].append(retrieved_row["web_urls"].iloc[0])
            retriev_result_dict["best_label"].append(retrieved_row["best_label"].iloc[0])
            retriev_result_dict["entities"].append(retrieved_row["entities"].iloc[0])
        else:
            target_image_name = wsd_df[wsd_df['word_index'] == word_idx]['gold_image'].iloc[0]
            target_image_path = os.path.join(image_dir, target_image_name)
            
            retrieved_result = perform_google_vision_search(target_image_path)
            
            if retrieved_result is None:
                retriev_result_dict["web_urls"].append("None")
                retriev_result_dict["best_label"].append("No label")
                retriev_result_dict["entities"].append("None")
            else:
                retriev_result_dict["web_urls"].append(json.dumps(retrieved_result["web_urls"]))
                retriev_result_dict["best_label"].append(retrieved_result["best_label"])
                retriev_result_dict["entities"].append(json.dumps(retrieved_result["entities"]))
    
    supplemented_df = pd.DataFrame(retriev_result_dict)
    supplemented_df.to_csv(exist_output_file_path, index=False, encoding='utf-8')
    print(f"Supplemented WSD set with retrieved image result saved to {exist_output_file_path}")

def divide_supplemented_set(original_df_path, supplemented_df_path, output_df_path):
    """
    divide_supplemented_set의 Docstring
    
    :param original_df_path: 원본 데이터가 저장된 파일 경로
    :param supplemented_df_path: 보완된 데이터와 원본 데이터가 합쳐져 저장된 파일 경로
    :param output_df_path: 나누어진 데이터를 저장할 파일 경로
    """
    import os
    import pandas as pd
    
    original_df = pd.read_csv(original_df_path)
    supplemented_df = pd.read_csv(supplemented_df_path)
    
    org_word_index = original_df["word_index"].tolist()
    
    drop_indices = list()
    for idx, row in supplemented_df.iterrows():
        if row["word_index"] in org_word_index:
            drop_indices.append(idx)

    divided_df = supplemented_df.drop(drop_indices)
    divided_df.to_csv(output_df_path, index=False, encoding='utf-8')

def change_summarize_prompt_with_template(wsd_set_path, output_file_path, prompt_template, search_result_path="/workspace/data/test_set_process/wsd_set_entire_google_vision_result.csv", format_template=["word", "context", "web_content", "entities"]):
    """
    change_summarize_prompt_with_prompt의 Docstring
    
    :param wsd_set_path: WSD 문제 세트가 저장된 txt 파일 경로
    :param output_file_path: 프롬프트가 포함된 데이터를 저장할 파일 경로
    :return: WSD 문제 세트가 포함된 DataFrame
    """
    from data_process import change_summarize_prompt
    
    df = change_summarize_prompt(wsd_set_path, prompt_template, search_result_path, format_template=format_template)
    
    df.to_csv(output_file_path, index=False, encoding='utf-8')

def extract_ambiguous_sentence_for_wsd_set(wsd_set_path, output_file_path):
    """
    extract_ambiguous_sentence_for_wsd_set의 Docstring
    
    :param wsd_set_path: WSD 문제 세트가 저장된 txt 파일 경로
    :param output_file_path: 모호한 문장이 포함된 데이터를 저장할 파일 경로
    :return: WSD 문제 세트가 포함된 DataFrame
    """
    import pandas as pd
    
    df = pd.read_csv(wsd_set_path)
    
    sentence_list = list()
    for idx, row in df.iterrows():
        prompt = row["prompt"]
        
        ambiguous_sentence = prompt.split("\n")[0].split("Text Context:")[-1].strip()
        sentence_list.append(ambiguous_sentence)
        
    df["word_phrase"] = sentence_list
    df = df[["word_index", "word", "word_phrase", "senses", "gold_image", "gold_pos", "gold_sense"]]
    df.to_csv(output_file_path, index=False, encoding='utf-8')

def image_data_export(wsd_set_path, image_dir, output_dir):
    """
    image_data_export의 Docstring
    
    :param wsd_set_path: WSD 데이터 세트가 저장된 파일 경로
    :param image_dir: 이미지가 저장된 디렉토리 경로
    :param output_dir: 이미지를 내보낼 디렉토리 경로
    """
    import os
    import shutil
    import pandas as pd
    from tqdm.auto import tqdm
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the WSD set to get the list of images to export
    df = pd.read_csv(wsd_set_path)
    image_filenames = set(df["gold_image"].dropna())
    
    for filename in tqdm(os.listdir(image_dir), desc="Exporting images"):
        if filename in image_filenames:
            src_path = os.path.join(image_dir, filename)
            dst_path = os.path.join(output_dir, filename)
            shutil.copy(src_path, dst_path)
    
    print(f"Images exported from {image_dir} to {output_dir}")

def extract_predicted_pos_with_wsd_set(predicted_pos_path, output_file_path):
    """
    extract_predicted_pos_with_wsd_set의 Docstring
    
    :param predicted_pos_path: file path of the predicted POS data
    :param output_file_path: file path to save the data set with extracted predicted POS
    """
    import pandas as pd
    
    predicted_pos_df = pd.read_csv(predicted_pos_path)
    
    pos_mapping = {
        "noun": "n",
        "verb": "v",
        "adjective": "a",
        "adverb": "r"
    }
    pos_list = list()
    for index, row in predicted_pos_df.iterrows():
        pred_label = row["generated_text"].split("A:")[-1].strip().lower()
        mapped_label = pos_mapping.get(pred_label, "invalid")
        pos_list.append(mapped_label)

    predicted_pos_df.loc[predicted_pos_df.index, "gold_pos"] = pos_list    
    predicted_pos_df = predicted_pos_df[["word_index", "word", "word_phrase", "gold_image", "gold_pos"]] # exclude "prompt" and "generated_text" columns
    predicted_pos_df.to_csv(output_file_path, index=False, encoding='utf-8')

def extract_predicted_senses_with_wsd_set(predicted_senses_path, output_file_path):
    """
    extract_predicted_senses_with_wsd_set의 Docstring
    
    :param predicted_senses_path: file path of the predicted senses data
    :param output_file_path: file path to save the data set with extracted predicted senses
    """
    import pandas as pd
    
    predicted_senses_df = pd.read_csv(predicted_senses_path)
    
    sense_list = list()
    for index, row in predicted_senses_df.iterrows():
        pred_label = row["generated_text"].split("A:")[-1].strip().lower()
        sense_list.append(pred_label)

    predicted_senses_df.loc[predicted_senses_df.index, "gold_sense"] = sense_list    
    predicted_senses_df = predicted_senses_df[["word_index", "word", "word_phrase", "gold_image", "gold_pos", "gold_sense"]] # exclude "prompt" and "generated_text" columns
    predicted_senses_df.to_csv(output_file_path, index=False, encoding='utf-8')

def extract_ambiguous_sentence_for_wsd_set(wsd_set_path, output_file_path):
    """
    extract_ambiguous_sentence_for_wsd_set의 Docstring
    
    :param wsd_set_path: WSD 문제 세트가 저장된 txt 파일 경로
    :param output_file_path: 모호한 문장이 포함된 데이터를 저장할 파일 경로
    :return: WSD 문제 세트가 포함된 DataFrame
    """
    import pandas as pd
    
    df = pd.read_csv(wsd_set_path)
    
    sentence_list = list()
    for idx, row in df.iterrows():
        
        ambiguous_sentence = row["generated_sentence"].strip()
        sentence_list.append(ambiguous_sentence)
        
    df["asen"] = sentence_list
    df = df[["word_index", "word", "asen", "senses", "gold_image", "gold_pos", "gold_sense"]]
    df.to_csv(output_file_path, index=False, encoding='utf-8')

def evaluate_wsd_accuracy(predicted_wsd_set_path):
    """
    evaluate_wsd_accuracy의 Docstring
    
    :param predicted_wsd_set_path: file path of the predicted WSD set
    :param gold_wsd_set_path: file path of the gold WSD set
    :param output_file_path: file path to save the evaluation results
    """
    import pandas as pd
    
    result_df = pd.read_csv(predicted_wsd_set_path)
    for index, row in result_df.iterrows():
        pred_label = row["generated_text"].split("A:")[-1].strip().lower()
        try:
            mapped_label = int(pred_label)
        except ValueError:
            mapped_label = -1
        result_df.at[index, "predicted_label"] = mapped_label
    acc = (result_df["predicted_label"] == result_df["gold_sense"]).sum() / len(result_df)
    print(f"Accuracy for {predicted_wsd_set_path.split('/')[-1]}: {acc:.4f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process WSD data and perform various tasks")
    parser.add_argument("--task", type=str, required=True, help="Task to perform: extract_data_from_vwsd, change_summarize_prompt", choices=["extract_data_from_vwsd", "change_summarize_prompt", "extract_predicted_pos", "extract_predicted_senses", "extract_ambiguous_sentences", "evaluate_wsd_accuracy"])
    parser.add_argument("--input_path", type=str, help="Path to the WSD set CSV file")
    parser.add_argument("--output_path", type=str, help="Path to save the output CSV")
    parser.add_argument("--search_result_path", type=str, help="Path to the search result CSV file (required for change_summarize_prompt task)")
    parser.add_argument("--prompt_template_path", type=str, help="Prompt template text file (required for change_summarize_prompt task)")
    parser.add_argument("--format_template", type=str, nargs='+', default="word context web_content entities", help="Format template for the prompt (default: word context web_content entities)")
    
    args = parser.parse_args()
    
    if args.task == "extract_data_from_vwsd":
        if not args.input_path or not args.output_path:
            print("For 'extract_data_from_vwsd' task, both --input_path and --output_path are required.")
            return
        sample_and_save_wsd_set(output_file_path=args.output_path, sample_size=-1, check_duplicates=False, data_path=args.input_path, gold_image_path=args.input_path.replace("data", "gold"))
    
    elif args.task == "change_summarize_prompt":
        if not args.input_path or not args.output_path or not args.search_result_path:
            print("For 'change_summarize_prompt' task, --input_path, --output_path, and --search_result_path are required.")
            return
        with open(args.prompt_template_path, 'r') as f:
            prompt_template = f.read()
        change_summarize_prompt_with_template(summarize_set_path=args.input_path, prompt_template=prompt_template, search_result_path=args.search_result_path, prompt_format_template=args.format_template.split(), output_file_path=args.output_path)
    
    elif args.task == "extract_predicted_pos":
        if not args.input_path or not args.output_path:
            print("For 'extract_predicted_pos' task, both --input_path and --output_path are required.")
            return
        extract_predicted_pos_with_wsd_set(predicted_pos_path=args.input_path, output_file_path=args.output_path)

    elif args.task == "extract_predicted_senses":
        if not args.input_path or not args.output_path:
            print("For 'extract_predicted_senses' task, both --input_path and --output_path are required.")
            return
        extract_predicted_senses_with_wsd_set(predicted_senses_path=args.input_path, output_file_path=args.output_path)

    elif args.task == "extract_ambiguous_sentences":
        if not args.inference_file_path or not args.output_file_path:
            print("For 'extract_ambiguous_sentences' task, both --inference_file_path and --output_file_path are required.")
            return
        extract_ambiguous_sentence_for_wsd_set(wsd_set_path=args.inference_file_path, output_file_path=args.output_file_path)

    elif args.task == "evaluate_wsd_accuracy":
        if not args.input_path:
            print("For 'evaluate_wsd_accuracy' task, --input_path is required.")
            return
        evaluate_wsd_accuracy(predicted_wsd_set_path=args.input_path)
    
if __name__ == "__main__":
    main()