"""
Inter-sense BERT Score

- sense 간에 BERT Score를 계산하는 모듈
- csv를 가지고 와서 sense 간의 bert score를 계산하고, 각 row에 score를 추가한 csv를 저장
"""

def inter_sense_maximum_bert_score(input_csv_path, output_csv_path):
    """
    input_csv_path: str, input csv file path
    output_csv_path: str, output csv file path
    """
    import json
    import pandas as pd
    from bert_score import BERTScorer

    # Load the input CSV file
    df = pd.read_csv(input_csv_path)

    scorer = BERTScorer(
        model_type="answerdotai/ModernBERT-base",
        num_layers=18,
        rescale_with_baseline=False,
        device="cuda",  # 또는 "cpu"
        batch_size=32   # 평가 루프 내 배치 크기 기본값 설정
    )
    
    for index, row in df.iterrows():
        sense_dict = json.loads(row['senses'])  # Assuming the column name is 'sense_dict'
        sense_list = sense_dict.get(row["gold_pos"], [])  # Get the list of senses for the gold_pos
        
        # 리그처럼 1에 대해 나머지 비교, 2에 대해 나머지 비교, 3에 대해 나머지 비교, ... 이런 식으로 반복
        # BERT Score는 양방향으로 비교하지 않고, 한쪽을 기준으로 나머지와 비교하는 방식으로 계산
        max_f1 = 0.0
        for i, sense1 in enumerate(sense_list[:-1]):  # Exclude the last sense since it has no pair to compare with
            sense2_list = sense_list[i+1:]

            # Calculate BERT Score for each pair of senses
            P, R, F1 = scorer.score(
                cands=sense2_list,
                refs=[sense1] * len(sense2_list)
            )
            cur_max = F1.max().item()  # Get the maximum F1 score from the current comparisons
            max_f1 = max(max_f1, cur_max)  # Update the maximum F1 score if the current one is higher
        
        df.at[index, 'max_f1'] = max_f1

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate inter-sense BERT Score and save to CSV.")
    parser.add_argument("--input_csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("--output_csv_path", type=str, help="Path to the output CSV file.")
    parser.add_argument("--task", type=str, default="max_f1", help="Task to perform (default: max_f1)", choices=["max_f1"])

    args = parser.parse_args()

    inter_sense_maximum_bert_score(args.input_csv_path, args.output_csv_path)