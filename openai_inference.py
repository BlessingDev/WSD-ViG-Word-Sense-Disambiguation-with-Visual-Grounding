import os
from openai import OpenAI
from dotenv import load_dotenv

from argparse import ArgumentParser
import pandas as pd
from tqdm.auto import tqdm

# 1. 환경 변수 로드 (.env 파일에 저장된 API 키를 읽어옴)
load_dotenv()

# 2. 클라이언트 객체 생성 (자동으로 OPENAI_API_KEY 환경변수를 참조함)
client = OpenAI()

def get_gpt_response(args, user_prompt):
    try:
        # 3. API 호출 (GPT-4o 모델 기준)
        response = client.responses.create(
            model=args.gpt_model,  # 사용할 모델명 (gpt-4o, gpt-4-turbo 등)
            input=[
                {"role": "system", "content": "You are a skilled linguistic expert."},
                {"role": "user", "content": user_prompt}
            ],
            reasoning={"effort": args.reasoning_effort},  # 추론 노력 조절 (low, medium, high)
        )
        
        # 4. 결과 출력
        return response.output_text

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return "None"

# 실행 테스트
if __name__ == "__main__":
    parser = ArgumentParser(description="Test OpenAI API with a custom prompt")
    parser.add_argument("--gpt_model", type=str, default="gpt-5.4-mini", help="GPT model to use for generation")
    parser.add_argument("--reasoning_effort", type=str, default="low", choices=["none", "low", "medium", "high"], help="Reasoning effort level for the model")
    parser.add_argument("--wsd_set_path", type=str, required=True, help="Path to the WSD set CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output CSV file")
    
    args = parser.parse_args()
    '''args = parser.parse_args([
        "--gpt_model", "gpt-5.4-mini",
        "--reasoning_effort", "medium",
        "--wsd_set_path", "/workspace/data/test_set_process/wsd_set_entire_labeled_ambiguous_sentence_sense_prompt.csv",
        "--output_path", "/workspace/data/test_set_process/wsd_set_anemia.csv"
    ])'''
    
    print(f"Loading WSD set from: {args.wsd_set_path}")
    df = pd.read_csv(args.wsd_set_path)
    '''df = pd.DataFrame({
        "prompt": ["""the noun 'anemia' that can be interpreted as:
 1. a deficiency of red blood cells
 2. a lack of vitality
 3. genus of terrestrial or lithophytic ferns having pinnatifid fronds; chiefly of tropical America
---
Generate a natural ambiguous sentence using polysemous word 'anemia' once so that the sentence can be interpreted in more than one meaning.
In the generated sentence, the noun 'anemia' should be interpreted as 'genus of terrestrial or lithophytic ferns having pinnatifid fronds; chiefly of tropical America'. However, the sentence should not imply any of the given definitions too strongly. The sentence should be natural and commonly used in daily life.
Suggest your final sentence in the format 'Generated Sentence: [your sentence]'. Do not include any explanation, and end your generation after providing the sentence."""]
    })'''
    answer_list = list()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating sentences"):
        prompt = row['prompt']
    
        result = get_gpt_response(args, prompt)
        
        # result 정제
        result = result.split("Generated Sentence:")[-1].strip() if "Generated Sentence:" in result else result.strip()
        answer_list.append(result)
    
    df["generated_sentence"] = answer_list
    print(f"Saving results to: {args.output_path}")
    df.to_csv(args.output_path, index=False)