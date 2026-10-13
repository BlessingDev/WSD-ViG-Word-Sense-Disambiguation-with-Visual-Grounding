"""
Calculate the probability of each answer token in a given context.

"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

def calculate_answer_token_probability(csv_path, model_checkpoint, output_csv_path):
    """
    Calculate the probability of each answer token in a given context.
    
    Args:
        csv_path (str): Path to the input CSV file containing 'prompt' and 'generated_text' columns.
        model_checkpoint (str): The model checkpoint to use for tokenization and probability calculation.
        output_csv_path (str): Path to save the output CSV file with probabilities.
    """
    
    # load the model and processor
    processor = AutoProcessor.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        device_map="auto",  # Automatically map model layers to available devices
        dtype=torch.bfloat16  # Use float16 for faster inference and lower memory usage
    )
    
    # Load the input CSV file and process it
    import pandas as pd
    import json
    from tqdm import tqdm
    df = pd.read_csv(csv_path)
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        prompt = row['prompt']
        generated_text = row['generated_text']

        # cut the answer generated in generated_text
        generated_text = generated_text.split("A:")[0] + "A: "
        
        # Tokenize the prompt and generated text
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": generated_text}
        ]
        
        prompt = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_tokens=False
        )
        
        # remove end_of_turn token at the end of the prompt
        prompt = prompt[0][:-2] + [236743] # add space token so that the next token be the first token of the answer
        
        with torch.no_grad():
            prompt = torch.tensor(prompt).unsqueeze(0).to(model.device)
            outputs = model(prompt)
            logits = outputs.logits
            
            # Get the last token's logits
            last_token_logits = logits[0, -1, :]
            
            # Calculate probabilities for top 3 answer tokens
            probabilities = torch.softmax(last_token_logits, dim=-1)
            top3_probs, top3_indices = torch.topk(probabilities, k=3)
            
            # Store the top 3 answer tokens and their probabilities in the DataFrame as a dictionary
            answer_token_probabilities = {
                processor.tokenizer.decode(top3_indices[i].item()): top3_probs[i].item() for i in range(3)
            }
            
            df.at[index, 'answer_token_probabilities'] = json.dumps(answer_token_probabilities)
        
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate answer token probabilities and save to CSV.")
    parser.add_argument("--csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("--model_checkpoint", type=str, help="Model checkpoint to use for tokenization and probability calculation.")
    parser.add_argument("--output_csv_path", type=str, help="Path to save the output CSV file with probabilities.")

    args = parser.parse_args([
        "--csv_path", "/workspace/data/test_set_process/inference/sense_ambiguous_sentence/wsd_set_entire_labeled_ambiguous_sentence_sense_text_gemma-3-27b-it.csv",
        "--model_checkpoint", "google/gemma-3-27b-it",
        "--output_csv_path", "/workspace/data/test_set_process/inference/sense_ambiguous_sentence/wsd_set_entire_labeled_ambiguous_sentence_sense_text_gemma-3-27b-it_answer_token_probabilities.csv"
    ])
    #args = parser.parse_args()

    calculate_answer_token_probability(args.csv_path, args.model_checkpoint, args.output_csv_path)