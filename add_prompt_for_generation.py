"""
Automatically add a prompt into the dataset for a specific task.
"""

def add_pos_prompt_for_generation(dataset_path, output_path):
    """
    add pos classfication prompt for generation task
    """
    
    prompt_template = """Ambiguous word: {word}
---
{sense_list}

You are an expert linguistic annotator. Image, a specific target word found within the context of that image, and a list of potential senses for that word are given. Each sense is categorized by a Part of Speech (POS): {pos_list}.

Your goal is to determine the correct POS by selecting the sense that most directly aligns with the visual context.
---
Decision Logic & Rules
Follow these rules in order of priority to make your decision:

{rule_list}

Always consider which sense can be most directly and concretely represented in the image. Avoid overcomplicating the reasoning process; focus on the most straightforward interpretation of the visual information in relation to the senses provided.

---
After all your reasoning is finished, provide your final decision as the format 'A: [POS]', where [POS] is your chosen Part of Speech ({pos_list})."""

    noun_template = """Noun Senses:
{noun_senses}"""
    verb_template = """Verb Senses:
{verb_senses}"""
    adj_template = """Adjective Senses:
{adj_senses}"""
    adv_template = """Adverb Senses:
{adv_senses}"""

    verb_rules = """Verb is considered when:
* no noun sense is provided.
* the photo depicts an action—either directly or indirectly—rather than the object itself, and if there is a verb sense in the provided list that corresponds to the action shown in the photo."""
    adj_rules = """Adjective is considered when:
* the photo can not be considered to depict an object or action of given word.
* the photo depicts a state rather than an action or an object itself, and there is a corresponding sense in the list of adjectives"""
    adv_rules = """Adverb is considered when:
* the photo can not be considered to depict an object or action or state of given word."""

    import pandas as pd
    import json
    
    wsd_set_df = pd.read_csv(dataset_path)
    prompt_list = list()
    for index, row in wsd_set_df.iterrows():
        sense_dict = json.loads(row["senses"])
        noun_senses = sense_dict.get("n", [])
        verb_senses = sense_dict.get("v", [])
        adj_senses = sense_dict.get("a", [])
        adv_senses = sense_dict.get("r", [])
        
        pos_list = list()
        sense_string = ""
        rule_string = ""
        if len(noun_senses) > 0:
            sense_string += (noun_template.format(noun_senses="\n".join([f"{i + 1}) {sense}" for i, sense in enumerate(noun_senses)])) + "\n---\n")
            pos_list.append("noun")
        if len(verb_senses) > 0:
            sense_string += (verb_template.format(verb_senses="\n".join([f"{i + 1}) {sense}" for i, sense in enumerate(verb_senses)])) + "\n---\n")
            rule_string += verb_rules + '\n\n'
            pos_list.append("verb")
        if len(adj_senses) > 0:
            sense_string += (adj_template.format(adj_senses="\n".join([f"{i + 1}) {sense}" for i, sense in enumerate(adj_senses)])) + "\n---\n")
            rule_string += adj_rules + '\n\n'
            pos_list.append("adjective")
        if len(adv_senses) > 0:
            sense_string += (adv_template.format(adv_senses="\n".join([f"{i + 1}) {sense}" for i, sense in enumerate(adv_senses)])) + "\n---\n")
            rule_string += adv_rules + '\n\n'
            pos_list.append("adverb")
        
        prompt = prompt_template.format(
            word=row["word"],
            sense_list=sense_string,
            pos_list=", ".join(pos_list),
            rule_list=rule_string,
        )
        prompt_list.append(prompt)

    wsd_set_df["prompt"] = prompt_list
    wsd_set_df.to_csv(output_path, index=False)

def add_sense_identification_prompt_for_generation(dataset_path, output_path, summarized_set_path, top_k=2):
    """
    add sense identification prompt for generation task
    """
    # Implement the logic for adding sense identification prompts here
    prompt_template = """Text Context: {context}
Ambiguous Word: {word}
---
Sense List:
{sense_list}
---
Image Web Search Results:
{web_search_results}
---
You are an expert linguistic annotator. A specific target word found within the context of a given text, and a list of potential senses for that word are provided. Your task is to determine the correct sense by selecting the one that most directly aligns with the context and its background event.
---
Decision Logic & Rules
Follow these rules in order of priority to make your decision:

First, Use web search results to determine whether there is any background information associated with the given image that cannot be discerned solely from its visual elements. If such information exists, prioritize the sense that best fits this background information, even if it is not the most direct match for the visual content.

Directness of Meaning: If no additional background information can be obtained from the web search results, choose the sense that provides the most direct and specific fit for the visual and linguistic context. Even if a perfect match does not exist, select the sense that is the closest indirect match.
---
After all your reasoning is finished, read through the list of senses provided once again, carefully and in detail, one by one. Then, provide your final decision as the format 'A: [sense number]' and end your generation. In here, [sense number] is the index of your chosen sense from the provided list."""

    import pandas as pd
    import json
    
    wsd_set_df = pd.read_csv(dataset_path)
    summarized_set_df = pd.read_csv(summarized_set_path)
    
    prompt_list = list()
    for index, row in wsd_set_df.iterrows():
        sense_dict = json.loads(row["senses"])
        sense_list = sense_dict.get(row["gold_pos"], [])
        web_search_results = summarized_set_df.loc[summarized_set_df["word_index"] == row["word_index"], "generated_text"].tolist()
        
        cropped_web_search_results = list()
        while len(cropped_web_search_results) < top_k and len(web_search_results) > 0:
            next_result = web_search_results.pop(0).strip()
            if next_result != "" and "Not Relevant" not in next_result:
                next_result = next_result.replace("Relevant\n", "").strip()
                cropped_web_search_results.append(next_result)
        
        prompt = prompt_template.format(
            word=row["word"],
            context=row["word_phrase"],
            sense_list="\n".join([f" {idx + 1}. {sense}" for idx, sense in enumerate(sense_list)]),
            web_search_results="\n".join(cropped_web_search_results)
        )
        prompt_list.append(prompt)

    wsd_set_df["prompt"] = prompt_list
    wsd_set_df.to_csv(output_path, index=False)

def add_ambiguous_sentence_generation_prompt_for_generation(dataset_path, output_path):
    """
    add ambiguous sentence generation prompt for generation task
    """
    # Implement the logic for adding ambiguous sentence generation prompts here
    prompt_template = """the {pos} '{word}' that can be interpreted as:
{sense_list}
---
Generate a natural ambiguous sentence using polysemous word '{word}' once so that the sentence can be interpreted in more than one meaning.
In the generated sentence, the {pos} '{word}' should be interpreted as '{gold_sense}'. However, the sentence should not imply any of the given definitions too strongly. The sentence should be natural and commonly used in daily life.
Suggest your final sentence in the format 'Generated Sentence: [your sentence]'. Do not include any explanation, and end your generation after providing the sentence."""

    import pandas as pd
    import json
    
    wsd_set_df = pd.read_csv(dataset_path)
    prompt_list = list()
    
    pos_mapping = {
        "n": "noun",
        "v": "verb",
        "a": "adjective",
        "r": "adverb"
    }

    for _, row in wsd_set_df.iterrows():
        sense_dict = json.loads(row["senses"])
        sense_list = sense_dict.get(row["gold_pos"], [])
        gold_sense = int(row["generated_text"].split("A:")[-1].strip())
        
        
        prompt = prompt_template.format(
            word=row["word"],
            sense_list="\n".join([f" {idx + 1}. {sense}" for idx, sense in enumerate(sense_list)]),
            gold_sense=sense_list[gold_sense - 1] if 0 < gold_sense <= len(sense_list) else "Unknown Sense",
            pos=pos_mapping.get(row["gold_pos"], row["gold_pos"])
        )
        prompt_list.append(prompt)

    wsd_set_df["prompt"] = prompt_list
    wsd_set_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add prompt for generation task")
    parser.add_argument("--task", type=str, required=True, choices=["pos_classification", "sense_identification", "ambiguous_sentence_generation"], help="Task for which to add prompts")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the input dataset CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output dataset CSV file with prompts")
    parser.add_argument("--summarized_set_path", type=str, help="Path to the summarized set CSV file")
    parser.add_argument("--top_k", type=int, default=2, help="Number of top web search results to include in the prompt (only for sense_identification task)")

    args = parser.parse_args()

    if args.task == "pos_classification":
        add_pos_prompt_for_generation(args.dataset_path, args.output_path)
    elif args.task == "sense_identification":
        add_sense_identification_prompt_for_generation(args.dataset_path, args.output_path, args.summarized_set_path, args.top_k)
    elif args.task == "ambiguous_sentence_generation":
        add_ambiguous_sentence_generation_prompt_for_generation(args.dataset_path, args.output_path)