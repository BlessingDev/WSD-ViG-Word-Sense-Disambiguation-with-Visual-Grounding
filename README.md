The paper is under submission stage.

## Reproducing Dataset Construction & Baselines

To reproduce the automated dataset construction pipeline and evaluate the baseline models, step-by-step Bash scripts are provided in the `ready_scripts/` directory:

- `ready_scripts/construct0_extract_and_filter.sh`: extract and filter seed words from V-WSD
- `ready_scripts/construct1_pos_classification.sh`: perform pos classification to obtain gold_pos
- `ready_scripts/construct2_sense_identification.sh`: perform sense identification to obtain gold_sense
- `ready_scripts/construct3_ambiguous_sentence_generation.sh`: perform ambiguous sentence generation using gpu-5.4-mini model
- `ready_scripts/baseline_evaluation.sh`: simple baseline inference and evaluation script under gemma-3-27b-it R summarization, k=2, and gemma-3-27b-it WSD inference
