"""
Sample generation from model files defined in the `exp` folder.
"""

import argparse

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    MaxLengthCriteria,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
)

argp = argparse.ArgumentParser()
argp.add_argument(
    "--model_path",
    default="debug-gpt2-verysmall-babylm_10M",
    help="(i.e. --model_path=debug-gpt2-small-babylm_10M)",
)
argp.add_argument(
    "--input_text", default=["I feel"], nargs="+", help='(i.e. --input_text "I feel")'
)
argp.add_argument("--max_length", default=20, type=int, help="(i.e. --max_length=20)")
argp.add_argument("--min_length", default=10, type=int, help="(i.e. --min_length=10)")
argp.add_argument(
    "--num_return_sequences", default=1, help="(i.e. --num_return_sequences=1)"
)
args = argp.parse_args()

# Load the trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "exp/" + args.model_path + "/" + args.model_path + "-run-42"
)  # adjust suffix if the seed is changed
tokenizer = AutoTokenizer.from_pretrained(
    "exp/" + args.model_path + "/" + args.model_path + "-run-42"
)  # adjust suffix if the seed is changed

input_prompt = args.input_text[0]
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

# Set pad_token_id to eos_token_id because GPT2 does not have a PAD token
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# Instantiate logits processors and stopping criteria objects
logits_processor = LogitsProcessorList(
    [
        MinLengthLogitsProcessor(
            min_length=args.min_length,
            eos_token_id=model.generation_config.eos_token_id,
        ),
    ]
)
stopping_criteria = StoppingCriteriaList(
    [MaxLengthCriteria(max_length=args.max_length)]
)

# Generate the output
outputs = model.greedy_search(
    input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
)

output = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Print the output
print("\nINPUT:\n", args.input_text[0])
print("\nGENERATED RESPONSE:\n", output[0])
