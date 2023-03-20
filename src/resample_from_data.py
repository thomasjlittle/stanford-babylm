"""
Sample generation from model files defined in the `exp` folder.
"""

import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

argp = argparse.ArgumentParser()
argp.add_argument(
    "--model_path",
    default="debug-gpt2-xsmall-babylm_10M",
    help="(i.e. --model_path=debug-gpt2-xsmall-babylm_10M)",
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

root_dir = "/home/ubuntu/stanford-babylm/babylm_data/babylm_10M"

# Load the trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "exp/" + args.model_path + "/" + args.model_path + "-run-42"
)  # adjust suffix if the seed is changed
tokenizer = AutoTokenizer.from_pretrained(
    "exp/" + args.model_path + "/" + args.model_path + "-run-42"
)  # adjust suffix if the seed is changed

# Set pad_token_id to eos_token_id because GPT2 does not have a PAD token
model.generation_config.pad_token_id = model.generation_config.eos_token_id

final_dir = "/home/ubuntu/stanford-babylm/babylm_data/"
to_dir = "resample_50k"
to_file = os.path.join(final_dir, to_dir)

if not os.path.exists(to_file):
    os.mkdir(to_file)


count_dir = {
    "aochildes.train": 429271,
    "bnc_spoken.train": 823446,
    # "cbt.train": 288992,
    # "children_stories.train": 18820,
    # "gutenberg.train":313446,
    "open_subtitles.train": 461994,
    "qed.train": 665159,
    # "simple_wikipedia.train":2172983,
    # "switchboard.train": 88983,
    # "wikipedia.train": 398946
}

total_word_count = []
for file_name in os.listdir(root_dir):
    num_words = 0
    if file_name.split(".")[-1] == "train" and (file_name in count_dir):
        print("-" * 100)
        print("Filename:", file_name)
        total_file = []
        with open(os.path.join(root_dir, file_name), "r") as f:
            new_name = to_dir + "_" + file_name
            sent_count = 0
            for line in f:
                if sent_count <= 1000:
                    sent_count += 1
                    continue
                words = line.split()
                if len(words) == 0:
                    continue
                sent_count += 1
                first_word = words[0]
                input_ids = tokenizer(first_word, return_tensors="pt").input_ids
                sample_output = model.generate(
                    input_ids,
                    do_sample=True,
                    max_length=min(len(words), 512),
                    top_k=50,
                    top_p=0.95,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                )
                output = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                # print(output)
                if (len(output.split()) == 0) or (output == "\n"):
                    continue
                num_words += len(output.split())
                total_file.append(output + "\n")

                # Save for every 100 sentences generated
                if sent_count % 100 == 0:
                    file1 = open(os.path.join(to_file, new_name), "w")
                    file1.writelines(total_file)
                    file1.close()
                    print("-" * 100)
                    print("Save path:", os.path.join(to_file, new_name))

                if num_words >= count_dir[file_name]:
                    file1 = open(os.path.join(to_file, new_name), "w")
                    file1.writelines(total_file)
                    file1.close()
                    print("-" * 100)
                    print("Finished path:", os.path.join(to_file, new_name))
                    break

            print("Number of words:", num_words)
            total_word_count.append(num_words)
print("Total count:", sum(total_word_count))
