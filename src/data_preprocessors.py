"""
Tokenization functions for different datasets.
"""

import functools
import logging
from itertools import chain

import transformers
from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast
from transformers.testing_utils import CaptureLogger

from .arguments import Arguments

logger = logging.getLogger(__name__)
# since this will be pickled to avoid _LazyModule error in Hasher force
# logger loading before preprocess_function
tok_logger = transformers.utils.logging.get_logger(
    "transformers.tokenization_utils_base"
)

TEXT_COLUMN_NAME = "text"


def clm_tokenize_function(examples, tokenizer):
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples[TEXT_COLUMN_NAME])
    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input "
            "will be chunked into smaller bits"
            " before being passed to the model."
        )
    return output


def clm_tokenize_and_group_function(examples, tokenizer, block_size: int):
    """
    Tokenize and group functions in one pass, to prevent needing to store
    multiple copies of large datasets
    """
    tokenized = clm_tokenize_function(examples, tokenizer)
    grouped = group_texts(tokenized, block_size)
    return grouped


def group_texts(examples, block_size: int):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model
    # supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def get_block_size(args, tokenizer):
    # Compute the block size first.
    if args.data.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The tokenizer picked seems to have a very large `model_max_length` "
                f"({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing "
                "--block_size xxx."
            )
            block_size = 1024
    else:
        if args.data.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.data.block_size}) is larger than the "
                "maximum length for the model"
                f"({tokenizer.model_max_length}). Using "
                f"block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.data.block_size, tokenizer.model_max_length)
    return block_size


def filter_max_length(example, input_max_length: int, label_max_length: int):
    if len(example["input_ids"]) > input_max_length:
        return False
    if "labels" in example and len(example["labels"]) > label_max_length:
        return False
    return True


def preprocess(
    raw_datasets: DatasetDict,
    args: Arguments,
    tokenizer: PreTrainedTokenizerFast,
) -> DatasetDict:
    with args.training.main_process_first(desc="preprocessing"):
        return preprocess_clm(raw_datasets, args, tokenizer)


def preprocess_clm(
    raw_datasets: DatasetDict,
    args: Arguments,
    tokenizer: PreTrainedTokenizerFast,
) -> DatasetDict:
    block_size = get_block_size(args, tokenizer)
    tokenize_and_group_function = functools.partial(
        clm_tokenize_and_group_function,
        tokenizer=tokenizer,
        block_size=block_size,
    )

    # Get names of columns to remove.
    if args.training.do_train:
        columns_to_remove = raw_datasets["train"].column_names
    else:
        columns_to_remove = raw_datasets["validation"].column_names

    lm_datasets = raw_datasets.map(
        tokenize_and_group_function,
        batched=True,
        num_proc=args.data.preprocessing_num_workers,
        remove_columns=columns_to_remove,
        load_from_cache_file=not args.data.overwrite_cache,
        desc="Preprocessing data",
    )
    lm_datasets.set_format("torch")
    return lm_datasets
