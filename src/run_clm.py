<<<<<<< HEAD
#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL,
...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task.
# Pointers for this are left as comments.

import logging
import math
import os

import hydra
import torch  # noqa
from datasets import load_dataset
from omegaconf.dictconfig import DictConfig
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from . import data_preprocessors
from .arguments import Arguments, global_setup
from .integrations import CustomWandbCallback

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from transformers import GPT2TokenizerFast

# Will error if the minimal version of Transformers is not installed. Remove at
# your own risks.
check_min_version("4.23.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


import evaluate  # Load after torch to avoid errors.  # noqa

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@hydra.main(config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    args: Arguments = global_setup(args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(args.training.output_dir)
        and args.training.do_train
        and not args.training.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(args.training.output_dir)
        if last_checkpoint is None and len(os.listdir(args.training.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.training.output_dir}) already exists and "
                "is not empty. Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and args.training.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To "
                "avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from "
                "scratch."
            )

    # Set seed before initializing model.
    set_seed(args.training.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training
    # and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the
    # hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the
    # first column if no column called 'text' is found. You can easily tweak
    # this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one
    # local process can concurrently download the dataset.
    if args.data.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.data.dataset_name,
            args.data.dataset_config_name,
            streaming=args.data.streaming,
            cache_dir=args.model.cache_dir,
            use_auth_token=True if args.model.use_auth_token else None,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.data.dataset_name,
                args.data.dataset_config_name,
                streaming=args.data.streaming,
                split=f"train[:{args.data.validation_split_percentage}%]",
                cache_dir=args.model.cache_dir,
                use_auth_token=True if args.model.use_auth_token else None,
            )
            raw_datasets["train"] = load_dataset(
                args.data.dataset_name,
                args.data.dataset_config_name,
                streaming=args.data.streaming,
                split=f"train[{args.data.validation_split_percentage}%:]",
                cache_dir=args.model.cache_dir,
                use_auth_token=True if args.model.use_auth_token else None,
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.data.train_file is not None:
            data_files["train"] = args.data.train_file
        if args.data.validation_file is not None:
            data_files["validation"] = args.data.validation_file
        extension = (
            args.data.train_file.split(".")[-1]
            if args.data.train_file is not None
            else args.data.validation_file.split(".")[-1]
        )
        if extension == "txt" or extension == "train":
            extension = "text"
            dataset_args["keep_linebreaks"] = args.data.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            streaming=args.data.streaming,
            data_files=data_files,
            cache_dir=args.model.cache_dir,
            use_auth_token=True if args.model.use_auth_token else None,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be
        # used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                streaming=args.data.streaming,
                data_files=data_files,
                split=f"train[:{args.data.validation_split_percentage}%]",
                cache_dir=args.model.cache_dir,
                use_auth_token=True if args.model.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                streaming=args.data.streaming,
                data_files=data_files,
                split=f"train[{args.data.validation_split_percentage}%:]",
                cache_dir=args.model.cache_dir,
                use_auth_token=True if args.model.use_auth_token else None,
                **dataset_args,
            )

    config_kwargs = {
        "cache_dir": args.model.cache_dir,
        "revision": args.model.model_revision,
        "use_auth_token": True if args.model.use_auth_token else None,
    }
    if args.model.config_name:
        config = AutoConfig.from_pretrained(args.model.config_name, **config_kwargs)
    elif args.model.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[args.model.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if args.model.config_overrides is not None:
            logger.info(f"Overriding config: {args.model.config_overrides}")
            config.update_from_string(args.model.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": args.model.cache_dir,
        "use_fast": args.model.use_fast_tokenizer,
        "revision": args.model.model_revision,
        "use_auth_token": True if args.model.use_auth_token else None,
    }

    if args.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
             args.model.tokenizer_name, **tokenizer_kwargs
         )

    elif args.model.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
             args.model.model_name_or_path, **tokenizer_kwargs
         )

    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported "
            "by this script."
            "You can do it from another script, save it, and load it from here, using "
            "--tokenizer_name."
        )

    bos = '<|bos|>'
    eos = '<|eos|>'
    pad = '<|pad|>'
    special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer_token = Tokenizer.from_pretrained("gpt2")
    tokenizer_token.post_processor = TemplateProcessing(
    single=bos + " $A " + eos,
    special_tokens=[(eos, tokenizer.eos_token_id), (bos, tokenizer.bos_token_id)],
    )
    tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer_token)


   # sample_sentence = 'I like eating pizza'
   # enconding = tokenizer(sample_sentence, return_tensors = "pt", add_special_tokens=True)['input_ids']

    # Initialize model
    model = AutoModelForCausalLM.from_config(config)
    n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    logger.info(
        f"Training new model from scratch - Total size={n_params/2**20:.2f}M params"
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocess datasets.
    lm_datasets = data_preprocessors.preprocess(
        raw_datasets,
        args,
        tokenizer,
    )
    if args.training.only_preprocess_dataset:
        logger.info("Was asked to only preprocess dataset. Exiting!")
        return

    if args.training.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if args.data.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), args.data.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if args.training.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        # Shuffle eval in case we are truncating
        eval_dataset = eval_dataset.shuffle()
        if args.data.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), args.data.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            del labels
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has
            # been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    custom_callbacks = []
    if args.wandb.log:
        custom_callbacks.append(CustomWandbCallback(args))

    

    trainer = Trainer(
        model=model,
        args=args.training,
        train_dataset=train_dataset if args.training.do_train else None,
        eval_dataset=eval_dataset if args.training.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
        if args.training.do_eval and not is_torch_tpu_available()
        else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if args.training.do_eval and not is_torch_tpu_available()
        else None,
        callbacks=custom_callbacks,
    )

    # Training
    if args.training.do_train:
        checkpoint = None
        if args.training.resume_from_checkpoint is not None:
            checkpoint = args.training.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            args.data.max_train_samples
            if args.data.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if args.training.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            args.data.max_eval_samples
            if args.data.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {
        "finetuned_from": args.model.model_name_or_path,
        "tasks": "text-generation",
    }
    if args.data.dataset_name is not None:
        kwargs["dataset_tags"] = args.data.dataset_name
        if args.data.dataset_config_name is not None:
            kwargs["dataset_args"] = args.data.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{args.data.dataset_name} {args.data.dataset_config_name}"
        else:
            kwargs["dataset"] = args.data.dataset_name

    if args.training.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
=======
#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL,
...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task.
# Pointers for this are left as comments.

import logging
import math
import os

import hydra
import torch  # noqa
from datasets import load_dataset
from omegaconf.dictconfig import DictConfig
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Tokenizer,
    Trainer,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from . import data_preprocessors
from .arguments import Arguments, global_setup
from .integrations import CustomWandbCallback

# Will error if the minimal version of Transformers is not installed. Remove at
# your own risks.
check_min_version("4.23.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


import evaluate  # Load after torch to avoid errors.  # noqa

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@hydra.main(config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    args: Arguments = global_setup(args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(args.training.output_dir)
        and args.training.do_train
        and not args.training.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(args.training.output_dir)
        if last_checkpoint is None and len(os.listdir(args.training.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.training.output_dir}) already exists and "
                "is not empty. Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and args.training.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To "
                "avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from "
                "scratch."
            )

    # Set seed before initializing model.
    set_seed(args.training.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training
    # and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the
    # hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the
    # first column if no column called 'text' is found. You can easily tweak
    # this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one
    # local process can concurrently download the dataset.
    if args.data.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.data.dataset_name,
            args.data.dataset_config_name,
            streaming=args.data.streaming,
            cache_dir=args.model.cache_dir,
            use_auth_token=True if args.model.use_auth_token else None,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.data.dataset_name,
                args.data.dataset_config_name,
                streaming=args.data.streaming,
                split=f"train[:{args.data.validation_split_percentage}%]",
                cache_dir=args.model.cache_dir,
                use_auth_token=True if args.model.use_auth_token else None,
            )
            raw_datasets["train"] = load_dataset(
                args.data.dataset_name,
                args.data.dataset_config_name,
                streaming=args.data.streaming,
                split=f"train[{args.data.validation_split_percentage}%:]",
                cache_dir=args.model.cache_dir,
                use_auth_token=True if args.model.use_auth_token else None,
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.data.train_file is not None:
            data_files["train"] = args.data.train_file
        if args.data.validation_file is not None:
            data_files["validation"] = args.data.validation_file
        extension = (
            args.data.train_file.split(".")[-1]
            if args.data.train_file is not None
            else args.data.validation_file.split(".")[-1]
        )
        if extension == "txt" or extension == "train":
            extension = "text"
            dataset_args["keep_linebreaks"] = args.data.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            streaming=args.data.streaming,
            data_files=data_files,
            cache_dir=args.model.cache_dir,
            use_auth_token=True if args.model.use_auth_token else None,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be
        # used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                streaming=args.data.streaming,
                data_files=data_files,
                split=f"train[:{args.data.validation_split_percentage}%]",
                cache_dir=args.model.cache_dir,
                use_auth_token=True if args.model.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                streaming=args.data.streaming,
                data_files=data_files,
                split=f"train[{args.data.validation_split_percentage}%:]",
                cache_dir=args.model.cache_dir,
                use_auth_token=True if args.model.use_auth_token else None,
                **dataset_args,
            )

    config_kwargs = {
        "cache_dir": args.model.cache_dir,
        "revision": args.model.model_revision,
        "use_auth_token": True if args.model.use_auth_token else None,
    }
    if args.model.config_name:
        config = AutoConfig.from_pretrained(args.model.config_name, **config_kwargs)
    elif args.model.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[args.model.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if args.model.config_overrides is not None:
            logger.info(f"Overriding config: {args.model.config_overrides}")
            config.update_from_string(args.model.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": args.model.cache_dir,
        "use_fast": args.model.use_fast_tokenizer,
        "revision": args.model.model_revision,
        "use_auth_token": True if args.model.use_auth_token else None,
    }
    if args.model.tokenizer_name:
        tokenizer = GPT2Tokenizer.from_pretrained(
            args.model.tokenizer_name, **tokenizer_kwargs
        )
    elif args.model.model_name_or_path:
        tokenizer = GPT2Tokenizer.from_pretrained(
            args.model.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported "
            "by this script."
            "You can do it from another script, save it, and load it from here, using "
            "--tokenizer_name."
        )

    # Initialize model
    model = AutoModelForCausalLM.from_config(config)
    n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    logger.info(
        f"Training new model from scratch - Total size={n_params/2**20:.2f}M params"
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocess datasets.
    lm_datasets = data_preprocessors.preprocess(
        raw_datasets,
        args,
        tokenizer,
    )
    if args.training.only_preprocess_dataset:
        logger.info("Was asked to only preprocess dataset. Exiting!")
        return

    if args.training.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if args.data.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), args.data.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if args.training.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        # Shuffle eval in case we are truncating
        eval_dataset = eval_dataset.shuffle()
        if args.data.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), args.data.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            del labels
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has
            # been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    custom_callbacks = []
    if args.wandb.log:
        custom_callbacks.append(CustomWandbCallback(args))

    trainer = Trainer(
        model=model,
        args=args.training,
        train_dataset=train_dataset if args.training.do_train else None,
        eval_dataset=eval_dataset if args.training.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
        if args.training.do_eval and not is_torch_tpu_available()
        else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if args.training.do_eval and not is_torch_tpu_available()
        else None,
        callbacks=custom_callbacks,
    )

    # Training
    if args.training.do_train:
        checkpoint = None
        if args.training.resume_from_checkpoint is not None:
            checkpoint = args.training.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            args.data.max_train_samples
            if args.data.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if args.training.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            args.data.max_eval_samples
            if args.data.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {
        "finetuned_from": args.model.model_name_or_path,
        "tasks": "text-generation",
    }
    if args.data.dataset_name is not None:
        kwargs["dataset_tags"] = args.data.dataset_name
        if args.data.dataset_config_name is not None:
            kwargs["dataset_args"] = args.data.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{args.data.dataset_name} {args.data.dataset_config_name}"
        else:
            kwargs["dataset"] = args.data.dataset_name

    if args.training.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
>>>>>>> remotes/origin/dev-tjlittle-6_layer_model
