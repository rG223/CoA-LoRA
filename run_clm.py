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
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
from optimize_config_lora import QuantConfigDataset
import logging
import math
import dill
import re
import random
from torch.utils.data import DataLoader
import os
import sys
import click
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Dict, Any

import datasets
import evaluate
import torch
from datasets import (
    load_dataset,
    concatenate_datasets)
import json
from tqdm import tqdm
import transformers
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from models import misc_utils
from models import lora_utils
from models import distributed_utils
from experiments import callback_utils
import numpy as np

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    lora_num_ranks: int = field(default=8)
    lora_dropout: float = field(default=0.05)
    lora_config: Optional[str] = field(default=None)
    lora_model_name: Optional[str] = field(default=None)
    hf_quantization_method: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    use_bay: Optional[bool] = field(default=True)
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    config_data_dir: Optional[str] = field(
        default= "/mnt/data1/big_file/yerg/quant_conf_mip")

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

def check_model_device(model):
    devices = set()
    for name, param in model.named_parameters():
        devices.add(param.device)
    for name, buf in model.named_buffers():
        devices.add(buf.device)

def main(return_trainer: bool = False):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)
    accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name == "c4":
        misc_utils.swarn(
            f"Using C4 dataset (`dataset_name` "
            f"= {data_args.dataset_name})",
            bg="yellow")
        raw_datasets = load_dataset(
            "json",
            data_files={
                "train": "/mnt/data1/yerg/c4-train.00000-of-01024.json.gz",
                "validation": "/mnt/data1/yerg/c4-validation.00000-of-00008.json.gz",
            },
        )
        _wikitext_dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test")
        # Hacks to be consistent with other works' preprocessing.
        wikitext_dataset = datasets.Dataset.from_dict(
            {
                "text": [
                    # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                    "\n\n".join(_wikitext_dataset["text"])
                ],
            },
        )
        # Hacks to get around the `remove_columns` to be used later.
        wikitext_dataset = (
            wikitext_dataset  # type: ignore
            .add_column(
                name="timestamp",
                column=wikitext_dataset["text"])
            .add_column(
                name="url",
                column=wikitext_dataset["text"])
        )
        raw_datasets["wikitext"] = wikitext_dataset
    elif data_args.dataset_name == "c4-wiki-large":
        misc_utils.swarn(
            f"Using C4+WikiText2 dataset (`dataset_name` "
            f"= {data_args.dataset_name})",
            bg="yellow")
        raw_datasets = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={
                "train": [
                    "en/c4-train.00000-of-01024.json.gz",
                    "en/c4-train.00001-of-01024.json.gz"],
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            },
        )
        _wikitext_dataset_train = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train")
        _wikitext_dataset_eval = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test")
        # Hacks to be consistent with other works' preprocessing.
        wikitext_dataset_train = datasets.Dataset.from_dict(
            {
                "text": [
                    # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                    "\n\n".join(_wikitext_dataset_train["text"])
                ],
            },
        )
        wikitext_dataset_eval = datasets.Dataset.from_dict(
            {
                "text": [
                    # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                    "\n\n".join(_wikitext_dataset_eval["text"])
                ],
            },
        )
        # Hacks to get around the `remove_columns` to be used later.
        wikitext_dataset_train = (
            wikitext_dataset_train  # type: ignore
            .add_column(
                name="timestamp",
                column=[None for _ in range(len(wikitext_dataset_train["text"]))])
            .add_column(
                name="url",
                column=wikitext_dataset_train["text"])
        )
        wikitext_dataset_eval = (
            wikitext_dataset_eval  # type: ignore
            .add_column(
                name="timestamp",
                column=wikitext_dataset_eval["text"])
            .add_column(
                name="url",
                column=wikitext_dataset_eval["text"])
        )
        raw_datasets["train"] = concatenate_datasets([
            raw_datasets["train"],
            wikitext_dataset_train])
        raw_datasets["wikitext"] = wikitext_dataset_eval
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        hf_quantization_kwargs = lora_utils.get_hf_quantization_config(
            method=model_args.hf_quantization_method,
            sequence_length=data_args.block_size)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            **hf_quantization_kwargs)
        # https://github.com/huggingface/transformers/pull/24906
        # if model.config.pretraining_tp != 1:
        #     raise NotImplementedError
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
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

    def init_base_model(model_args, config, hf_quantization_kwargs, training_args, torch_dtype, cpu=False):
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            **hf_quantization_kwargs)
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
        if cpu:
            pass
        else:
            model = model.to(training_args.device)
        return model

    def run_eval(eval_dataloader, model, accelerator):
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            batch = {k: v.to(training_args.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                # Hugging Face causal LM models return `loss` when labels are provided
                loss = outputs.loss
                # gather across processes (for multi-GPU training)
                loss = accelerator.gather(loss.unsqueeze(0).expand(len(batch['input_ids'])))
                losses.append(loss)

        losses = torch.cat(losses)
        try:
            perplexity = math.exp(torch.mean(losses).item())
        except OverflowError:
            perplexity = float("inf")

        return {"perplexity": perplexity}

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric_name = "perplexity"
        metric = evaluate.load(metric_name)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)
    if model_args.lora_config in ["lora", "lora-lpq"]:
        click.secho(f"LoRA Finetuning with `{model_args.lora_config}`", bg="yellow")
        if not all([
            model_args.lora_config is not None,
            model_args.lora_model_name is not None,
            model_args.hf_quantization_method is None]):
            raise ValueError

        model = lora_utils.prepare_model_for_lora(
            model=model,
            num_ranks=model_args.lora_num_ranks,
            lora_dropout=model_args.lora_dropout,
            use_gradient_checkpointing=training_args.gradient_checkpointing)
        lora_utils.transform_lora_layers(
            lpq=(model_args.lora_config == "lora-lpq"),
            model=model,
            model_name=model_args.lora_model_name,
            device="cuda")
    elif model_args.lora_config in ["lora-gptq"]:
        click.secho(f"GPTQ-LoRA Finetuning with `{model_args.lora_config}`", bg="yellow")
        if not all([
            model_args.lora_config is not None,
            model_args.hf_quantization_method is not None]):
            raise ValueError
        model = lora_utils.prepare_model_for_lora(
            model=model,
            num_ranks=model_args.lora_num_ranks,
            lora_dropout=model_args.lora_dropout,
            use_gradient_checkpointing=training_args.gradient_checkpointing)
    elif model_args.lora_config == 'lorashare':
        model_idx = [5, 15, 25, 35]
        Model_set = []
        base_path = data_args.config_data_dir
        for idx in model_idx:
            model = torch.load(os.path.join(base_path, f"model_state_dict_{idx}.pth"))
            Model_set.append(model)
        share_lora = lora_utils.extract_lora_weights(Model_set[0])
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=training_args.learning_rate
        )
        model_set = [model.to(training_args.device) for model in Model_set]
        optimizer_dict = {}
        optimizer_dict[0] = optimizer
    elif model_args.lora_config == 'lorasync':
        # pass
        # # for idx, budget in enumerate(np.arange(2.28, 6.53, 0.25)):
        # for idx, budget in enumerate(np.arange(2.25, 7.3, 0.1)):
        # # for idx, budget in enumerate(np.arange(6.55, 7.3, 0.05)):
        #     model = init_base_model(model_args, config, hf_quantization_kwargs, training_args, torch_dtype)
        #     model_plus_lora = lora_utils.prepare_model_for_lora(
        #         model=model,
        #         num_ranks=model_args.lora_num_ranks,
        #         lora_dropout=model_args.lora_dropout,
        #         use_gradient_checkpointing=training_args.gradient_checkpointing)
        #     lora_utils.transform_lora_layers(
        #         lpq=(model_args.lora_config == "lora-lpq"),
        #         model=model_plus_lora,
        #         model_name=model_args.lora_model_name,
        #         device="cuda",
        #         given_budget=budget,
        #         idx=idx)
        # exit()

        param_names = []
        param_shapes = {}
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ]
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.bias = None
            if any(t in name for t in target_modules):
                full_name = "base_model.model." + name
                param_names.append(full_name)
                num_params = sum(p.numel() for p in module.parameters())
                param_shapes[full_name] = num_params
    else:
        click.secho(f"Full Finetuning", bg="yellow")

    # This has no effects if we are not using DDP. But if we are, this
    # will patch the model to be compatible with DDP.
    distributed_utils.maybe_prepare_model_for_ddp(
        args=training_args,
        model=model)
    if data_args.dataset_name in ["c4", "c4-wiki-large"]:
        eval_dataset_dict = {
            "c4": eval_dataset,
            "wikitext": lm_datasets["wikitext"]}
        misc_utils.swarn(f"Using evaluation data: {eval_dataset_dict}")
    else:
        eval_dataset_dict = eval_dataset

    data_collator = default_data_collator
    if training_args.do_train:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
        )

    if training_args.do_eval:
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size)


    elif model_args.lora_config in ["lora-gptq"]:
        raise NotImplementedError
    else:
        click.secho(f"Full Finetuning", bg="yellow")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True
    training_args.warmup_steps = training_args.warmup_ratio * training_args.max_train_steps
    if model_args.lora_config != 'lorasync':
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=training_args.learning_rate
        )
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )
    elif model_args.lora_config == 'FullFinetuning':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_args.learning_rate
        )
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )
    else:
        values = [round(2.25 + 0.05 * i, 2) for i in range(101)]
        values = [v for v in values if v != 6.5]
        value_dict = {i: v for i, v in enumerate(values)}

        base_path = data_args.config_data_dir
        model = torch.load(os.path.join(base_path, f"model_state_dict_0.pth"))
        qconfig_dict = json.load(open(os.path.join(base_path, f"qconfig_dict_0.json")))
        model.add_module("quant_embedding",
                           lora_utils.QuantConfigEmbedding(device=training_args.device, r=model_args.lora_num_ranks, target_modules=target_modules))
        lora_utils.patch_lora_forward_with_quant_embedding(model, qconfig_dict, torch.tensor([2.25]).to(training_args.device), target_modules)
        model.to(training_args.device)
        trainable_modules = lora_utils.extract_lora_weights(model)

        optimizer = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if p.requires_grad],
            lr=training_args.learning_rate
        )
        # train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

        meta_data_numbers = list(range(0, 51, 1))
        meta_data_numbers_select = list(range(0, 51, 1))

        all_y = []
        dataset = QuantConfigDataset(data_args.config_data_dir, sample_list=meta_data_numbers)
        all_x = dataset.samples
        for _, model_idx in tqdm(enumerate(meta_data_numbers), desc="Evaluating models"):

            state_path = os.path.join(base_path, f"model_state_dict_{model_idx}.pth")
            qconfig_path = os.path.join(base_path, f"qconfig_dict_{model_idx}.json")

            q_model, qconfig_dict = lora_utils.load_quant_model(state_path, qconfig_path)

            q_model.add_module("quant_embedding", lora_utils.QuantConfigEmbedding(
                device=training_args.device,
                r=model_args.lora_num_ranks,
                target_modules=target_modules
            ))
            bit_tensor = torch.tensor([value_dict[model_idx]]).to(training_args.device)
            lora_utils.patch_lora_forward_with_quant_embedding(q_model, qconfig_dict, bit_tensor, target_modules)
            q_model.to(training_args.device)

            q_model.load_state_dict(trainable_modules, strict=False)
            q_model.eval()


            qconfig_obj_dict = {k: lora_utils.QuantConfig(**v) for k, v in qconfig_dict.items()}
            total_storage = 0.0
            model_total_params = 6476005376 #301989888
            for name, numel in param_shapes.items():
                match = re.search(r"layers\.(\d+)", name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx > 31:
                        qconfig = lora_utils.QuantConfig(
                            num_bits=8,
                            num_bits_0=8,
                            num_bits_1="bf16",
                            block_size_0=64,
                            block_size_1=256
                        )
                    else:
                        qconfig = qconfig_obj_dict[name]
                else:
                    qconfig = qconfig_obj_dict[name]
                cost = lora_utils.estimate_storage_from_config_numel(numel, qconfig)
                total_storage += cost / model_total_params

            if model_idx == 0:
                first_batch = next(iter(train_dataloader))
                first_batch = {k: v.to(training_args.device) for k, v in first_batch.items()}

            with torch.no_grad():
                outputs = q_model(**first_batch)
                loss = outputs.loss.item()

            all_y.append([-total_storage, -loss])

            del q_model
            torch.cuda.empty_cache()
            if data_args.use_bay:
                pass
            else:
                break
        all_y = torch.tensor(all_y, dtype=torch.float32).to(training_args.device)
        all_x = all_x.to(training_args.device)
        all_x_ref = [all_x]
        all_y_ref = [all_y]

        dummy_model = init_base_model(model_args, config, hf_quantization_kwargs, training_args, torch_dtype, True)
        dummy_model = dummy_model.cpu()
        for name, module in dummy_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.bias = None
        dummy_model_plus_lora = lora_utils.prepare_model_for_lora(
            model=dummy_model,
            num_ranks=model_args.lora_num_ranks,
            lora_dropout=model_args.lora_dropout)
        for name, param in dummy_model_plus_lora.named_parameters():
            dummy_model_plus_lora._parameters[name] = param.cpu()

        for name, buf in dummy_model_plus_lora.named_buffers():
            dummy_model_plus_lora._buffers[name] = buf.cpu()


    if overrode_max_train_steps:
        training_args.max_train_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)
    # Figure out how many steps we should save the Accelerator states
    training_args.checkpointing_steps = 'epoch'
    checkpointing_steps = training_args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
    # Train!
    total_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps

    from collections import defaultdict
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {training_args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {training_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_train_steps))
    completed_steps = 0
    starting_epoch = 0
    resume_step = None
    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint is not None or training_args.resume_from_checkpoint != "":
            # accelerator.print(f"Resumed from checkpoint: {training_args.resume_from_checkpoint}")
            # accelerator.load_state(training_args.resume_from_checkpoint)
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
    performace_dict = {}
    selection_counter = defaultdict(int)
    # tracker = eco2ai.Tracker(
    #     project_name=f"{model_args.lora_config}_finetune",
    #     experiment_description=f"Fine-tuning with {model_args.lora_config} at small model",
    #     file_name=f"./{training_args.output_dir}/emission.csv"
    # )
    if model_args.lora_config == 'lorasync':
        random.seed(training_args.seed)
        num_meta_model = [100]
    for epoch in range(starting_epoch, training_args.num_train_epochs):
    #     if training_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
    #         # We skip the first `n` batches in the dataloader when resuming from a checkpoint
    #         train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        total_loss = 0.0
        if model_args.lora_config in ["lora", "lora-lpq", "FullFinetuning", "lora-gptq"]:
            model.train()
            lora_utils.train_one_epoch(model, optimizer, train_dataloader, eval_dataloader, accelerator, training_args,
                                        data_args, metric, run_eval, logger, 'qnli', epoch, starting_epoch,
                                       resume_step, checkpointing_steps, performace_dict, total_loss, False, metric_name)
        elif model_args.lora_config == 'lorasync':
            lora_utils.train_one_epoch_lora_generate(train_dataloader, model, param_shapes, dummy_model_plus_lora, num_meta_model,
                                                                 eval_dataloader, accelerator, training_args, data_args,
                                                                 model_args, metric, trainable_modules, optimizer,
                                                                 run_eval, logger, 'qnli', epoch, starting_epoch, target_modules,
                                                                 resume_step, checkpointing_steps, performace_dict,
                                                                 total_loss, False, base_path, value_dict, meta_data_numbers, meta_data_numbers_select,
                                                                 all_x_ref, all_y_ref, metric_name, model_total_params, 1120)
        elif model_args.lora_config == 'lorashare':
            lora_utils.train_one_epoch_lora_shared_with_tracking(model_set, share_lora, train_dataloader, eval_dataloader, accelerator, training_args, data_args, metric,
                                                      run_eval, logger, 'qnli', epoch, starting_epoch, optimizer_dict,
                                                      resume_step, checkpointing_steps, performace_dict, total_loss, False, metric_name)

        if model_args.lora_config != 'lorasync':
            # pure_model = accelerator.unwrap_model(model)
            # state_dict = pure_model.state_dict()
            if model_args.lora_config == "lora-gptq":
                save_dict = {
                    k: v.cpu() for k, v in lora_utils.extract_lora_weights(model).items()
                }
                torch.save(save_dict, training_args.output_dir + "model.pth")
            else:
                torch.save(model, training_args.output_dir + "model.pth")
        #         # save_file(state_dict, output_dir + "/model.safetensors")
        #         # print("Saved checkpoint in ", output_dir + "/model.safetensors")
        #     accelerator.wait_for_everyone()

def _evaluation_post_processing(
    prefix: str,
    metrics: Dict[str, Any],
    eval_dataset: datasets.Dataset,
) -> None:

    metrics[f"eval_{prefix}samples"] = len(eval_dataset)
    try:
        perplexity = math.exp(metrics[f"eval_{prefix}loss"])
    except OverflowError:
        perplexity = float("inf")

    metrics[f"{prefix}perplexity"] = perplexity


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
