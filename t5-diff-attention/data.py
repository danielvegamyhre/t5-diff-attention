#!/usr/bin/env python3
from datasets.dataset_dict import DatasetDict
from transformers.models.t5.tokenization_t5 import T5Tokenizer

from enum import Enum

class DataSplit(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

def preprocess(examples: DatasetDict, tokenizer: T5Tokenizer, split: DataSplit):
    # prepare the inputs
    inputs = ["summarize: " + doc for doc in examples[split.value]["document"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # prepare the targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples[split.value]["summary"], max_length=150, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
