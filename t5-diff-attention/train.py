import os
from datetime import datetime
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import wandb
import torch
from transformers import T5Tokenizer, TrainingArguments, Trainer
from datasets import load_dataset

from model import get_model
from data import preprocess, DataSplit

def train(args: Namespace):
    model = get_model(base="t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset = load_dataset("xsum")
    train_data, eval_data = preprocess(dataset, tokenizer, DataSplit.TRAIN), preprocess(dataset, tokenizer, DataSplit.VALIDATION)
    num_devices = min(1, torch.cuda.device_count())
    per_device_batch_size = args.batch_size // num_devices

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        num_train_epochs=args.epoch,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
    )
    trainer.train()

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--learning-rate", type=float, default=1e-3)
    argparser.add_argument("--batch-size", type=int, default=32) 
    args = argparser.parse_args()
    train(args)