# Add the training API since this is just standard T5 summarization training in the docs
# just use the longt5 example / docs


from src.annotate import ANNOTATION_TOKENS
from datasets import load_dataset, Dataset
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import evaluate
from tqdm.auto import tqdm
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

CKPT_BASE = "allenai/led-{}-16384"

MAX_INPUT_LENGTH = 16384  # ideally 16384 maximum token input size 8192
MAX_OUTPUT_LENGTH = 512  # idealy 1024
BATCH_SIZE = 2
EPOCHS = 1

rouge = evaluate.load('rouge')


def process_dataset(dataset):
    train_df = dataset[dataset['split'] == 'train']
    val_df = dataset[dataset['split'] == 'val']
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    return train_dataset, val_dataset


def train(
    dataset,
    model_size,
    dataset_name,
    variation='baseline',
    batch_size=BATCH_SIZE,
    num_epochs=EPOCHS,
    data_key='article'
):
    tokenizer = AutoTokenizer.from_pretrained(
        CKPT_BASE.format(model_size)
    )
    if variation in ['annotated', 'annotated_original']:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(ANNOTATION_TOKENS.values())})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # avoid issue with position embeddings for prompts in conditional generation
        tokenizer.padding_side = 'left'

    model = AutoModelForSeq2SeqLM.from_pretrained(
        CKPT_BASE.format(model_size)
    )

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(
            batch[data_key],
            padding="max_length",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )
        outputs = tokenizer(
            batch["lay_summary"],
            padding="max_length",
            truncation=True,
            max_length=MAX_OUTPUT_LENGTH,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        # create 0 global_attention_mask lists
        batch["global_attention_mask"] = len(batch["input_ids"]) * [
            [0 for _ in range(len(batch["input_ids"][0]))]
        ]

        # since above lists are references, the following line changes the 0 index for all samples
        # only place global attention on the first token
        batch["global_attention_mask"][0][0] = 1

        batch["labels"] = outputs.input_ids

        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(
            labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(
            predictions=pred_str, references=label_str
        )

        return rouge_output
    train_dataset, val_dataset = process_dataset(dataset)
    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=train_dataset.column_names,
    )

    val_dataset = val_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=val_dataset.column_names,
    )

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask",
                 "global_attention_mask", "labels"],
    )
    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask",
                 "global_attention_mask", "labels"],
    )

    model_name = CKPT_BASE.format(model_size).replace('google/', '')
    training_args = Seq2SeqTrainingArguments(
        f"{model_name}-biolaysum-{dataset_name}-{variation}",
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=2,
        # gradient_accumulation_steps=4,
        num_train_epochs=num_epochs,
        hub_token="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",
        #    fp16=True,
        push_to_hub=True,
    )

    # set generate hyperparameters
    model.config.num_beams = 4
    model.config.max_length = MAX_INPUT_LENGTH + MAX_OUTPUT_LENGTH
    model.config.min_length = 100
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.evaluate()
    trainer.push_to_hub()
