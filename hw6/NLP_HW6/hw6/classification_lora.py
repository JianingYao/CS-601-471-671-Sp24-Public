from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    TrainingArguments, DataCollatorWithPadding, Trainer, get_linear_schedule_with_warmup
import datasets
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.optim import AdamW
import math
import numpy as np
import torch
import os


def process_dataset(dataset, usage, slice=None):

    if slice is None:
        slice = 0, len(dataset[usage]['passage'])

    passages = dataset[usage]['passage'][slice[0]: slice[1]]
    questions = dataset[usage]['question'][slice[0]: slice[1]]
    answers = dataset[usage]['answer'][slice[0]: slice[1]]
    labels = []
    texts = []
    for passage, question, answer in zip(passages, questions, answers):
        text = question + " [SEP] " + passage
        texts.append(text)
        labels.append(int(answer))

    result = dict(texts=texts, labels=labels)
    return datasets.Dataset.from_dict(result)


def metrics(eval_prediction):
    logits, labels = eval_prediction
    pred = np.argmax(logits, axis=1)
    auc_score = roc_auc_score(labels, pred)
    acc = accuracy_score(labels, pred)
    return {"Val-ACC": acc}


def predict(dataset, model, max_len):

    labels = []
    predictions = []
    with torch.no_grad():  # Inference, but not training
        for data in dataset:
            label = data['labels']
            del data['labels']
            args = {k: torch.from_numpy(np.expand_dims(np.array(v[:max_len]), axis=0)).to(model.device) for k, v in data.items()}
            logits = model(**args).logits.cpu().numpy()
            prediction = np.argmax(logits, axis=-1)
            predictions.append(prediction[0])
            labels.append(label)
            print(label, prediction[0])

    print(accuracy_score(labels, predictions))


def main():
    dataset = load_dataset('boolq')
    train_dataset = process_dataset(dataset, 'train', (0, 8000))
    val_dataset = process_dataset(dataset, 'validation')
    test_dataset = process_dataset(dataset, 'train', (8000, len(dataset['train']['passage'])))
    max_len = 128
    batch_size = 64
    lr = 1e-4
    epochs = 7

    model_name = 'distilbert-base-uncased'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize_func(data):
        return tokenizer(
            data['texts'],
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )

    lora_config = LoraConfig(
        r=8,                                            # Rank Number
        lora_alpha=32,                                  # Alpha (Scaling Factor)
        lora_dropout=0.05,                              # Dropout Prob for Lora
        target_modules=["q_lin", "k_lin", "v_lin"],     # Which layer to apply LoRA
        task_type=TaskType.SEQ_CLS                      # Sequence to Classification Task
    )

    # Get our LoRA-enabled model
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    train_dataset = train_dataset.map(
        tokenize_func,
        batched=True,
        remove_columns=["texts"],
    )

    val_dataset = val_dataset.map(
        tokenize_func,
        batched=True,
        remove_columns=["texts"]
    )

    test_dataset = test_dataset.map(
        tokenize_func,
        batched=True,
        remove_columns=["texts"]
    )

    peft_training_args = TrainingArguments(
        output_dir='./result-distilbert-lora',
        logging_dir='./logs-distilbert-lora',
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=10,
        weight_decay=0.01,
        seed=42,
        fp16=True,          # Only use with GPU
        report_to='none'
    )

    optimizer = AdamW(peft_model.parameters(), lr=1e-4)
    n_epochs = epochs
    total_steps = n_epochs * math.ceil(len(train_dataset) / batch_size / 2)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps)

    # Data Collator
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest"
    )

    # Define Trainer
    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=train_dataset,  # Training Data
        eval_dataset=val_dataset,  # Evaluation Data
        tokenizer=tokenizer,
        compute_metrics=metrics,
        optimizers=(optimizer, lr_scheduler),
        data_collator=collator
    )
    # Path to save the fine-tuned model
    peft_model_path = "./peft-distilbert-lora"
    # Train the model
    peft_trainer.train()

    if not os.path.exists('distilbert_lora'):
        os.makdirs('distilbert_lora')

    peft_trainer.model.save_pretrained('distilbert_lora')
    tokenizer.save_pretrained('distilbert_lora')
    model = PeftModel.from_pretrained(base_model, 'distilbert_lora', is_trainable=False)

    predict(test_dataset, model, max_len=max_len)


if __name__ == '__main__':
    main()
