import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class Example:
    text: str
    label: str


def read_csv(path: str, text_col: str = "text", label_col: str = "label") -> List[Example]:
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns: '{text_col}' and '{label_col}'")
    df = df[[text_col, label_col]].dropna()
    return [Example(text=str(t), label=str(l)) for t, l in zip(df[text_col], df[label_col])]


def build_label_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    uniq = sorted(list(set(labels)))
    label2id = {l: i for i, l in enumerate(uniq)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def split_dataset(examples: List[Example], test_size: float, seed: int) -> Tuple[List[Example], List[Example]]:
    rnd = random.Random(seed)
    idx = list(range(len(examples)))
    rnd.shuffle(idx)
    n_test = max(1, int(len(idx) * test_size))
    test_idx = set(idx[:n_test])
    train = [examples[i] for i in range(len(examples)) if i not in test_idx]
    test = [examples[i] for i in range(len(examples)) if i in test_idx]
    return train, test


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def save_error_analysis(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id2label: Dict[int, str],
    out_path: str,
    top_k: int = 30,
):
    errors = []
    for t, yt, yp in zip(texts, y_true, y_pred):
        if yt != yp:
            errors.append(
                {
                    "text": t,
                    "true_label": id2label[int(yt)],
                    "pred_label": id2label[int(yp)],
                }
            )

    df_err = pd.DataFrame(errors)
    df_err.to_csv(out_path, index=False)

    # Print quick summary
    print("\n--- Error analysis (examples) ---")
    if len(errors) == 0:
        print("No misclassifications 🎉")
        return
    show = df_err.head(top_k)
    for i, row in show.iterrows():
        text_short = (row["text"][:160] + "...") if len(row["text"]) > 160 else row["text"]
        print(f"- TRUE={row['true_label']} | PRED={row['pred_label']} | text: {text_short}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to CSV with columns: text,label")
    ap.add_argument("--model", type=str, default="distilbert-base-uncased", help="HF model name")
    ap.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    examples = read_csv(args.data)
    if len(examples) < 10:
        print("⚠️ Your dataset is very small. This is OK for demo, but results will be unstable.")

    labels = [e.label for e in examples]
    label2id, id2label = build_label_maps(labels)

    train_ex, test_ex = split_dataset(examples, test_size=args.test_size, seed=args.seed)

    train_ds = Dataset.from_dict(
        {"text": [e.text for e in train_ex], "label": [label2id[e.label] for e in train_ex]}
    )
    test_ds = Dataset.from_dict(
        {"text": [e.text for e in test_ex], "label": [label2id[e.label] for e in test_ex]}
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    train_ds = train_ds.map(tok, batched=True)
    test_ds = test_ds.map(tok, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    # Predictions for detailed report
    preds = trainer.predict(test_ds)
    logits = preds.predictions
    y_true = preds.label_ids
    y_pred = np.argmax(logits, axis=1)

    print("\n--- Final metrics ---")
    print(metrics)

    print("\n--- Classification report ---")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[id2label[i] for i in range(len(id2label))],
            digits=4,
        )
    )

    print("\n--- Confusion matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Save artifacts
    model_dir = os.path.join(args.out_dir, "best_model")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Save error analysis CSV
    error_csv = os.path.join(args.out_dir, "error_analysis.csv")
    save_error_analysis(
        texts=[e.text for e in test_ex],
        y_true=y_true,
        y_pred=y_pred,
        id2label=id2label,
        out_path=error_csv,
        top_k=30,
    )

    print(f"\n✅ Saved model to: {model_dir}")
    print(f"✅ Saved error analysis to: {error_csv}")


if __name__ == "__main__":
    main()
