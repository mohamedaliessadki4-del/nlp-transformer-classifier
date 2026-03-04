import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True, help="Path to saved model folder")
    ap.add_argument("--text", type=str, required=True, help="Text to classify")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_id = int(torch.argmax(probs).item())

    label = model.config.id2label[str(pred_id)] if isinstance(model.config.id2label, dict) else model.config.id2label[pred_id]
    conf = float(probs[pred_id].item())

    print(f"Prediction: {label} (confidence={conf:.4f})")


if __name__ == "__main__":
    main()
