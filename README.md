# NLP Transformer Classifier (BERT)

Text classification project using Transformer models (DistilBERT/BERT) with:
- **Evaluation**: Accuracy, Macro-F1, Weighted-F1
- **Reports**: classification report + confusion matrix
- **Error analysis**: export of misclassified examples for debugging

## Features
- Fine-tuning a Transformer for text classification
- Metrics and detailed evaluation
- Error analysis (misclassified samples saved to CSV)

## Setup
```bash
pip install -r requirements.txt

python src/train.py --data data/sample.csv --epochs 3 --out_dir outputs

python src/predict.py --model_dir outputs/best_model --text "This is a great product!"
