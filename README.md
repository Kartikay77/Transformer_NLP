# Transformer-based Sequence Learning: Letter Counting

## 📌 Project Overview
This project implements a Transformer-based sequence learning model in PyTorch for a letter counting task.
The task: given a sequence of characters, the model predicts whether each character has appeared 0, 1, or ≥2 times before in the sequence.
This demonstrates how attention-based architectures like Transformers can be applied to simple symbolic sequence tasks, serving as an educational introduction to self-attention, embeddings, and sequence modeling
![b](https://github.com/Kartikay77/Transformer_NLP/blob/main/Transformer1.png)
![a](https://github.com/Kartikay77/Transformer_NLP/blob/main/Transformer2.png)

## 🚀 Features
Custom Transformer Implementation (no reliance on HuggingFace/torch.nn.Transformer).
Self-Attention + Positional Encoding built from scratch.
Training & Evaluation pipelines with accuracy tracking.
Letter Counting Task with sequence-to-sequence predictions.
Supports multiple tasks (BEFORE, BEFOREAFTER) defined in letter_counting.py.
Visualization of attention maps (optional).
Achieved ~97% accuracy on test data.

## ⚙️ Installation
## Clone the repo:
git clone https://github.com/Kartikay77/Transformer_NLP.git
cd Transformer_NLP

## Create a virtual environment and install dependencies:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Minimal requirements:
torch
numpy
matplotlib
scikit-learn


## ▶️ Usage
## Train & Evaluate (BeforeAfter Task)
python letter_counting.py --task BEFOREAFTER

## Train & Evaluate (Before Task)
python letter_counting.py --task BEFORE

## 📊 Example Output
epoch 0: train loss 0.6342521639947295
epoch 1: train loss 0.4171089551694691
epoch 2: train loss 0.28187534732371566
epoch 3: train loss 0.2060582914231887
epoch 4: train loss 0.15030208903034217

INPUT 0: heir average albedo
GOLD 0: [0, 0, 0, 0, 0, 1, 1, 1, ...]
PRED 0: [0, 0, 0, 0, 0, 1, 1, 1, ...]

Training accuracy (100 exs): 0.951000
Dev accuracy (whole set):   0.952100

## 🧠 Key Learnings
Implementation of multi-head self-attention from scratch.
Using positional encodings to capture sequence order.
Training on symbolic sequences rather than natural language.
Visualizing attention distributions to understand model behavior.

## 📌 Future Work
Extend to multi-head attention.
Compare performance with LSTM/GRU baselines.
Apply model to real NLP tasks (POS tagging, NER, etc.).

