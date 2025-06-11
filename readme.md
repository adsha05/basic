Transformer From Scratch

This project is a ground-up implementation of the Transformer architecture, based on the seminal paper “Attention is All You Need” by Vaswani et al. The aim is to demystify the inner workings of modern language models by reconstructing each core component using PyTorch — without relying on high-level deep learning libraries like HuggingFace or TensorFlow Hub. This makes it an excellent educational resource for anyone looking to understand how large language models like BERT, GPT, and T5 operate under the hood.

# Project Purpose

Transformers have revolutionized the field of Natural Language Processing (NLP), forming the backbone of today’s state-of-the-art models. However, understanding them deeply requires more than just using pre-trained models. This project is designed to bridge that gap by reimplementing the architecture from scratch, giving learners full visibility into:

Scaled dot-product attention and multi-head attention
Positional encoding to handle word order
Layer normalization and residual connections
The encoder-decoder structure
Masking mechanisms for autoregressive tasks
Each module is built in a readable and modular fashion so that others can follow along, modify components, and experiment with new ideas.

# What’s Implemented

Token Embedding Layer: Converts input tokens into dense vectors
Positional Encoding: Injects position information using sinusoidal functions
Multi-Head Attention: Implements parallel attention heads with proper scaling and masking
Feed Forward Layer: Two-layer MLP with ReLU and dropout
Layer Normalization & Residuals: Ensures better gradient flow and training stability
Encoder Stack: A repeated stack of multi-head attention + feed-forward blocks
Decoder Stack: Similar to encoder but with masked attention for autoregressive generation
Training Loop: Minimal training framework with dummy data support
All components are implemented from first principles using only PyTorch (no external transformers libraries), making this repository ideal for educational use or deep debugging practice.

# Getting Started

To run the model:

cd transformer-from-scratch
pip install -r requirements.txt
python train.py
Dummy input/output examples are provided in the examples/ folder. Training scripts are simple and designed to be extended — for example, to integrate a tokenizer or connect to real NLP datasets such as WMT or WikiText.

# Planned Extensions

Add tokenizers (WordPiece or BPE)
Support for translation datasets (e.g., English → German)
Mini GPT-style decoder-only version
Inference scripts for text generation
