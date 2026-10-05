# Decoder-Only Transformer from Scratch

A GPT-2-style decoder-only Transformer implemented in PyTorch and trained from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

The model uses approximately 124M parameters and follows the core architecture of GPT-2 Small:

- 12 Transformer blocks
- 12 attention heads
- 768-dimensional embeddings
- GPT-2 BPE tokenizer through `tiktoken`
- Causal self-attention
- Pre-LayerNorm Transformer blocks
- GELU feed-forward networks
- Residual connections
- Weight tying between token embeddings and the language-model head
- AdamW optimization
- Linear learning-rate warmup
- Cosine learning-rate decay
- Gradient clipping
- bfloat16 autocasting on CUDA

## Project Structure

```text
decoder-only-transformer/
├── model.py
├── train.py
├── generate.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

Python 3.10+ is recommended.

Install the dependencies:

```bash
pip install -r requirements.txt
```

For CUDA training, install a PyTorch build compatible with your NVIDIA driver and CUDA environment.

## Training

The training script loads TinyStories, tokenizes the first 250,000 stories with the GPT-2 tokenizer, and creates a 90/10 training/validation split.

Run:

```bash
python train.py
```

The default configuration is:

```text
Batch size       : 64
Sequence length  : 512
Embedding size   : 768
Attention heads  : 12
Transformer layers: 12
Max iterations   : 10,000
Max learning rate: 6e-4
Min learning rate: 6e-5
Warmup steps     : 500
```

A CUDA-capable GPU is strongly recommended for this configuration. The original experiment was run in Google Colab using an NVIDIA A100 80 GB GPU.

After training, the checkpoint is saved as:

```text
tiny_stories_gpt2.pth
```

The checkpoint is intentionally excluded from Git through `.gitignore`.

## Generation

After training:

```bash
python generate.py
```

The default prompt is:

```text
Once upon a time, there was a clever little fox named
```

Generation supports:

- `temperature` — controls sampling randomness.
- `top_k` — restricts sampling to the top K candidate tokens.
- `max_new_tokens` — controls the number of generated tokens.

## How It Works

At a high level:

```text
Text
  ↓
GPT-2 Tokenizer
  ↓
Token IDs
  ↓
Token + Position Embeddings
  ↓
12 Transformer Blocks
  ↓
Final LayerNorm
  ↓
Vocabulary Logits
  ↓
Softmax
  ↓
Next-Token Sampling
  ↓
Append Token
  ↓
Repeat
```

During training, the model receives a sequence and learns to predict the next token at every position.

For example:

```text
Input : The capital of
Target: India
```

The model actually performs this prediction in parallel across all positions in a training sequence.

## Training Objective

The model minimizes next-token cross-entropy loss.

Conceptually:

```text
Input tokens
     ↓
Transformer
     ↓
Logits
     ↓
Cross-entropy loss
     ↓
Backpropagation
     ↓
AdamW
     ↓
Updated parameters
```

## Notes

This is an educational implementation intended to make the internals of a decoder-only language model easier to understand. It is not intended to reproduce the capabilities of a production-scale LLM.

Because the model is trained on a relatively small dataset for a limited number of iterations, generated text can show repetition, weak long-range coherence, and inconsistent story details.

## Blog Post

The accompanying Dev.to article explains the architecture, training pipeline, attention mechanism, tokenization, sampling, and generation process in more detail.

```text
https://dev.to/s_aarondennis_29b8169de9/demystifying-llms-building-a-124m-parameter-decoder-only-transformer-in-pytorch-5fn1
```

