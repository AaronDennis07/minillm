import math
import time

import torch
import tiktoken
from datasets import load_dataset

from model import MiniLLM


# ==========================================
# Configuration
# ==========================================

batch_size = 64
max_seq_len = 512

embed_dim = 768
num_heads = 12
num_layers = 12

learning_rate = 6e-4
min_lr = 6e-5

max_iters = 10000
warmup_iters = 500
lr_decay_iters = 10000

eval_interval = 500

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_float32_matmul_precision("high")


# ==========================================
# Tokenizer and Dataset
# ==========================================

print("Loading tokenizer and TinyStories...")

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train",
)

print("Tokenizing data...")

full_text = "\n<|endoftext|>\n".join(
    dataset["text"][:250000]
)

data = torch.tensor(
    enc.encode(
        full_text,
        allowed_special={"<|endoftext|>"},
    ),
    dtype=torch.long,
)

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data_split = (
        train_data
        if split == "train"
        else val_data
    )

    ix = torch.randint(
        len(data_split) - max_seq_len,
        (batch_size,),
    )

    x = torch.stack(
        [
            data_split[
                i : i + max_seq_len
            ]
            for i in ix
        ]
    )

    y = torch.stack(
        [
            data_split[
                i + 1 : i + max_seq_len + 1
            ]
            for i in ix
        ]
    )

    return x.to(device), y.to(device)


# ==========================================
# Learning-rate schedule
# ==========================================

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters

    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (
        it - warmup_iters
    ) / (
        lr_decay_iters - warmup_iters
    )

    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )

    return min_lr + coeff * (
        learning_rate - min_lr
    )


# ==========================================
# Model
# ==========================================

model = MiniLLM(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
).to(device)

optimizer_kwargs = {
    "lr": learning_rate,
}

if device == "cuda":
    optimizer_kwargs["fused"] = True

optimizer = torch.optim.AdamW(
    model.parameters(),
    **optimizer_kwargs,
)

print(
    f"Model parameters: "
    f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
)

print(f"Training on: {device.upper()}")
print("Starting training...")

t0 = time.time()

for iteration in range(max_iters):
    lr = get_lr(iteration)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Validation
    if (
        iteration % eval_interval == 0
        or iteration == max_iters - 1
    ):
        model.eval()

        with torch.no_grad():
            x_val, y_val = get_batch("val")

            if device == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    _, val_loss = model(
                        x_val,
                        y_val,
                    )
            else:
                _, val_loss = model(
                    x_val,
                    y_val,
                )

        print(
            f"Step {iteration} | "
            f"Validation Loss: {val_loss.item():.4f} | "
            f"LR: {lr:.4e}"
        )

        model.train()

    # Training batch
    xb, yb = get_batch("train")

    if device == "cuda":
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            _, loss = model(xb, yb)
    else:
        _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0,
    )

    optimizer.step()

    if iteration % 100 == 0:
        elapsed = time.time() - t0

        print(
            f"Iter {iteration} | "
            f"Loss: {loss.item():.4f} | "
            f"Time/100 steps: {elapsed:.2f}s | "
            f"LR: {lr:.4e}"
        )

        t0 = time.time()


# ==========================================
# Save checkpoint
# ==========================================

checkpoint = "tiny_stories_gpt2.pth"

torch.save(
    model.state_dict(),
    checkpoint,
)

print(
    f"\nModel saved to '{checkpoint}'"
)


# ==========================================
# Generate a sample
# ==========================================

print("\n--- Training Complete ---\n")

prompt = (
    "Once upon a time, "
    "there was a little dog named"
)

context = torch.tensor(
    [enc.encode(prompt)],
    dtype=torch.long,
    device=device,
)

generated_indices = model.generate(
    context,
    max_new_tokens=250,
    temperature=0.7,
    top_k=40,
)

print(
    enc.decode(
        generated_indices[0].tolist()
    )
)
