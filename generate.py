import sys

import torch
import tiktoken

from model import MiniLLM


# ==========================================
# Configuration
# ==========================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

max_seq_len = 512
embed_dim = 768
num_heads = 12
num_layers = 12

checkpoint = "tiny_stories_gpt2.pth"


# ==========================================
# Tokenizer
# ==========================================

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab


# ==========================================
# Load model
# ==========================================

model = MiniLLM(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
).to(device)

try:
    state_dict = torch.load(
        checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print(
        f"Model loaded from '{checkpoint}'"
    )

except Exception as exc:
    print(
        f"Failed to load model: {exc}"
    )
    sys.exit(1)


# ==========================================
# Generation
# ==========================================

@torch.no_grad()
def generate_text(
    prompt,
    max_new_tokens=150,
    temperature=0.8,
    top_k=40,
):
    print("\n--- Output ---")
    print(prompt, end="", flush=True)

    idx = torch.tensor(
        [enc.encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -max_seq_len:]

        if device == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                logits, _ = model(idx_cond)
        else:
            logits, _ = model(idx_cond)

        logits = logits[:, -1, :]

        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(
                logits,
                min(top_k, logits.size(-1)),
            )

            logits[
                logits < v[:, [-1]]
            ] = -float("Inf")

        probs = torch.softmax(
            logits,
            dim=-1,
        )

        idx_next = torch.multinomial(
            probs,
            num_samples=1,
        )

        idx = torch.cat(
            (idx, idx_next),
            dim=1,
        )

        print(
            enc.decode(
                [idx_next.item()]
            ),
            end="",
            flush=True,
        )

    print("\n")


if __name__ == "__main__":
    generate_text(
        "Once upon a time, "
        "there was a clever little fox named",
        max_new_tokens=150,
        temperature=0.8,
        top_k=40,
    )
