"""
Step 3 (bis) : Couche Sémantique — vrai LogBERT (petit transformer entraîné from scratch)

Contrairement à step3_semantic_layer.py (proxy bigramme), ce script implémente
le VRAI principe du papier LogBERT (Guo et al., 2021) :
  - PAS un BERT pré-entraîné sur du texte naturel (le papier n'en utilise pas
    non plus : "BERT" ici désigne l'architecture Transformer bidirectionnelle,
    pas un modèle NLP pré-entraîné téléchargé).
  - Un petit Transformer encoder est entraîné FROM SCRATCH sur le vocabulaire
    des templates de logs de CE système, avec une tâche auto-supervisée de
    "Masked Log Key Prediction" (MLKP) : on masque aléatoirement des templates
    dans une trace et le modèle doit prédire le template masqué à partir du
    contexte bidirectionnel (comme le masked language modeling de BERT, mais
    appliqué à des templates de logs plutôt qu'à des mots).
  - L'entraînement se fait UNIQUEMENT sur les traces normales (non supervisé),
    comme dans le papier : le modèle apprend "à quoi ressemble une séquence
    normale", puis une trace est anormale si le modèle échoue à prédire ses
    templates à partir du contexte.

Limite honnête : nos traces normales synthétiques suivent TOUTES exactement
la même séquence de templates (voir step1_generate_logs.py) — il n'y a aucune
variabilité. La tâche de prédiction masquée est donc trivialement facile ici.
Sur un vrai corpus de logs de production, la séquence normale varie
(branches conditionnelles, retries bénins, ordres différents selon la charge),
et c'est LÀ que la capacité du transformer à apprendre un vrai modèle de
contexte (plutôt qu'un simple comptage de bigrammes) ferait la différence.
"""
import json
import random
import time

import torch
import torch.nn as nn

from step2_parse_and_group import parse_and_group

random.seed(42)
torch.manual_seed(42)

PAD_ID, MASK_ID = 0, 1
MASK_PROB = 0.15
N_EPOCHS = 300
TOP_G = 3  # une prédiction est jugée "correcte" si le vrai token est dans le top-g


class LogBERT(nn.Module):
    """Petit transformer encoder bidirectionnel, entraîné from scratch (pas de poids pré-entraînés)."""

    def __init__(self, vocab_size, max_len, d_model=32, nhead=4, num_layers=2, dim_ff=64):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x, pad_mask):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.out(h)


def build_vocab(traces):
    templates = sorted({l["template"] for logs in traces.values() for l in logs})
    vocab = {"<PAD>": PAD_ID, "<MASK>": MASK_ID}
    for t in templates:
        vocab[t] = len(vocab)
    return vocab


def to_ids(logs, vocab):
    return [vocab[l["template"]] for l in logs]


def pad_sequences(seqs, pad_value, max_len):
    padded = [seq + [pad_value] * (max_len - len(seq)) for seq in seqs]
    pad_mask = [[False] * len(seq) + [True] * (max_len - len(seq)) for seq in seqs]
    return torch.tensor(padded), torch.tensor(pad_mask)


def mask_batch(id_seqs, mask_prob=MASK_PROB):
    """Masquage dynamique : à chaque epoch, on masque un sous-ensemble différent de positions."""
    inputs, targets = [], []
    for seq in id_seqs:
        inp = list(seq)
        tgt = [-100] * len(seq)  # -100 = ignoré par la cross-entropy (positions non masquées)
        n_mask = max(1, round(len(seq) * mask_prob))
        for i in random.sample(range(len(seq)), n_mask):
            tgt[i] = seq[i]
            inp[i] = MASK_ID
        inputs.append(inp)
        targets.append(tgt)
    return inputs, targets


def train_logbert(traces, labels, vocab, max_len):
    normal_ids = [to_ids(logs, vocab) for tid, logs in traces.items() if labels[tid] == 0]

    model = LogBERT(vocab_size=len(vocab), max_len=max_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for epoch in range(N_EPOCHS):
        inputs, targets = mask_batch(normal_ids)
        x, pad_mask = pad_sequences(inputs, PAD_ID, max_len)
        y, _ = pad_sequences(targets, -100, max_len)

        logits = model(x, pad_mask)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == N_EPOCHS - 1:
            print(f"  epoch {epoch:3d}  loss={loss.item():.4f}")

    model.eval()
    return model


@torch.no_grad()
def score_trace(model, logs, vocab, max_len, top_g=TOP_G):
    """
    Score d'anomalie = moyenne de -log P(vrai template | contexte) sur toutes les
    positions de la trace, en masquant une position à la fois (leave-one-out),
    comme dans l'évaluation du papier LogBERT. On compte aussi le nombre de
    positions où le vrai template n'est PAS dans le top-g des prédictions du modèle.
    """
    ids = to_ids(logs, vocab)
    n = len(ids)
    surprises = []
    flagged = []
    for i in range(n):
        masked = list(ids)
        true_id = masked[i]
        masked[i] = MASK_ID
        x, pad_mask = pad_sequences([masked], PAD_ID, max_len)
        logits = model(x, pad_mask)[0, i]
        probs = torch.softmax(logits, dim=-1)
        surprises.append(-torch.log(probs[true_id] + 1e-12).item())

        topg = torch.topk(probs, top_g).indices.tolist()
        if true_id not in topg:
            id_to_template = {v: k for k, v in vocab.items()}
            template = id_to_template[true_id]
            flagged.append((i, template))

    avg_surprise = sum(surprises) / n if n else 0.0
    return avg_surprise, flagged


if __name__ == "__main__":
    with open("raw_logs.json") as f:
        logs = json.load(f)
    with open("labels.json") as f:
        labels = json.load(f)

    traces = parse_and_group(logs)
    vocab = build_vocab(traces)
    max_len = max(len(v) for v in traces.values())

    print(f"Vocabulaire de {len(vocab)} tokens (dont {len(vocab) - 2} templates de logs).")
    print(f"Longueur de trace max : {max_len}\n")

    print("Entraînement du LogBERT (transformer, masked log-key prediction, "
          "traces normales uniquement) :")
    model = train_logbert(traces, labels, vocab, max_len)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModèle entraîné : {n_params} paramètres (petit transformer, "
          f"pas de poids pré-entraînés téléchargés).\n")

    results = []
    t0 = time.perf_counter()
    for tid, tr_logs in traces.items():
        score, flagged = score_trace(model, tr_logs, vocab, max_len)
        results.append((tid, score, labels[tid], flagged))
    t1 = time.perf_counter()

    results.sort(key=lambda r: -r[1])

    print("Top 5 traces les plus 'surprenantes' selon le vrai LogBERT :")
    for tid, score, label, flagged in results[:5]:
        tag = "ANORMALE (vrai label)" if label == 1 else "normale (vrai label)"
        print(f"  {tid}  score={score:.3f}  [{tag}]")
        for pos, template in flagged:
            print(f"       position {pos} : vrai template hors du top-{TOP_G} prédit -> '{template}'")

    n_anom = sum(1 for _, _, l, _ in results if l == 1)
    top_n = results[:n_anom]
    tp = sum(1 for _, _, l, _ in top_n if l == 1)
    print(f"\nSur les {n_anom} traces réellement anormales, "
          f"{tp}/{n_anom} sont bien dans le top-{n_anom} des scores (rappel@k).")

    print(f"\nTemps de scoring pour les {len(traces)} traces (vrai forward-pass "
          f"transformer, {n_params} paramètres) : {(t1 - t0) * 1000:.3f} ms "
          f"({(t1 - t0) * 1000 / len(traces):.4f} ms/trace en moyenne)")

    with open("anomaly_scores_logbert.json", "w") as f:
        json.dump({tid: {"score": s, "label": l} for tid, s, l, _ in results}, f, indent=2)
