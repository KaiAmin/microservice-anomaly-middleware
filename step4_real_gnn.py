"""
Step 4 (bis) : Couche Topologique — vrai GNN entraîné (GAT codé à la main)

Contrairement à step4_graph_propagation.py (règle de diffusion à la main), ce
script entraîne un vrai Graph Attention Network (GAT) qui apprend à PRÉDIRE
quels services sont "sous stress" dans une trace donnée, à partir :
  - de la structure du graphe de dépendance des services (Gateway -> Order ->
    Payment -> Inventory -> Shipping)
  - de features par service/trace : score de surprise sémantique du LogBERT
    (step3_real_logbert.py) + signaux d'erreur/retry observés dans les logs

LIMITE CENTRALE, À NE PAS CACHER : pour entraîner ce GNN, il faut des labels
"quel service a été stressé, dans quelle trace". Ces labels n'existent dans
QUASIMENT AUCUN dataset public de logs (même DeepTraLog n'étiquette que la
trace entière comme anormale, pas le chemin de cause à effet). Ici, on peut
entraîner un vrai GNN UNIQUEMENT parce qu'on est nous-mêmes les générateurs
des données synthétiques (step1_generate_logs.py) : on CONNAÎT la vérité
terrain injectée, lue depuis stress_labels.json (deux types d'anomalies
désormais : "cascade" -> Inventory+Shipping stressés, "contained" -> Order
seul stressé, pas de propagation). C'est exactement le verrou scientifique
identifié : en conditions réelles, ce label n'est disponible que via des
campagnes de chaos engineering / injection de fautes contrôlées avec
traçage de l'origine exacte de la panne — ce qui n'existe pas "sur étagère".
Ce script prouve la FAISABILITÉ TECHNIQUE (un GAT entraîné, rapide, qui
généralise sur un split train/test), pas que le problème des labels est
résolu.
"""
import json
import random
import time

import torch
import torch.nn as nn

from step2_parse_and_group import parse_and_group
from step3_real_logbert import (
    PAD_ID, MASK_ID, LogBERT, build_vocab, to_ids, pad_sequences, train_logbert,
)

random.seed(42)
torch.manual_seed(42)

SERVICES = ["Gateway", "Order", "Payment", "Inventory", "Shipping"]
SVC_IDX = {s: i for i, s in enumerate(SERVICES)}
N_NODES = len(SERVICES)

N_EPOCHS_GNN = 150
TRAIN_RATIO = 0.8


def build_adjacency():
    """Graphe de dépendance causal (Gateway->...->Shipping), SENS UNIQUE aval
    (adj[receveur, émetteur]) + self-loops.

    Historique : la première version utilisait des arêtes bidirectionnelles
    (le GAT recevait aussi des messages en sens inverse, amont<-aval). Ça
    causait une fuite de stress : le stress d'Order (panne "contenue")
    remontait vers Gateway par message-passing, alors que Gateway n'est
    jamais réellement affecté. En sens unique (aval uniquement, cohérent
    avec la direction réelle de propagation d'une panne), un nœud ne peut
    plus "recevoir" le stress de son prédécesseur causal — seulement le
    transmettre à son successeur."""
    adj = torch.zeros(N_NODES, N_NODES, dtype=torch.bool)
    causal_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]  # Gateway->Order->Payment->Inventory->Shipping
    for a, b in causal_edges:
        adj[b, a] = True  # b (aval) reçoit de a (amont) — sens causal uniquement
    for i in range(N_NODES):
        adj[i, i] = True
    return adj


@torch.no_grad()
def per_position_surprise(model, logs, vocab, max_len):
    """Recalcule, position par position (leave-one-out masking), la surprise du
    LogBERT déjà entraîné — reprend le principe de score_trace() de step3_real_logbert.py
    mais retourne le détail par position au lieu de la seule moyenne."""
    ids = to_ids(logs, vocab)
    n = len(ids)
    surprises = []
    for i in range(n):
        masked = list(ids)
        true_id = masked[i]
        masked[i] = MASK_ID
        x, pad_mask = pad_sequences([masked], PAD_ID, max_len)
        logits = model(x, pad_mask)[0, i]
        probs = torch.softmax(logits, dim=-1)
        surprises.append(-torch.log(probs[true_id] + 1e-12).item())
    return surprises


def build_node_features(traces, labels, stress_labels, bert_model, vocab, max_len):
    """Pour chaque trace, calcule un vecteur de features par service :
    [surprise sémantique moyenne, nb de logs ERROR/Retry, nb de logs] et le label
    binaire "stressé" (vérité terrain connue car on a généré les données, lue
    depuis stress_labels.json — varie selon le type d'anomalie de la trace)."""
    features, node_labels = {}, {}
    for tid, logs in traces.items():
        surprises = per_position_surprise(bert_model, logs, vocab, max_len)
        per_svc = {s: {"surprise": [], "errors": 0, "count": 0} for s in SERVICES}
        for log, surprise in zip(logs, surprises):
            svc = log["service"]
            per_svc[svc]["surprise"].append(surprise)
            per_svc[svc]["count"] += 1
            if log["level"] == "ERROR":
                per_svc[svc]["errors"] += 1

        feat = torch.zeros(N_NODES, 3)
        lab = torch.zeros(N_NODES)
        stressed = set(stress_labels.get(tid, []))
        for svc, idx in SVC_IDX.items():
            s = per_svc[svc]["surprise"]
            feat[idx, 0] = sum(s) / len(s) if s else 0.0
            feat[idx, 1] = per_svc[svc]["errors"]
            feat[idx, 2] = per_svc[svc]["count"]
            if svc in stressed:
                lab[idx] = 1.0
        features[tid] = feat
        node_labels[tid] = lab
    return features, node_labels


class GATLayer(nn.Module):
    """Graph Attention Layer codé à la main (graphe minuscule à 5 noeuds -> pas
    besoin de torch_geometric). e_ij = attention(Wh_i, Wh_j), alpha = softmax(e),
    h_i' = sum_j alpha_ij * Wh_j."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        Wh = self.W(h)  # (N, out_dim)
        N = Wh.size(0)
        Wh_i = Wh.unsqueeze(1).expand(N, N, -1)
        Wh_j = Wh.unsqueeze(0).expand(N, N, -1)
        e = self.leaky(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))
        e = e.masked_fill(~adj, float("-inf"))
        alpha = torch.softmax(e, dim=1)
        return torch.matmul(alpha, Wh)


class StressGAT(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=16):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden_dim)
        self.gat2 = GATLayer(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.elu = nn.ELU()

    def forward(self, feat, adj):
        h = self.elu(self.gat1(feat, adj))
        h = self.elu(self.gat2(h, adj))
        return self.out(h).squeeze(-1)  # (N,) logits


def normalize_features(features, train_ids):
    stacked = torch.stack([features[tid] for tid in train_ids])
    mean = stacked.mean(dim=(0, 1))
    std = stacked.std(dim=(0, 1)) + 1e-6
    return {tid: (feat - mean) / std for tid, feat in features.items()}


def train_gnn(features, node_labels, train_ids, adj):
    model = StressGAT()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(N_EPOCHS_GNN):
        random.shuffle(train_ids)
        total_loss = 0.0
        for tid in train_ids:
            logits = model(features[tid], adj)
            loss = loss_fn(logits, node_labels[tid])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 25 == 0 or epoch == N_EPOCHS_GNN - 1:
            print(f"  epoch {epoch:3d}  loss={total_loss / len(train_ids):.4f}")

    model.eval()
    return model


@torch.no_grad()
def evaluate(model, features, node_labels, ids, adj):
    tp = fp = fn = tn = 0
    for tid in ids:
        logits = model(features[tid], adj)
        preds = (torch.sigmoid(logits) > 0.5).float()
        true = node_labels[tid]
        tp += ((preds == 1) & (true == 1)).sum().item()
        fp += ((preds == 1) & (true == 0)).sum().item()
        fn += ((preds == 0) & (true == 1)).sum().item()
        tn += ((preds == 0) & (true == 0)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall, (tp, fp, fn, tn)


if __name__ == "__main__":
    with open("raw_logs.json") as f:
        logs = json.load(f)
    with open("labels.json") as f:
        labels = json.load(f)
    with open("stress_labels.json") as f:
        stress_labels = json.load(f)

    traces = parse_and_group(logs)
    vocab = build_vocab(traces)
    max_len = max(len(v) for v in traces.values())
    adj = build_adjacency()

    print("Ré-entraînement du LogBERT (fournit les features sémantiques par service)...")
    bert_model = train_logbert(traces, labels, vocab, max_len)

    print("\nConstruction des features par service/trace (surprise LogBERT + erreurs/retry)...")
    features, node_labels = build_node_features(traces, labels, stress_labels, bert_model, vocab, max_len)

    all_ids = list(traces.keys())
    random.shuffle(all_ids)
    split = int(len(all_ids) * TRAIN_RATIO)
    train_ids, test_ids = all_ids[:split], all_ids[split:]
    n_anom_test = sum(labels[t] for t in test_ids)
    print(f"Split train/test : {len(train_ids)} traces train / {len(test_ids)} traces test "
          f"({n_anom_test} anormales dans le test set).")

    features = normalize_features(features, train_ids)

    print("\nEntraînement du GAT (supervisé, vérité terrain connue car données synthétiques) :")
    model = train_gnn(features, node_labels, train_ids, adj)
    n_params = sum(p.numel() for p in model.parameters())

    precision, recall, (tp, fp, fn, tn) = evaluate(model, features, node_labels, test_ids, adj)
    print(f"\nÉvaluation sur le test set ({len(test_ids)} traces, {N_NODES} noeuds/trace) :")
    print(f"  precision={precision:.3f}  recall={recall:.3f}  (TP={tp} FP={fp} FN={fn} TN={tn})")

    # exemple concret sur la trace la plus anormale du test set (si présente)
    anomalous_test = [t for t in test_ids if labels[t] == 1]
    if anomalous_test:
        tid = anomalous_test[0]
        with torch.no_grad():
            probs = torch.sigmoid(model(features[tid], adj))
        print(f"\nExemple : prédictions de stress du GAT pour la trace {tid} (ANORMALE, non vue à l'entraînement) :")
        for svc, idx in SVC_IDX.items():
            print(f"   {svc:>10} : P(stress) = {probs[idx]:.3f}")

    t0 = time.perf_counter()
    with torch.no_grad():
        for tid in test_ids:
            model(features[tid], adj)
    t1 = time.perf_counter()
    n_params_str = f"{n_params} paramètres"
    print(f"\nTemps de propagation (vrai forward-pass GAT, {n_params_str}) pour {len(test_ids)} traces : "
          f"{(t1 - t0) * 1000:.3f} ms ({(t1 - t0) * 1000 / len(test_ids):.4f} ms/trace en moyenne)")
