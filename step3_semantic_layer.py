"""
Step 3 : Couche Sémantique (proxy pour LogBERT)

IMPORTANT (limite honnête) : cet environnement sandbox n'a pas d'accès réseau
à huggingface.co, donc impossible de télécharger un vrai BERT pré-entraîné ici.
On simule le PRINCIPE de LogBERT avec un modèle de transitions (bigrammes de
templates), dans l'esprit de DeepLog : on apprend "quelles suites de logs sont
normales" à partir des traces normales, puis on score une nouvelle trace par
la probabilité de ses transitions. C'est un proxy pédagogique, PAS un LogBERT
réel — un vrai LogBERT capturerait aussi la sémantique du texte (embeddings),
pas seulement l'ordre des templates.
"""
import json
import time
from collections import defaultdict, Counter

from step2_parse_and_group import parse_and_group


def train_transition_model(traces, labels):
    """Apprend les transitions (template_i -> template_i+1) sur les traces normales uniquement."""
    transitions = defaultdict(Counter)
    for tid, logs in traces.items():
        if labels[tid] != 0:
            continue  # on n'apprend QUE sur les traces normales (non supervisé)
        templates = [l["template"] for l in logs]
        for a, b in zip(templates, templates[1:]):
            transitions[a][b] += 1

    # normaliser en probabilités
    probs = {}
    for a, counter in transitions.items():
        total = sum(counter.values())
        probs[a] = {b: c / total for b, c in counter.items()}
    return probs


def score_trace(logs, probs, epsilon=0.01):
    """Score d'anomalie = -log-vraisemblance moyenne des transitions observées."""
    templates = [l["template"] for l in logs]
    surprises = []
    flagged_transitions = []
    for a, b in zip(templates, templates[1:]):
        p = probs.get(a, {}).get(b, epsilon)  # transition jamais vue -> improbable
        surprises.append(-1 * (p and __import__("math").log(p)))
        if p <= epsilon:
            flagged_transitions.append((a, b))
    avg_surprise = sum(surprises) / len(surprises) if surprises else 0.0
    return avg_surprise, flagged_transitions


if __name__ == "__main__":
    with open("raw_logs.json") as f:
        logs = json.load(f)
    with open("labels.json") as f:
        labels = json.load(f)

    traces = parse_and_group(logs)
    probs = train_transition_model(traces, labels)

    print("Modèle de transitions appris sur traces normales.\n")

    results = []
    t0 = time.perf_counter()
    for tid, tr_logs in traces.items():
        score, flagged = score_trace(tr_logs, probs)
        results.append((tid, score, labels[tid], flagged))
    t1 = time.perf_counter()

    results.sort(key=lambda r: -r[1])

    print("Top 5 traces les plus 'surprenantes' selon le modèle (score de surprise) :")
    for tid, score, label, flagged in results[:5]:
        tag = "ANORMALE (vrai label)" if label == 1 else "normale (vrai label)"
        print(f"  {tid}  score={score:.3f}  [{tag}]")
        for a, b in flagged:
            print(f"       transition jamais vue : '{a}'  ->  '{b}'")

    # petite évaluation
    n_anom = sum(1 for _, _, l, _ in results if l == 1)
    top_n = results[:n_anom]
    tp = sum(1 for _, _, l, _ in top_n if l == 1)
    print(f"\nSur les {n_anom} traces réellement anormales, "
          f"{tp}/{n_anom} sont bien dans le top-{n_anom} des scores (rappel@k).")

    print(f"\nTemps de scoring pour les {len(traces)} traces : {(t1 - t0) * 1000:.3f} ms "
          f"({(t1 - t0) * 1000 / len(traces):.4f} ms/trace en moyenne)")

    with open("anomaly_scores.json", "w") as f:
        json.dump({tid: {"score": s, "label": l} for tid, s, l, _ in results}, f, indent=2)
