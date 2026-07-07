"""
Step 5 : Mesure de latence bout-en-bout PAR TRACE (simulation streaming),
pour comparer honnêtement au seuil visé de 50ms.

ATTENTION (résultat à interpréter avec prudence) : ce pipeline est fait de
règles simples (regex, dict lookup, diffusion sur un graphe à 5 nœuds).
Un vrai LogBERT (forward pass transformer) et un vrai GNN entraîné (GraphSAGE/GAT)
coûteraient plusieurs ordres de grandeur de plus en temps de calcul.
Ce benchmark valide donc l'ARCHITECTURE du pipeline, pas la latence réelle
d'un système en production.
"""
import json
import time
import statistics

from step2_parse_and_group import to_template
from step3_semantic_layer import train_transition_model, score_trace
from step4_graph_propagation import build_service_graph, propagate_stress
from step2_parse_and_group import parse_and_group


def process_one_trace(raw_trace_logs, probs, g):
    t0 = time.perf_counter()

    # 1. parsing + tri chronologique (déjà groupé ici par trace_id en entrée)
    for l in raw_trace_logs:
        l["template"] = to_template(l["message"])
    raw_trace_logs.sort(key=lambda x: x["timestamp"])

    # 2. scoring sémantique
    score, flagged = score_trace(raw_trace_logs, probs)

    # 3. propagation topologique (seed = 1er service impliqué dans une transition surprenante)
    seed_service = None
    if flagged:
        for l in raw_trace_logs:
            if l["template"] == flagged[0][0]:
                seed_service = l["service"]
                break
    propagated = {}
    if seed_service:
        propagated = propagate_stress(g, {seed_service: 1.0}, hops=3, decay=0.6)

    t1 = time.perf_counter()
    return (t1 - t0) * 1000, score, propagated


if __name__ == "__main__":
    with open("raw_logs.json") as f:
        logs = json.load(f)
    with open("labels.json") as f:
        labels = json.load(f)

    traces = parse_and_group(logs)
    probs = train_transition_model(traces, labels)
    g = build_service_graph(traces)

    latencies = []
    for tid, tr_logs in traces.items():
        lat, score, propagated = process_one_trace(list(tr_logs), probs, g)
        latencies.append(lat)

    print(f"Latence moyenne par trace   : {statistics.mean(latencies):.4f} ms")
    print(f"Latence médiane par trace   : {statistics.median(latencies):.4f} ms")
    print(f"Latence p95                 : {statistics.quantiles(latencies, n=100)[94]:.4f} ms")
    print(f"Latence max                 : {max(latencies):.4f} ms")
    print(f"\n-> Toutes largement < 50ms, MAIS ce pipeline ne contient ni forward-pass")
    print("   transformer ni forward-pass GNN entraîné : ce n'est pas représentatif")
    print("   de la latence d'inférence d'un vrai LogBERT+GNN en production.")
