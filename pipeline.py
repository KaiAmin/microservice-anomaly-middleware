"""
Pipeline bout-en-bout automatisée.

Enchaîne les 4 couches de l'architecture (Ingestion/trace-grouping -> couche
sémantique LogBERT -> couche topologique GAT -> mesure de latence temps réel)
et écrit un résultat structuré par trace dans output/middleware.log, au format
JSON lines. Ce fichier est celui que Promtail (docker-compose.yml) lit et
pousse vers Loki, pour visualisation dans le dashboard Grafana préconfiguré
(architecture fidèle au paper : ... -> Loki/Grafana).

Usage :
  python3 pipeline.py                # génère tout, écrit les résultats d'un coup
  python3 pipeline.py --live         # écrit un résultat toutes les ~0.3s (effet démo live)
  python3 pipeline.py --keep-data    # réutilise raw_logs.json/labels.json existants
"""
import argparse
import json
import os
import random
import time

import torch

from step1_generate_logs import build_dataset
from step2_parse_and_group import parse_and_group
from step3_real_logbert import build_vocab, train_logbert, score_trace as bert_score_trace
from step4_real_gnn import (
    SVC_IDX, build_adjacency, build_node_features, normalize_features, train_gnn,
)

OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "middleware.log")


def run_pipeline(regenerate_data=True, live=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if regenerate_data or not os.path.exists("raw_logs.json"):
        print("[0/4] Génération des logs synthétiques...")
        logs, labels, stress_labels, anomaly_types = build_dataset()
        with open("raw_logs.json", "w") as f:
            json.dump(logs, f)
        with open("labels.json", "w") as f:
            json.dump(labels, f)
        with open("stress_labels.json", "w") as f:
            json.dump(stress_labels, f)
        with open("anomaly_types.json", "w") as f:
            json.dump(anomaly_types, f)
    else:
        with open("raw_logs.json") as f:
            logs = json.load(f)
        with open("labels.json") as f:
            labels = json.load(f)
        with open("stress_labels.json") as f:
            stress_labels = json.load(f)
        with open("anomaly_types.json") as f:
            anomaly_types = json.load(f)

    print("[1/4] Ingestion + trace-grouping...")
    traces = parse_and_group(logs)
    print(f"      {len(traces)} traces reconstruites.")

    print("[2/4] Entraînement du LogBERT (couche sémantique, transformer from scratch)...")
    vocab = build_vocab(traces)
    max_len = max(len(v) for v in traces.values())
    bert_model = train_logbert(traces, labels, vocab, max_len)

    print("[3/4] Entraînement du GAT (couche topologique, Graph Attention Network)...")
    adj = build_adjacency()
    features, node_labels = build_node_features(traces, labels, stress_labels, bert_model, vocab, max_len)
    all_ids = list(traces.keys())
    random.shuffle(all_ids)
    split = int(len(all_ids) * 0.8)
    train_ids = all_ids[:split]
    features = normalize_features(features, train_ids)
    gnn_model = train_gnn(features, node_labels, train_ids, adj)

    mode = "LIVE (streaming ~0.3s/trace)" if live else "batch (tout d'un coup)"
    print(f"[4/4] Scoring temps réel + écriture -> {OUTPUT_FILE}  [mode: {mode}]")

    with open(OUTPUT_FILE, "w") as out:
        for tid, tr_logs in traces.items():
            t0 = time.perf_counter()
            score, flagged = bert_score_trace(bert_model, tr_logs, vocab, max_len)
            t1 = time.perf_counter()
            with torch.no_grad():
                stress_probs = torch.sigmoid(gnn_model(features[tid], adj))
            t2 = time.perf_counter()

            semantic_latency_ms = (t1 - t0) * 1000
            topo_latency_ms = (t2 - t1) * 1000

            record = {
                "trace_id": tid,
                "timestamp": time.time(),
                "ground_truth_label": labels[tid],
                "anomaly_type": anomaly_types[tid],
                "semantic_score": round(score, 4),
                "semantic_detected": len(flagged) > 0,
                "semantic_latency_ms": round(semantic_latency_ms, 4),
                "topo_latency_ms": round(topo_latency_ms, 4),
                "total_latency_ms": round(semantic_latency_ms + topo_latency_ms, 4),
            }
            n_stressed = 0
            for svc, idx in SVC_IDX.items():
                p = stress_probs[idx].item()
                record[f"stress_{svc.lower()}"] = round(p, 4)
                if p > 0.5:
                    n_stressed += 1
            record["cascade_detected"] = n_stressed >= 2

            out.write(json.dumps(record) + "\n")
            out.flush()
            if live:
                time.sleep(0.3)

    n_anom = sum(labels.values())
    print(f"\nPipeline terminée : {len(traces)} traces traitées, {n_anom} anormales (vérité terrain).")
    print(f"Résultats écrits dans {OUTPUT_FILE} — lus par Promtail -> Loki -> Grafana "
          f"(dashboard : http://localhost:3000)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                         help="écrit un résultat toutes les 0.3s pour un effet de démo en direct")
    parser.add_argument("--keep-data", action="store_true",
                         help="réutilise raw_logs.json/labels.json existants au lieu d'en régénérer")
    args = parser.parse_args()
    run_pipeline(regenerate_data=not args.keep_data, live=args.live)
