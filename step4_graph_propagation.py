"""
Step 4 : Couche Topologique (proxy pour le GNN)

IMPORTANT (limite honnête) : un vrai GNN (GraphSAGE/GAT) doit être ENTRAÎNÉ sur
des exemples étiquetés de propagation de panne (quel service a stressé quel
autre service). Ce genre de label fin n'existe quasiment dans aucun dataset
public, y compris DeepTraLog (qui étiquette la trace comme anormale, pas le
chemin de propagation lui-même). Ici on implémente donc une propagation par
RÈGLE (diffusion pondérée sur le graphe, façon PageRank), pas un GNN appris.
C'est suffisant pour valider l'architecture, pas pour prouver la précision
scientifique de détection de cascade.
"""
import json
import time
import networkx as nx

from step2_parse_and_group import parse_and_group


def build_service_graph(traces):
    g = nx.DiGraph()
    for tid, logs in traces.items():
        services_seq = [l["service"] for l in logs]
        for a, b in zip(services_seq, services_seq[1:]):
            if a == b:
                continue
            if g.has_edge(a, b):
                g[a][b]["weight"] += 1
            else:
                g.add_edge(a, b, weight=1)
    return g


def propagate_stress(g, seed_scores, hops=3, decay=0.5):
    """Diffusion simple : le stress d'un nœud se propage à ses voisins sortants,
    atténué par 'decay' à chaque saut. Remplace un forward-pass de GNN."""
    scores = dict(seed_scores)
    for _ in range(hops):
        new_scores = dict(scores)
        for node, s in scores.items():
            if s <= 0:
                continue
            for neighbor in g.successors(node):
                new_scores[neighbor] = new_scores.get(neighbor, 0) + s * decay
        scores = new_scores
    return scores


if __name__ == "__main__":
    with open("raw_logs.json") as f:
        logs = json.load(f)
    with open("anomaly_scores.json") as f:
        anomaly_scores = json.load(f)

    traces = parse_and_group(logs)
    g = build_service_graph(traces)

    print("Graphe de dépendance des services appris à partir des traces :")
    for a, b, data in g.edges(data=True):
        print(f"  {a:>10} -> {b:<10}  (observé {data['weight']} fois)")

    # on prend la trace la plus anormale et on regarde où le "stress" sémantique est apparu
    anomalous_tid = max(anomaly_scores, key=lambda t: anomaly_scores[t]["score"] if anomaly_scores[t]["label"] == 1 else -1)
    trace_logs = traces[anomalous_tid]

    # seed = service où la 1ère transition surprenante a été détectée (ici Inventory, cf step3)
    t0 = time.perf_counter()
    seed = {"Inventory": 1.0}
    propagated = propagate_stress(g, seed, hops=3, decay=0.6)
    t1 = time.perf_counter()

    print(f"\nPropagation du stress depuis 'Inventory' (trace {anomalous_tid}) :")
    for node, score in sorted(propagated.items(), key=lambda kv: -kv[1]):
        print(f"   {node:>10} : score de stress = {score:.3f}")

    print(f"\nTemps de propagation graphe : {(t1 - t0) * 1000:.4f} ms")
