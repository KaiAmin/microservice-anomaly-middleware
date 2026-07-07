"""
Step 2 : Ingestion et Reconstruction des Scénarios (Trace-Grouping)

- Parsing "type Drain" simplifié : on remplace les tokens variables (nombres, IDs)
  par des wildcards pour obtenir un TEMPLATE de log (ex: "Order ORD75 validated
  successfully" -> "Order <*> validated successfully")
  NB: le vrai Drain utilise un arbre de parsing à profondeur fixe ; ici on fait
  une version régulière volontairement simplifiée pour la démonstration.
- Regroupement par trace_id
- Reconstruction chronologique (tri par timestamp malgré l'arrivée désordonnée)
"""
import json
import re
from collections import defaultdict

TOKEN_RE = re.compile(r"\b([A-Z0-9]*\d[A-Z0-9]*)\b")  # IDs contenant un chiffre


def to_template(message: str) -> str:
    return TOKEN_RE.sub("<*>", message)


def parse_and_group(logs):
    traces = defaultdict(list)
    for log in logs:
        log = dict(log)
        log["template"] = to_template(log["message"])
        traces[log["trace_id"]].append(log)

    # reconstruction chronologique
    for tid in traces:
        traces[tid].sort(key=lambda x: x["timestamp"])

    return traces


if __name__ == "__main__":
    with open("raw_logs.json") as f:
        logs = json.load(f)

    traces = parse_and_group(logs)

    templates_seen = set(l["template"] for t in traces.values() for l in t)
    print(f"{len(traces)} traces reconstruites, {len(templates_seen)} templates distincts extraits:")
    for t in sorted(templates_seen):
        print("   -", t)

    # exemple : une trace normale et une trace anormale
    with open("labels.json") as f:
        labels = json.load(f)
    normal_tid = next(tid for tid, lab in labels.items() if lab == 0)
    anomalous_tid = next(tid for tid, lab in labels.items() if lab == 1)

    print(f"\nExemple trace normale ({normal_tid}) reconstituée :")
    for l in traces[normal_tid]:
        print(f"   [{l['service']:>9}] {l['template']}")

    print(f"\nExemple trace ANORMALE ({anomalous_tid}) reconstituée :")
    for l in traces[anomalous_tid]:
        print(f"   [{l['service']:>9}] {l['template']}")

    with open("traces_grouped.json", "w") as f:
        json.dump(traces, f, indent=2)
