"""
Step 1 : Génération de logs synthétiques multi-microservices.

Architecture simulée (5 services, typique e-commerce) :
  Gateway -> Order -> Payment -> Inventory -> Shipping

Chaque requête utilisateur = 1 trace_id.
On génère :
  - 200 traces "normales" (séquence stable, latences normales)
  - 15 traces "anormales" avec une VRAIE panne en cascade :
        Payment_Success (Payment) est immédiatement suivi d'un Retry
        anormal côté Inventory, qui se propage en timeout côté Shipping.

Chaque log a : trace_id, timestamp, service, level, message (texte brut, pas encore de template)
"""
import random
import json
import time

random.seed(42)

SERVICES = ["Gateway", "Order", "Payment", "Inventory", "Shipping"]

NORMAL_TEMPLATES = {
    "Gateway":   ["Received request for endpoint {ep}", "Routing request to Order service"],
    "Order":     ["Order {oid} created", "Order {oid} validated successfully"],
    "Payment":   ["Processing payment for order {oid}", "Payment_Success for order {oid}"],
    "Inventory": ["Stock check for order {oid}", "Stock reserved for order {oid}"],
    "Shipping":  ["Shipment scheduled for order {oid}", "Shipment confirmed for order {oid}"],
}

ANOMALY_INJECTION = {
    # ce qui remplace la séquence normale dans les traces "anormales"
    "Payment":   ["Processing payment for order {oid}", "Payment_Success for order {oid}"],
    "Inventory": ["Stock check for order {oid}", "Retry: stock service unresponsive for order {oid}",
                  "Retry: stock service unresponsive for order {oid}"],
    "Shipping":  ["Shipment scheduled for order {oid}", "ERROR: Timeout waiting for inventory confirmation (order {oid})"],
}


def gen_trace(trace_id, oid, anomalous=False):
    logs = []
    t = time.time() + trace_id * 0.01
    templates = ANOMALY_INJECTION if anomalous else NORMAL_TEMPLATES
    for svc in SERVICES:
        for msg_template in (templates.get(svc) if anomalous and svc in ["Payment", "Inventory", "Shipping"] else NORMAL_TEMPLATES[svc]):
            msg = msg_template.format(ep="/checkout", oid=oid)
            t += random.uniform(0.005, 0.03)  # jitter réseau
            logs.append({
                "trace_id": f"T{trace_id:05d}",
                "timestamp": round(t, 4),
                "service": svc,
                "level": "ERROR" if "ERROR" in msg or "Retry" in msg else "INFO",
                "message": msg,
            })
    return logs


def build_dataset(n_normal=200, n_anomalous=15):
    all_logs = []
    trace_id = 0
    labels = {}
    for _ in range(n_normal):
        trace_id += 1
        oid = f"ORD{trace_id}"
        all_logs.extend(gen_trace(trace_id, oid, anomalous=False))
        labels[f"T{trace_id:05d}"] = 0
    for _ in range(n_anomalous):
        trace_id += 1
        oid = f"ORD{trace_id}"
        all_logs.extend(gen_trace(trace_id, oid, anomalous=True))
        labels[f"T{trace_id:05d}"] = 1

    random.shuffle(all_logs)  # simulate out-of-order delivery (network jitter)
    return all_logs, labels


if __name__ == "__main__":
    logs, labels = build_dataset()
    with open("raw_logs.json", "w") as f:
        json.dump(logs, f, indent=2)
    with open("labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Générés : {len(logs)} lignes de logs, {len(labels)} traces "
          f"({sum(labels.values())} anormales / {len(labels)} totales)")
