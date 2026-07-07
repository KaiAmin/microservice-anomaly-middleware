"""
Step 1 : Génération de logs synthétiques multi-microservices — AVEC VARIABILITÉ.

Architecture simulée (5 services, typique e-commerce) :
  Gateway -> Order -> Payment -> Inventory -> Shipping

Contrairement à la toute première version du prototype (où TOUTES les traces
normales étaient rigoureusement identiques, et TOUTES les traces anormales
aussi), cette version introduit deux sources de variabilité réalistes :

1. BRUIT BÉNIN dans les traces normales, à deux niveaux d'intensité :
   - "léger" (BENIGN_NOISE) : un évènement bénin proche du vocabulaire des
     anomalies (recheck de stock, léger délai transporteur...).
   - "dur" (hard_noise_messages) : STRUCTURELLEMENT IDENTIQUE à une vraie
     anomalie (même nombre de "Retry:" qu'une vraie cascade) mais qui se
     résout bien. Un modèle qui se contente de compter les erreurs locales
     ne peut plus s'en sortir — il doit regarder le contexte complet de la
     trace (est-ce que Shipping réussit vraiment, ou time-out derrière ?).

2. DEUX TYPES D'ANOMALIES distincts, avec un nombre de retries variable :
   - "cascade" : Inventory retry (1 à 3 fois) -> Shipping timeout. Le stress
     se propage sur 2 services (Inventory + Shipping).
   - "contained" : Order échoue après retries (1 à 2 fois) et la requête
     s'arrête là — Payment/Inventory/Shipping ne sont même pas appelés.
     Un seul service est stressé (Order), pas de cascade.

On génère toujours 200 traces normales + 15 anormales (8 cascades + 7
contenues), pour rester comparable à la version précédente en volume.

Chaque log a : trace_id, timestamp, service, level, message (texte brut, pas
encore de template).

Fichiers produits :
  - raw_logs.json      : logs bruts, désordonnés (simulation du jitter réseau)
  - labels.json         : trace_id -> 0 (normale) / 1 (anormale)
  - stress_labels.json  : trace_id -> liste des services stressés (vérité
                           terrain fine, utilisée pour entraîner le GNN)
  - anomaly_types.json  : trace_id -> "normal" / "cascade" / "contained"
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

# Variantes bénignes : toujours des traces NORMALES (elles se terminent bien),
# mais qui contiennent un évènement "bruyant" (Retry/delay) proche du
# vocabulaire des vraies anomalies.
BENIGN_NOISE = {
    "Gateway":   ["Received request for endpoint {ep}", "Rate limit check passed",
                  "Routing request to Order service"],
    "Order":     ["Order {oid} created", "Discount code applied for order {oid}",
                  "Order {oid} validated successfully"],
    "Payment":   ["Processing payment for order {oid}",
                  "Retry: payment gateway timeout for order {oid}",
                  "Payment_Success for order {oid}"],
    "Inventory": ["Stock check for order {oid}", "Stock recheck for order {oid}",
                  "Stock reserved for order {oid}"],
    "Shipping":  ["Shipment scheduled for order {oid}",
                  "Shipping partner delay, rescheduling for order {oid}",
                  "Shipment confirmed for order {oid}"],
}
NOISE_MILD_PROB = 0.20  # probabilité, par service et par trace normale, d'utiliser la variante bruitée légère
NOISE_HARD_PROB = 0.15  # probabilité d'utiliser la variante bruitée "dure" (voir hard_noise_messages)


def hard_noise_messages(svc):
    """Bruit bénin 'dur' : structurellement IDENTIQUE à une vraie anomalie
    (même vocabulaire "Retry", même nombre de retries qu'une vraie cascade),
    mais qui se résout bien. Contrairement à BENIGN_NOISE (mild), un modèle
    qui se contente de compter les erreurs locales à CE service ne peut plus
    distinguer ce cas d'une vraie anomalie — il doit regarder ce qui se passe
    ENSUITE dans la trace (Shipping réussit-il vraiment, ou time-out ?).
    Retourne None si aucune variante dure n'est définie pour ce service."""
    if svc == "Order":
        n = random.randint(1, 2)
        return (["Order {oid} created"]
                + ["Retry: validation service unavailable for order {oid}"] * n
                + ["Order {oid} validated successfully"])
    if svc == "Inventory":
        n = random.randint(1, 2)
        return (["Stock check for order {oid}"]
                + ["Retry: stock service unresponsive for order {oid}"] * n
                + ["Stock reserved for order {oid}"])
    if svc == "Shipping":
        return ["Shipment scheduled for order {oid}",
                "ERROR: carrier temporarily unreachable for order {oid}",
                "Shipment confirmed for order {oid}"]
    return None


def normal_recipe():
    """Une trace normale = 5 étapes de service. Chaque étape a une petite
    chance d'être une variante "dure" (confusable avec une vraie anomalie),
    une chance un peu plus grande d'être une variante "légère" (bruit léger),
    et sinon la séquence normale de base."""
    recipe = []
    for svc in SERVICES:
        roll = random.random()
        hard = hard_noise_messages(svc)
        if hard is not None and roll < NOISE_HARD_PROB:
            templates = hard
        elif roll < NOISE_HARD_PROB + NOISE_MILD_PROB:
            templates = BENIGN_NOISE[svc]
        else:
            templates = NORMAL_TEMPLATES[svc]
        recipe.append((svc, templates))
    return recipe


def cascade_recipe():
    """Panne en cascade : Inventory retry (1 à 3 fois) -> Shipping timeout.
    Vérité terrain : stress propagé sur Inventory ET Shipping."""
    n_retries = random.randint(1, 3)
    recipe = [
        ("Gateway", NORMAL_TEMPLATES["Gateway"]),
        ("Order", NORMAL_TEMPLATES["Order"]),
        ("Payment", NORMAL_TEMPLATES["Payment"]),
        ("Inventory", ["Stock check for order {oid}"]
                      + ["Retry: stock service unresponsive for order {oid}"] * n_retries),
        ("Shipping", ["Shipment scheduled for order {oid}",
                      "ERROR: Timeout waiting for inventory confirmation (order {oid})"]),
    ]
    return recipe, {"Inventory", "Shipping"}


def contained_recipe():
    """Panne contenue : Order échoue après retries, la requête s'arrête là
    (Payment/Inventory/Shipping ne sont jamais appelés). Vérité terrain :
    un seul service stressé (Order), PAS de cascade."""
    n_retries = random.randint(1, 2)
    recipe = [
        ("Gateway", NORMAL_TEMPLATES["Gateway"]),
        ("Order", ["Order {oid} created"]
                  + ["Retry: validation service unavailable for order {oid}"] * n_retries
                  + ["ERROR: Order {oid} validation failed after retries"]),
    ]
    return recipe, {"Order"}


def gen_trace(trace_id, oid, recipe):
    logs = []
    t = time.time() + trace_id * 0.01
    for svc, message_templates in recipe:
        for msg_template in message_templates:
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


def build_dataset(n_normal=200, n_cascade=8, n_contained=7):
    all_logs = []
    trace_id = 0
    labels = {}
    stress_labels = {}
    anomaly_types = {}

    for _ in range(n_normal):
        trace_id += 1
        oid = f"ORD{trace_id}"
        tid = f"T{trace_id:05d}"
        all_logs.extend(gen_trace(trace_id, oid, normal_recipe()))
        labels[tid] = 0
        stress_labels[tid] = []
        anomaly_types[tid] = "normal"

    for _ in range(n_cascade):
        trace_id += 1
        oid = f"ORD{trace_id}"
        tid = f"T{trace_id:05d}"
        recipe, stressed = cascade_recipe()
        all_logs.extend(gen_trace(trace_id, oid, recipe))
        labels[tid] = 1
        stress_labels[tid] = sorted(stressed)
        anomaly_types[tid] = "cascade"

    for _ in range(n_contained):
        trace_id += 1
        oid = f"ORD{trace_id}"
        tid = f"T{trace_id:05d}"
        recipe, stressed = contained_recipe()
        all_logs.extend(gen_trace(trace_id, oid, recipe))
        labels[tid] = 1
        stress_labels[tid] = sorted(stressed)
        anomaly_types[tid] = "contained"

    random.shuffle(all_logs)  # simulate out-of-order delivery (network jitter)
    return all_logs, labels, stress_labels, anomaly_types


if __name__ == "__main__":
    logs, labels, stress_labels, anomaly_types = build_dataset()
    with open("raw_logs.json", "w") as f:
        json.dump(logs, f, indent=2)
    with open("labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    with open("stress_labels.json", "w") as f:
        json.dump(stress_labels, f, indent=2)
    with open("anomaly_types.json", "w") as f:
        json.dump(anomaly_types, f, indent=2)

    n_anom = sum(labels.values())
    n_cascade_actual = sum(1 for t in anomaly_types.values() if t == "cascade")
    n_contained_actual = sum(1 for t in anomaly_types.values() if t == "contained")
    print(f"Générés : {len(logs)} lignes de logs, {len(labels)} traces "
          f"({n_anom} anormales / {len(labels)} totales : "
          f"{n_cascade_actual} cascades, {n_contained_actual} pannes contenues) — "
          f"traces normales avec ~{NOISE_HARD_PROB*100:.0f}% de bruit 'dur' (confusable) "
          f"et ~{NOISE_MILD_PROB*100:.0f}% de bruit léger, par service")
