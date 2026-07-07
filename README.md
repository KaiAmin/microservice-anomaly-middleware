# Prototype — Détection d'anomalies dans les microservices (POC)

Ce dossier contient un prototype du pipeline décrit dans le paper :
Ingestion/trace-grouping → Couche sémantique (LogBERT) → Couche topologique
(GNN) → mesure de latence → visualisation temps réel (Loki/Grafana).

**Important** : ce n'est PAS le système final. Voir la section "Limites" en
bas — c'est la partie à discuter avec ton prof. Il y a deux versions des
couches sémantique et topologique dans ce dossier :

- **Version proxy** (`step3_semantic_layer.py`, `step4_graph_propagation.py`)
  : règles écrites à la main (bigrammes, diffusion), pour valider vite la
  mécanique du pipeline.
- **Version "vrais modèles"** (`step3_real_logbert.py`, `step4_real_gnn.py`)
  : un vrai petit Transformer et un vrai GAT (Graph Attention Network),
  entraînés from scratch avec PyTorch — voir "Limites" pour ce que ça prouve
  et ne prouve pas.

## Installation (sur ta machine)

Il te faut Python 3.9+ et Docker (Docker Desktop suffit). Puis, dans ce
dossier :

```bash
pip install networkx numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu   # ou juste "pip install torch"
```

## 1. Pipeline pédagogique (proxies, rapide, sans PyTorch)

```bash
python3 step1_generate_logs.py       # génère raw_logs.json + labels.json
python3 step2_parse_and_group.py     # parsing + regroupement par trace_id
python3 step3_semantic_layer.py      # scoring d'anomalie sémantique (proxy bigrammes)
python3 step4_graph_propagation.py   # propagation sur le graphe (règle de diffusion)
python3 step5_latency_benchmark.py   # mesure de latence bout-en-bout
```

Chaque script affiche ses résultats dans le terminal et génère des fichiers
`.json` intermédiaires (`raw_logs.json`, `traces_grouped.json`,
`anomaly_scores.json`) que tu peux inspecter directement.

## 2. Vrais modèles entraînés (LogBERT + GAT, nécessite PyTorch)

```bash
python3 step3_real_logbert.py   # entraîne un vrai Transformer (masked log-key prediction)
python3 step4_real_gnn.py       # entraîne un vrai GAT (prédiction de propagation de stress)
```

- `step3_real_logbert.py` : petit Transformer bidirectionnel (~18k
  paramètres), entraîné from scratch sur les traces normales avec une tâche
  de masked log-key prediction (le principe réel du papier LogBERT — qui
  n'utilise pas non plus de BERT NLP pré-entraîné téléchargé). Détecte
  15/15 anomalies, latence réelle ~3.8ms/trace.
- `step4_real_gnn.py` : Graph Attention Network (~385 paramètres) codé à la
  main (pas besoin de torch_geometric vu la taille du graphe), entraîné en
  supervisé à prédire quel service est "sous stress" à partir des features
  du LogBERT + signaux d'erreur/retry. Split train/test 172/43 traces :
  precision=1.0, recall=1.0 sur des traces jamais vues à l'entraînement,
  latence réelle ~0.08ms/trace.

## 3. Pipeline automatisée bout-en-bout + dashboard live (Grafana)

C'est la démo à montrer en live : la pipeline complète tourne (génération →
parsing → entraînement LogBERT → entraînement GAT → scoring temps réel) et
pousse ses résultats vers un vrai stack Loki/Grafana, comme prévu dans
l'architecture de la thèse.

**Étape 1 — démarrer l'infra de visualisation** (une seule fois, la garder
ouverte pendant la démo) :

```bash
docker compose up -d
```

Ça démarre 3 conteneurs : Loki (stockage des logs), Promtail (lit
`output/middleware.log` et l'envoie à Loki), Grafana (dashboard, déjà
préconfiguré automatiquement — datasource + panels).

Ouvre ensuite **http://localhost:3000** dans le navigateur : le dashboard
"Middleware IA - Détection d'anomalies microservices" est déjà là (pas de
login requis).

**Étape 2 — lancer la pipeline en mode live** (à faire pendant que le prof
regarde le dashboard) :

```bash
python3 pipeline.py --live
```

Ce mode entraîne les deux modèles (quelques secondes) puis écrit un résultat
par trace toutes les ~0.3s dans `output/middleware.log` — tu vois les
compteurs, la latence et les barres de stress par service se remplir en
direct sur le dashboard Grafana au fur et à mesure.

Pour rejouer instantanément sans l'effet "live" (tout d'un coup) :

```bash
python3 pipeline.py
```

**Pour tout arrêter** :

```bash
docker compose down
```

## Ce que tu devrais observer

- Step 1 : ~2000 lignes de logs générées, 215 traces dont 15 anormales
  (panne en cascade injectée : `Payment_Success` → `Retry` → `Timeout`).
- Step 2 : les traces sont reconstruites dans le bon ordre malgré le
  désordre réseau simulé, via `trace_id`.
- Step 3 (proxy et réel) : le modèle détecte les 15 traces anormales
  (rappel parfait sur ces données synthétiques — voir limites ci-dessous).
- Step 4 (proxy et réel) : le graphe de services s'apprend automatiquement
  (proxy) ou le GAT prédit correctement (réel) que le stress se propage de
  `Inventory` vers `Shipping`.
- Step 5 / pipeline : latence bout-en-bout (LogBERT + GAT réels compris)
  de l'ordre de quelques ms par trace — à interpréter avec prudence, voir
  limites.
- Dashboard Grafana : compteurs de traces/anomalies en direct, latence
  d'inférence réelle en time series, barres de stress par service, flux des
  traces anormales en logs live.

## Limites à présenter honnêtement à ton prof

1. **LogBERT réel construit, mais sur données trop propres** :
   `step3_real_logbert.py` est un vrai Transformer entraîné (pas un proxy),
   mais nos traces normales synthétiques suivent TOUTES exactement la même
   séquence de templates (zéro variabilité) — la tâche de prédiction
   masquée est donc trivialement facile ici. Sur un vrai corpus avec de la
   variabilité naturelle, ce serait un test bien plus dur.
2. **GAT réel construit, mais grâce à un privilège de générateur de
   données** : `step4_real_gnn.py` entraîne un vrai GAT qui généralise
   parfaitement (precision=1.0, recall=1.0 sur des traces de test jamais
   vues). MAIS ça n'a été possible que parce qu'on est nous-mêmes les
   générateurs des données synthétiques : on connaît la vérité terrain
   injectée (quel service est la source du stress, quel service en est la
   victime). **En conditions réelles, ce label fin ("le service A a stressé
   le service B") n'existe dans quasiment aucun dataset public** — même
   DeepTraLog étiquette la trace entière comme anormale, pas le chemin de
   cause à effet précis. Ce script prouve la faisabilité technique de
   l'approche (rapide, généralise, localise bien la cascade), pas que le
   problème des labels réels est résolu. **C'est le vrai verrou
   scientifique de la thèse.**
3. **Détection "trop parfaite" (15/15, precision/recall=1.0)** : nos
   anomalies synthétiques sont trop propres. Sur des données réelles
   bruitées, les scores chuteront.
4. **Latence maintenant représentative pour la couche sémantique et
   topologique** (vrais forward-pass PyTorch, quelques ms/trace au total),
   mais toujours pas de vrai déploiement réseau/production (L_network et
   L_storage de l'équation L_total = L_network + L_inference + L_storage ne
   sont pas mesurés ici, seul L_inference l'est).

Ces limites sont de bons points de discussion scientifique pour ta thèse :
elles montrent que tu as identifié précisément où se situe la difficulté
réelle du problème (l'acquisition de labels de propagation fine), pas juste
"on manque de temps".
