# Prototype — Détection d'anomalies dans les microservices (POC)

> **Branche `experiment/donnees-variabilite`** : cette branche remplace le
> dataset synthétique "trop parfait" (`main`) par un dataset avec bruit bénin
> et deux types d'anomalies distincts. Voir la section "Expérimentation" plus
> bas pour les résultats avant/après. Si les résultats ici ne conviennent pas,
> `main` reste la version baseline intacte (`git checkout main`).

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

- `step3_real_logbert.py` : petit Transformer bidirectionnel (~19k
  paramètres), entraîné from scratch sur les traces normales avec une tâche
  de masked log-key prediction (le principe réel du papier LogBERT — qui
  n'utilise pas non plus de BERT NLP pré-entraîné téléchargé). Détecte
  15/15 anomalies, latence réelle ~4-5ms/trace.
- `step4_real_gnn.py` : Graph Attention Network (~385 paramètres) codé à la
  main (pas besoin de torch_geometric vu la taille du graphe), entraîné en
  supervisé à prédire quel service est "sous stress" à partir des features
  du LogBERT + signaux d'erreur/retry, avec des arêtes DIRIGÉES (sens causal
  uniquement — voir "Expérimentation" plus bas pour pourquoi). Split
  train/test 172/43 traces : precision=0.833, recall=1.0 sur des traces
  jamais vues à l'entraînement, latence réelle ~0.08ms/trace.

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

- Step 1 : ~2300-2500 lignes de logs générées, 215 traces dont 15 anormales
  (8 pannes en cascade `Inventory→Shipping`, 7 pannes "contenues" qui
  échouent à `Order` sans se propager). Les traces normales contiennent
  ~15-20% de bruit bénin par service (retries/délais qui se résolvent bien).
- Step 2 : les traces sont reconstruites dans le bon ordre malgré le
  désordre réseau simulé, via `trace_id`.
- Step 3 (proxy et réel) : le LogBERT détecte les 15 traces anormales
  (rappel parfait) mais peut désormais générer un faux positif occasionnel
  sur une trace normale bruitée — voir "Expérimentation" ci-dessous.
- Step 4 (proxy et réel) : le graphe de services s'apprend automatiquement
  (proxy) ou le GAT prédit (réel) quels services sont stressés — avec une
  precision imparfaite mais explicable sur données bruitées.
- Step 5 / pipeline : latence bout-en-bout (LogBERT + GAT réels compris)
  de l'ordre de quelques ms par trace.
- Dashboard Grafana : compteurs de traces/anomalies en direct (qui peuvent
  désormais diverger entre eux — voir plus bas), latence d'inférence réelle
  en time series, barres de stress par service, flux des traces anormales
  en logs live.

## Expérimentation : données avec variabilité/bruit (cette branche)

Sur `main`, toutes les traces normales étaient rigoureusement identiques et
toutes les traces anormales aussi — LogBERT et le GAT obtenaient un score
parfait (15=15=15, precision=recall=1.0) mais ça ne prouvait rien sur la
capacité réelle des modèles. Cette branche introduit :

- **Du bruit bénin dans les traces normales**, à deux intensités : léger
  (`BENIGN_NOISE`, ex: "Stock recheck") et **dur** (`hard_noise_messages`,
  ex: "Retry: stock service unresponsive" x1-2 puis résolution normale) —
  structurellement identique à une vraie anomalie, distinguable seulement en
  regardant la suite de la trace (Shipping réussit-il vraiment, ou pas ?).
- **Deux types d'anomalies** : "cascade" (Inventory→Shipping, comme avant)
  et "contenue" (Order échoue seul, sans propagation — un seul service
  stressé). Vérité terrain fine dans `stress_labels.json` /
  `anomaly_types.json`.

### Résultats avant/après

| Métrique | `main` (données parfaites) | Cette branche (bruit + fix) |
|---|---|---|
| LogBERT — recall | 15/15 | 15/15 |
| LogBERT — faux positifs (sur 200 normales) | 0 | 1 |
| GAT — precision (test set) | 1.000 | 0.833 |
| GAT — recall (test set) | 1.000 | 1.000 |
| `cascade_detected` sur pannes "contenues" | n/a | 0/7 (correct) |

### Un vrai bug de GNN trouvé et corrigé en cours de route

Première tentative avec un graphe à arêtes **bidirectionnelles** (comme sur
`main`) : le GAT propageait le stress d'`Order` (panne contenue) **en sens
inverse** jusqu'à `Gateway` (P(stress) ≈ 0.59 sur les deux, alors que Gateway
n'est jamais affecté) — un phénomène de sur-propagation typique des GNN sur
petits graphes peu diversifiés (le modèle généralise "je diffuse le stress à
mes voisins" dans les deux sens faute d'assez d'exemples contrastés).
**Fix** : `build_adjacency()` dans `step4_real_gnn.py` n'utilise plus que des
arêtes dirigées dans le sens causal réel (amont → aval). Résultat : plus
aucune fuite vers Gateway, `cascade_detected` correctement à 0/7 sur les
pannes contenues, et la precision du GAT remonte de 0.714 à 0.833.

C'est un bon exemple concret à raconter au prof : un vrai défaut
d'architecture GNN (message-passing non directionnel), diagnostiqué sur des
prédictions incohérentes, corrigé, et vérifié par une nouvelle mesure —
la démarche scientifique complète, pas juste "on a fait tourner un modèle".

## Limites à présenter honnêtement à ton prof

1. **LogBERT réel, testé sur données avec variabilité (cette branche)** :
   contrairement à `main` (séquences normales toutes identiques), cette
   branche introduit du bruit bénin volontairement confusable. Le rappel
   reste parfait (15/15) mais un faux positif apparaît sur les traces
   normales bruitées — un signal plus honnête que le 100% de `main`.
2. **GAT réel, testé sur deux types d'anomalies + un vrai bug corrigé** :
   precision=0.833 (recall=1.0) après avoir corrigé une fuite de
   message-passing en sens inverse (voir ci-dessus). MAIS l'entraînement
   reste possible UNIQUEMENT parce qu'on est nous-mêmes les générateurs des
   données synthétiques : on connaît la vérité terrain injectée (quel
   service est la source du stress, quel service en est la victime).
   **En conditions réelles, ce label fin ("le service A a stressé le
   service B") n'existe dans quasiment aucun dataset public** — même
   DeepTraLog étiquette la trace entière comme anormale, pas le chemin de
   cause à effet précis. Ce script prouve la faisabilité technique de
   l'approche (rapide, généralise, localise bien la cascade même sur
   données bruitées), pas que le problème des labels réels est résolu.
   **C'est le vrai verrou scientifique de la thèse.**
3. **Le bruit ajouté reste synthétique et contrôlé** : même "dur", notre
   bruit bénin est généré par les mêmes règles que les anomalies (mêmes
   templates, juste une issue différente). De vraies données de production
   auraient une diversité lexicale et structurelle bien plus large — nos
   chiffres (precision=0.833) restent donc optimistes par rapport à un
   déploiement réel.
4. **Latence représentative pour les couches sémantique et topologique**
   (vrais forward-pass PyTorch, quelques ms/trace au total), mais toujours
   pas de vrai déploiement réseau/production (L_network et L_storage de
   l'équation L_total = L_network + L_inference + L_storage ne sont pas
   mesurés ici, seul L_inference l'est).

Ces limites sont de bons points de discussion scientifique pour ta thèse :
elles montrent que tu as identifié précisément où se situe la difficulté
réelle du problème (l'acquisition de labels de propagation fine, la
diversité des données réelles), pas juste "on manque de temps".
