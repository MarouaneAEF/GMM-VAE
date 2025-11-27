## GM‑VAE – Détection d’anomalies pour l’industrie & maintenance prédictive

**GM‑VAE** est un moteur d’IA conçu pour l’**industrie** : il apprend automatiquement les **modes de fonctionnement normaux** de vos machines, lignes de production ou procédés, puis détecte les **comportements anormaux** avant qu’ils ne se transforment en pannes, dérives qualité ou arrêts imprévus.

L’objectif : **réduire les arrêts non planifiés**, **diminuer la casse & les rebuts**, et **augmenter la disponibilité des équipements** sans dépendre de gros volumes de données labellisées.

---

## Problème industriel adressé

Dans les usines, ateliers et infrastructures :

- Les équipements génèrent **des milliers de signaux** (température, vibration, pression, courant, débits, vitesses…).
- Les **pannes critiques** et les **dérives de process** sont rares mais **extrêmement coûteuses** :
  - arrêts de ligne,
  - pénalités de retard,
  - rebuts et retours clients,
  - risques sécurité.
- Les approches classiques reposent sur :
  - des **règles fixes** (seuils sur capteurs, règles métier codées à la main) → beaucoup de fausses alertes, peu de robustesse,
  - des **modèles supervisés** qui exigent un historique riche d’incidents bien labellisés → rarement disponible.

Résultat :  
Les équipes maintenance & process passent trop de temps à **éteindre des incendies**, et pas assez à **prévenir les incidents**.

---

## Solution : apprendre automatiquement les régimes de fonctionnement “sains”

**GM‑VAE** apprend à partir des **données historiques de fonctionnement normal** de vos équipements (ou majoritairement normales), sans labels d’anomalies, pour :

- **Modéliser les différents régimes de fonctionnement** (vitesse lente/rapide, charge partielle/pleine charge, modes jour/nuit, configuration produit, etc.).
- **Attribuer un score d’anomalie** à chaque nouvelle observation ou fenêtre temporelle :
  - si le comportement ressemble à un régime connu → normal,
  - s’il s’en écarte fortement → suspicion d’anomalie.
- **Fournir un signal unifié** (“santé” de la machine / du process) que l’on peut surveiller en continu et connecter à l’alerting existant.

Techniquement, GM‑VAE combine :

- un **Variational Autoencoder (VAE)** qui apprend une représentation compacte des signaux,
- un **Gaussian Mixture Model (GMM)** qui segmente automatiquement ces représentations en **régimes de fonctionnement typiques**.

---

## KPI & impact industriel

GM‑VAE vise à améliorer des indicateurs industriels clés :

- **Disponibilité & OEE (Overall Equipment Effectiveness)**
  - **KPI** : réduction des arrêts non planifiés (%), augmentation de l’OEE global.
- **MTBF / MTTR**
  - **KPI** : augmentation du **Mean Time Between Failures** (MTBF), réduction du **Mean Time To Repair** (MTTR) grâce à une détection plus précoce.
- **Taux de rebuts & retours**
  - **KPI** : diminution du taux de produits non conformes, réduction des coûts de non‑qualité.
- **Coûts de maintenance**
  - **KPI** : part de la maintenance passant du correctif au préventif/prédictif, baisse du coût global de maintenance par équipement.
- **Charge des équipes**
  - **KPI** : réduction du nombre de fausses alertes, temps économisé sur l’analyse manuelle de données ou de logs.

Ces KPI peuvent être suivis dans un **tableau de bord opérationnel** pour mesurer le ROI de la solution.

---

## Cas d’usage industriels

- **Maintenance prédictive sur équipements tournants**
  - Pompes, moteurs, ventilateurs, compresseurs, convoyeurs…
  - Utilisation des signaux de vibration, courant, température pour détecter :
    - déséquilibres,
    - défauts de roulements,
    - surchauffes,
    - dérives mécaniques.
- **Surveillance de lignes de production**
  - Lignes d’assemblage, conditionnement, embouteillage, impression, etc.
  - Détection de dérives subtiles dans les cadences, forces, temps de cycle, qui annoncent une future panne ou une chute de qualité.
- **Contrôle de procédés continus**
  - Chimie, agroalimentaire, pharmaceutique, énergie.
  - Surveillance de variables de process (pressions, débits, températures, concentrations) pour anticiper :
    - dérives de consignes,
    - instabilités,
    - pertes de rendement.
- **Qualité & métrologie**
  - Analyse de mesures dimensionnelles, tests de fin de ligne, signaux de contrôle qualité.
  - Identification de **profils de production anormaux** avant que la non‑qualité ne devienne massive.

---

## Pour qui dans l’usine ?

- **Responsables maintenance & fiabilité**
  - qui veulent réduire les pannes surprises et mieux planifier les interventions.
- **Responsables de production & responsables d’atelier**
  - qui visent une meilleure stabilité des lignes et un OEE plus élevé.
- **Ingénieurs process & data engineers industriels**
  - qui souhaitent exploiter pleinement la data existante pour créer des indicateurs avancés de santé machine / process.
- **Direction industrielle**
  - qui cherche des leviers concrets de **réduction des coûts** et d’**amélioration de la performance opérationnelle** via la data.

---

## Comment GM‑VAE fonctionne (version simplifiée)

Sans entrer dans les détails mathématiques :

- Chaque fenêtre de données (par ex. quelques secondes ou minutes de capteurs) est encodée en un **vecteur latent** qui capture l’essentiel du comportement.
- Un **mélange de gaussiennes** (clusters) représente les **régimes de fonctionnement normaux** appris : différents modes de marche, différentes configurations produit, etc.
- Le **score d’anomalie** est calculé à partir de :
  - la **probabilité** d’appartenance aux clusters,
  - la **distance** à ces clusters,
  - éventuellement l’**erreur de reconstruction**.
- Les points très improbables ou mal reconstruits sont considérés comme **candidats anomalies**.

---

## Intégration dans un workflow industriel

1. **Collecte & préparation des données**
   - Agrégation des signaux capteurs, états machine, métriques de production.
   - Construction de fenêtres temporelles (glissantes ou non) pour capturer le contexte.
2. **Entraînement sur données historiques**
   - Utilisation d’une période représentative (ex. plusieurs semaines/mois) où le système a principalement bien fonctionné.
   - Entraînement de GM‑VAE pour apprendre les régimes normaux.
3. **Scoring en temps réel ou quasi temps réel**
   - Pour chaque nouvelle fenêtre de données :
     - calcul du score d’anomalie,
     - mise à jour d’indicateurs de santé.
4. **Seuils & alerting**
   - Définition de seuils par niveau de criticité (avertissement, alerte, arrêt).
   - Intégration avec les outils existants : SCADA, MES, CMMS, e‑mail, SMS, Slack, etc.
5. **Boucle d’amélioration continue**
   - Retour des équipes terrain sur les alertes (vraie panne, fausse alerte, pré‑anomalie).
   - Ajustement des seuils et, si besoin, ré‑entraînement périodique avec de nouvelles données.

---

## Mise en route technique (haut niveau)

Même si le focus est industriel, GM‑VAE reste un projet Python basé sur PyTorch :

- **Dépendances** : Python 3, PyTorch, torchvision, numpy, matplotlib, tensorboard.
- **Entraînement de base** (exemple sur un jeu de données standard, à remplacer par vos données industrielles) :

```bash
python train_gmvae.py --dataset cifar10 --K 10 --epochs 100
```

Pour un cas industriel réel :

- remplacer le loader de données par vos propres signaux (ou embeddings déjà calculés),
- ajuster le nombre de clusters `K` pour refléter vos différents régimes de fonctionnement,
- connecter la sortie du modèle (score d’anomalie, appartenance aux clusters) à vos outils de monitoring.

---

## Vision produit pour l’industrie

- **Aujourd’hui** : un moteur open‑source pour prototyper rapidement des cas de **maintenance prédictive** et de **détection de dérives de process**.
- **Demain (exemples de roadmap orientée industrie)** :
  - Packaging en **microservice d’“anomaly scoring” industriel**.
  - Connecteurs standards (OPC‑UA, MQTT, Kafka, historiseurs, etc.).
  - **Dashboard industriel** dédié à la santé des machines : carte des équipements, courbes de score d’anomalie, timelines d’incidents.
  - Templates de déploiement par secteur (automobile, process, agro, énergie…) avec KPI préconfigurés (OEE, MTBF, scrap rate, etc.).


