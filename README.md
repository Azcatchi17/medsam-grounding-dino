# MedSAM + Grounding DINO Pipeline

Pipeline de détection et segmentation d'images médicales combinant **Grounding DINO** (génération de boîtes englobantes à partir de requêtes textuelles) et **MedSAM** (segmentation guidée par invites géométriques).

Ce dépôt propose une version refactorisée, modulaire et orientée reproductibilité du projet initialement développé dans le cadre du cours INF8225 (Polytechnique Montréal). Il formalise un pipeline d'évaluation propre et standardisé pour Grounding DINO et MedSAM.

Le projet évalue les performances du système sur deux cas d'usage :
* Polypes colorectaux (Dataset Kvasir-SEG)
* Tumeurs pancréatiques (Dataset MSD Pancreas)

---

## Structure du projet

* `assets/` : Visuels et schémas pour la documentation.
* `checkpoints/` : Poids des modèles locaux (.pth), exclus du suivi Git.
* `configs/` : Fichiers de configuration (hyperparamètres, requêtes textuelles, seuils optimaux).
* `data/` : Données locales brutes (`raw/`) et prétraitées (`processed/`), exclues du suivi Git.
* `notebooks/` : Démonstrations interactives et visualisations des résultats.
* `results/` : Métriques d'évaluation et exports de prédictions.
* `src/` : Code source modulaire :
  * `data/` : Prétraitement des images/volumes et création des jeux de données.
  * `models/` : Wrappers d'inférence pour Grounding DINO, MedSAM et ResNet.
  * `pipeline/` : Scripts d'exécution de bout en bout (calibration, évaluation).
  * `utils/` : Fonctions utilitaires (calcul des métriques Dice/IoU, vérification d'environnement).

---

## Installation

### Prérequis matériels
* Système d'exploitation : Linux ou WSL2 (Ubuntu)
* Carte graphique : GPU NVIDIA avec support CUDA 12.1+
* Python : 3.11

### Installation avec uv (Recommandé)

```bash
# 1. Cloner le dépôt et se placer dedans
git clone [https://github.com/](https://github.com/)<VOTRE_PSEUDO>/medsam-grounding-dino.git
cd medsam-grounding-dino

# 2. Synchroniser les dépendances de base
uv sync

# 3. Installer les dépendances OpenMMLab (sans isolation de build)
uv pip install --no-build-isolation mmcv==2.1.0 mmdet==3.3.0

# 4. Vérifier l'environnement
uv run python src/utils/check_env.py
```

### Installation classique (pip)

```bash
# 1. Créer et activer l'environnement virtuel
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Installer les paquets requis
pip install -r requirements.txt
pip install --no-build-isolation mmcv==2.1.0 mmdet==3.3.0

# 3. Vérifier l'environnement
python src/utils/check_env.py
```

## Crédits & Remerciements

Projet initial développé en collaboration avec Morad Bel Melih dans le cadre du cours INF8225 à Polytechnique Montréal.

Ce projet s'appuie sur les travaux et implémentations de recherche suivants :

* **MedSAM** : Wang, B., et al. (2024). *Segment Anything in Medical Images*. Nature Communications.  
  Code source officiel : [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM)
* **Segment Anything (SAM)** : Kirillov, A., et al. (2023). *Segment Anything*. ICCV 2023.  
  Architecture et code de base : [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
* **Grounding DINO** : Liu, S., et al. (2023). *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*.  
  Code source officiel : [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
* **Dataset Kvasir-SEG** : Jha, D., et al. (2020). *Kvasir-SEG: A Segmented Polyp Dataset*. Simula Research Laboratory. [datasets.simula.no/kvasir-seg](https://datasets.simula.no/kvasir-seg/)