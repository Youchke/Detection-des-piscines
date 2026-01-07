# Structure Professionnelle du Projet

## Vue d'ensemble

Ce document explique l'organisation du projet de détection de piscines.

## Arborescence

```
swimming_pool/
│
├── src/                          # Code source
│   └── detect_pool.py            # Script principal de détection
│
├── models/                       # Modèles pré-entraînés
│   └── model.pt                  # YOLO11m (mozilla-ai/swimming-pool-detector)
│
├── data/                         # Données
│   ├── input/                    # Données d'entrée
│   │   └── images/               # Images aériennes à traiter
│   │       └── swimming pool/    # Dossier des images
│   │
│   └── output/                   # Résultats générés
│       ├── output_*.jpg          # Images annotées
│       └── coordinates.txt       # Coordonnées des piscines
│
├── docs/                         # Documentation
│   └── STRUCTURE.md              # Ce fichier
│
├── .venv/                        # Environnement virtuel Python
│
├── requirements.txt              # Dépendances Python
└── README.md                     # Documentation principale
```

## Description des Dossiers

### `src/`
Contient le code source du projet.
- **detect_pool.py**: Script principal qui orchestre la détection hybride (YOLO + OpenCV)

### `models/`
Stocke les modèles de machine learning.
- **model.pt**: Modèle YOLO11m pré-entraîné pour la détection de piscines (40.5 MB)
  - Source: [mozilla-ai/swimming-pool-detector](https://huggingface.co/mozilla-ai/swimming-pool-detector)
  - Entraîné sur le dataset mozilla-ai/osm-swimming-pools

### `data/`
Gère toutes les données du projet.

#### `data/input/`
Données d'entrée pour le traitement.
- **images/swimming pool/**: Placez vos images aériennes ici

#### `data/output/`
Résultats générés par le script.
- **output_[nom].jpg**: Images avec contours bleus et labels
- **coordinates.txt**: Fichier texte avec les coordonnées des contours

### `docs/`
Documentation technique et guides.

### `.venv/`
Environnement virtuel Python isolé pour les dépendances du projet.

## Workflow

1. **Préparation**: Placez les images dans `data/input/images/swimming pool/`
2. **Exécution**: Lancez `python detect_pool.py` depuis le dossier `src/`
3. **Résultats**: Consultez les fichiers générés dans `data/output/`

## Avantages de cette Structure

- ✅ **Séparation claire**: Code, données, modèles et documentation séparés
- ✅ **Scalabilité**: Facile d'ajouter de nouveaux scripts ou modèles
- ✅ **Maintenabilité**: Structure intuitive pour les collaborateurs
- ✅ **Professionnalisme**: Suit les standards de l'industrie
- ✅ **Versionning**: Facile à gérer avec Git (`.venv` et `data/output` dans .gitignore)

## Bonnes Pratiques

- Ne versionnez pas `data/output/` (résultats temporaires)
- Ne versionnez pas `.venv/` (environnement local)
- Versionnez `models/` si le modèle est petit, sinon utilisez Git LFS
- Documentez tout changement dans la structure
