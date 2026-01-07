# Swimming Pool Detection - Professional Structure

## Project Structure

```
swimming_pool/
├── src/                    # Source code
│   └── detect_pool.py      # Main detection script
├── models/                 # Pre-trained models
│   └── model.pt            # YOLO11m swimming pool detector (40.5 MB)
├── data/
│   ├── input/              # Input data
│   │   └── images/         # Aerial images to process
│   │       └── swimming pool/
│   └── output/             # Generated outputs
│       ├── output_*.jpg    # Annotated images
│       └── coordinates.txt # Pool coordinates
├── docs/                   # Documentation
├── .venv/                  # Python virtual environment
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Architecture

This project uses a hybrid approach combining:
- **YOLOv8 (YOLO11m)**: Robust pool localization from mozilla-ai/swimming-pool-detector
- **OpenCV**: Precise contour refinement within detected regions

## Features

- Hybrid detection combining deep learning (YOLO11m) and traditional CV (OpenCV)
- Handles pools of various shapes (rectangle, oval, irregular)
- Precise contour extraction within YOLO-detected regions
- Outputs coordinates of detected pool contours
- Generates annotated images with blue contours and labels
- Batch processing of multiple images

## Dependencies

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Ultralytics YOLO (`ultralytics`)

## Installation

1. Créer et activer un environnement virtuel:
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. Installer les dépendances:
```bash
pip install -r requirements.txt
```

**Note**: Le modèle `model.pt` (40.5 MB) est déjà inclus dans le dossier `models/`. Si vous devez le re-télécharger :
```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='mozilla-ai/swimming-pool-detector', filename='model.pt', local_dir='models')"
```

## Utilisation

1. Placez vos images aériennes dans `data/input/images/swimming pool/`

2. Exécutez le script depuis le dossier `src`:
```bash
cd src
python detect_pool.py
```

3. Les résultats seront générés dans `data/output/`:
   - `output_[nom_image].jpg` : Images annotées avec contours bleus
   - `coordinates.txt` : Coordonnées des contours détectés

## Exemple de Sortie

```
Loading YOLOv8 model from ..\models\model.pt...
Found 6 images. Starting batch processing...
[1/6] Processing: ..\data\input\images\swimming pool\000000079.jpg

0: 512x512 2 swimming_pools, 1116.7ms
Output saved to: ..\data\output\output_000000079.jpg
Detected 2 pools.
...
Batch processing complete. Output images and coordinates.txt generated in ..\data\output.
```

Le script affiche :
- Le nombre d'images trouvées
- Le temps d'inférence pour chaque image
- Le nombre de piscines détectées par image
- L'emplacement des fichiers de sortie

**Fichiers générés :**
- `output_[nom_image].jpg` : Images avec contours bleus et labels ("Pool 1", "Pool 2", etc.)
- `coordinates.txt` : Coordonnées au format `x,y;` pour chaque point du contour

## Modèle

Le projet utilise le modèle **mozilla-ai/swimming-pool-detector** :
- Basé sur YOLO11m
- Entraîné sur le dataset mozilla-ai/osm-swimming-pools
- Optimisé pour la détection de piscines dans les images satellites
- Disponible sur [Hugging Face](https://huggingface.co/mozilla-ai/swimming-pool-detector)

## License

Ce projet utilise des modèles open-source. Consultez les licences respectives des dépendances.
