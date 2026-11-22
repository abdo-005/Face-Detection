"""
config.py : tous les chemins du projet au même endroit.

Les dossiers WIDER FACE sont souvent décompressés avec un dossier imbriqué
(ex. WIDER_train/WIDER_train/images). Ce fichier cherche donc automatiquement
les bons dossiers. Pour vérifier ce qu'il a trouvé :

    python config.py
"""
import os
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent


def _premier_existant(candidats, test=lambda p: p.exists()):
    """Renvoie le premier chemin qui passe le test (sinon le premier de la liste)."""
    for c in candidats:
        if test(c):
            return c
    return candidats[0]


# Dossier racine contenant WIDER_train, WIDER_val, WIDER_test et wider_face_split.
# Par défaut : « Wider Face » dans le dossier du projet. Si le vôtre est ailleurs, définissez la variable
# d'environnement WIDER_FACE (ex. PowerShell : $env:WIDER_FACE = "D:\datasets\Wider Face").
RACINE_WIDER = _premier_existant([
    Path(os.environ["WIDER_FACE"]) if os.environ.get("WIDER_FACE") else DOSSIER_PROJET / "Wider Face",
    DOSSIER_PROJET / "Wider Face",
    Path.home() / "Downloads" / "Wider Face",
])


def _dossier_images(nom):
    """Trouve le dossier qui contient directement les sous-dossiers 0--Parade, 1--Handshaking..."""
    r = RACINE_WIDER
    candidats = [r / nom / "images", r / nom / nom / "images",
                 r / nom / "Dataset", r / nom / nom / "Dataset"]
    return _premier_existant(candidats, lambda p: (p / "0--Parade").exists())


def _annotations(nom_fichier):
    r = RACINE_WIDER
    candidats = [r / "wider_face_split" / "wider_face_split" / nom_fichier,
                 r / "wider_face_split" / nom_fichier,
                 DOSSIER_PROJET / nom_fichier]
    return _premier_existant(candidats)


# Fichiers d'annotations (boîtes des visages)
ANNOT_TRAIN = _annotations("wider_face_train_bbx_gt.txt")
ANNOT_VAL = _annotations("wider_face_val_bbx_gt.txt")

# Dossiers d'images
IMAGES_TRAIN = _dossier_images("WIDER_train")   # fine-tuning
IMAGES_VAL = _dossier_images("WIDER_val")       # évaluation chiffrée (annotée)
IMAGES_TEST = _dossier_images("WIDER_test")     # démonstration (sans annotations)

# Dossier de sortie : modèles, courbes, figures, résultats
SORTIES = DOSSIER_PROJET / "sorties"
DOSSIER_MODELES = SORTIES / "modeles"
DOSSIER_EVAL = SORTIES / "evaluation"
DOSSIER_DEMO = SORTIES / "demo"

# Modèles
MODELE_MOBILENET = DOSSIER_MODELES / "meilleur.pt"                         # fine-tuné par train_mobilenet.py
PROTOTXT = DOSSIER_PROJET / "deploy.prototxt"                              # ResNet-10 (OpenCV)
CAFFEMODEL = DOSSIER_PROJET / "res10_300x300_ssd_iter_140000.caffemodel"


if __name__ == "__main__":
    print(f"Racine WIDER : {RACINE_WIDER}  ->  {'OK' if RACINE_WIDER.exists() else 'INTROUVABLE'}")
    for nom, chemin in [("ANNOT_TRAIN", ANNOT_TRAIN), ("ANNOT_VAL", ANNOT_VAL),
                        ("IMAGES_TRAIN", IMAGES_TRAIN), ("IMAGES_VAL", IMAGES_VAL), ("IMAGES_TEST", IMAGES_TEST),
                        ("MODELE_MOBILENET", MODELE_MOBILENET), ("PROTOTXT", PROTOTXT), ("CAFFEMODEL", CAFFEMODEL)]:
        print(f"{nom:13s}: {chemin}  ->  {'OK' if chemin.exists() else 'INTROUVABLE'}")
