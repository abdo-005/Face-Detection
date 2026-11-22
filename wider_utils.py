"""
wider_utils.py : outils WIDER FACE (sans PyTorch).
 - lecture des annotations
 - filtrage des visages
 - préparation d'une image pour le réseau (redimensionnement, flip, luminosité)
 - dessin des boîtes

Format du fichier d'annotations (voir readme.txt de WIDER FACE) :
    chemin/relatif/image.jpg
    nombre_de_visages
    x1 y1 w h blur expression illumination invalid occlusion pose   (une ligne par visage)
Quand une image n'a aucun visage, une ligne factice de zéros suit quand même.
"""
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Colonnes 4 à 9 de chaque ligne d'annotation
NOMS_ATTRIBUTS = ["blur", "expression", "illumination", "invalid", "occlusion", "pose"]
IDX_INVALID = NOMS_ATTRIBUTS.index("invalid")


@dataclass
class Echantillon:
    chemin: Path            # chemin complet de l'image
    boites: np.ndarray      # (N, 4) en (x_min, y_min, x_max, y_max), pixels de l'image d'origine
    attrs: np.ndarray       # (N, 6) attributs dans l'ordre de NOMS_ATTRIBUTS

    @property
    def evenement(self):
        return self.chemin.parent.name   # ex. "0--Parade"


def lire_annotations(fichier_annot, dossier_images):
    """Lit un fichier d'annotations WIDER FACE et renvoie une liste d'Echantillon."""
    dossier_images = Path(dossier_images)
    with open(fichier_annot, encoding="utf-8") as f:
        lignes = [l.strip() for l in f if l.strip()]

    echantillons, i = [], 0
    while i < len(lignes):
        nom = lignes[i]
        n = int(lignes[i + 1])
        i += 2
        nb_lignes = max(n, 1)                 # ligne factice quand n == 0
        bloc = lignes[i:i + nb_lignes]
        i += nb_lignes
        if n == 0:
            boites = np.zeros((0, 4), np.float32)
            attrs = np.zeros((0, 6), np.float32)
        else:
            arr = np.array([[float(v) for v in l.split()] for l in bloc], np.float32)
            # (x, y, largeur, hauteur) -> (x_min, y_min, x_max, y_max)
            boites = np.stack([arr[:, 0], arr[:, 1], arr[:, 0] + arr[:, 2], arr[:, 1] + arr[:, 3]], axis=1)
            attrs = arr[:, 4:10]
        echantillons.append(Echantillon(dossier_images / nom, boites, attrs))
    return echantillons


def masque_visages_valides(boites, attrs, taille_min=0):
    """True pour les visages non invalides dont la largeur et la hauteur sont >= taille_min (pixels)."""
    larg = boites[:, 2] - boites[:, 0]
    haut = boites[:, 3] - boites[:, 1]
    return (attrs[:, IDX_INVALID] == 0) & (larg >= taille_min) & (haut >= taille_min)


def filtrer_echantillons(echantillons, taille_min=0):
    """Garde les images qui ont au moins un visage valide (et ne garde que ces visages)."""
    resultat = []
    for e in echantillons:
        m = masque_visages_valides(e.boites, e.attrs, taille_min)
        if m.any():
            resultat.append(Echantillon(e.chemin, e.boites[m], e.attrs[m]))
    return resultat


def separer(echantillons, fraction=0.05, graine=42):
    """Sépare au hasard (reproductible) : (grande partie, petite partie)."""
    rng = random.Random(graine)
    liste = list(echantillons)
    rng.shuffle(liste)
    n_petit = max(1, int(len(liste) * fraction))
    return liste[n_petit:], liste[:n_petit]


def preparer_echantillon(image_bgr, boites, taille, entrainement=False):
    """
    Prépare une image pour le réseau.
    Entrée : image BGR (OpenCV), boîtes (N, 4) xyxy en pixels d'origine.
    Sortie : image RGB uint8 (taille, taille, 3) et boîtes (M, 4) xyxy en pixels de l'image redimensionnée.
    En entraînement : retournement horizontal (50 %) et variation de luminosité/contraste.
    """
    h, w = image_bgr.shape[:2]
    img = cv2.resize(image_bgr, (taille, taille), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    b = np.asarray(boites, np.float32).reshape(-1, 4).copy()
    b *= np.array([taille / w, taille / h, taille / w, taille / h], np.float32)
    b = np.clip(b, 0, taille)
    if len(b):
        b = b[((b[:, 2] - b[:, 0]) >= 1) & ((b[:, 3] - b[:, 1]) >= 1)]

    if entrainement:
        if random.random() < 0.5:                       # retournement horizontal
            img = img[:, ::-1]
            if len(b):
                b = np.stack([taille - b[:, 2], b[:, 1], taille - b[:, 0], b[:, 3]], axis=1)
        alpha = random.uniform(0.7, 1.3)                # contraste
        beta = random.uniform(-25, 25)                  # luminosité
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), b.astype(np.float32)


def dessiner_boites(image_bgr, boites, couleur=(0, 255, 0), epaisseur=2):
    """Dessine des boîtes xyxy sur une copie de l'image."""
    sortie = image_bgr.copy()
    for x1, y1, x2, y2 in np.asarray(boites).reshape(-1, 4).astype(int):
        cv2.rectangle(sortie, (x1, y1), (x2, y2), couleur, epaisseur)
    return sortie
