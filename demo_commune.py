"""
demo_commune.py : code partagé par detection_mobilenet.py et detection_dnn.py.

Un « détecteur » est n'importe quel objet qui possède la méthode
    detecter(image_bgr, seuil_min) -> tableau (N, 5) : x_min, y_min, x_max, y_max, confiance
(voir detecteurs.py). Ce module gère tout le reste : lecture de la source (image, dossier, vidéo ou webcam),
dessin des boîtes, floutage, affichage et enregistrement.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import config

EXTENSIONS_IMAGES = (".png", ".jpg", ".jpeg", ".bmp")
EXTENSIONS_VIDEOS = (".mp4", ".avi", ".mov", ".mkv")


def arguments(description, seuil_defaut, nom_sortie):
    """Options communes aux deux scripts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--source", default=None,
                   help="image, dossier d'images, video, ou numero de webcam (ex. 0). "
                        "Defaut : le dossier 0--Parade de WIDER_test")
    p.add_argument("--seuil", type=float, default=seuil_defaut, help="seuil de confiance (0 a 1)")
    p.add_argument("--flouter", action="store_true", help="floute les visages (anonymisation)")
    p.add_argument("--max", type=int, default=20, help="nombre maximal d'images pour un dossier")
    p.add_argument("--sortie", default=str(config.SORTIES / nom_sortie), help="dossier des resultats")
    p.add_argument("--sans-affichage", action="store_true", help="aucune fenetre (enregistre seulement)")
    p.add_argument("--delai", type=int, default=1500, help="duree d'affichage de chaque image (ms)")
    return p


def flouter(image, boite):
    """Floute la région du visage (modifie l'image). Le noyau du flou est proportionnel au visage."""
    x1, y1, x2, y2 = [int(v) for v in boite]
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    k = max(3, (max(x2 - x1, y2 - y1) // 2) | 1)            # taille impaire, exigée par GaussianBlur
    image[y1:y2, x1:x2] = cv2.GaussianBlur(image[y1:y2, x1:x2], (k, k), 0)


def traiter(image, detecteur, seuil, floutage):
    """Détecte les visages, puis dessine (ou floute) chaque visage. Renvoie (image résultat, détections)."""
    dets = detecteur.detecter(image, seuil_min=seuil)
    dets = dets[dets[:, 4] >= seuil]
    sortie = image.copy()
    for x1, y1, x2, y2, conf in dets:
        if floutage:
            flouter(sortie, (x1, y1, x2, y2))
        cv2.rectangle(sortie, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(sortie, f"{conf:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return sortie, dets


def bandeau(image, texte):
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(image, texte, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image


def pour_ecran(image, hauteur_max=720):
    """Réduit les grandes images pour qu'elles tiennent à l'écran (l'image enregistrée garde sa taille)."""
    h = image.shape[0]
    return cv2.resize(image, (int(image.shape[1] * hauteur_max / h), hauteur_max)) if h > hauteur_max else image


def _est_webcam(source):
    return source is not None and str(source).isdigit()


def lancer(detecteur, a):
    """Point d'entrée commun : choisit le mode selon la source."""
    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    if _est_webcam(a.source):
        return flux_video(detecteur, int(a.source), a, None)
    source = Path(a.source) if a.source else config.IMAGES_TEST / "0--Parade"
    if not source.exists():
        raise SystemExit(f"Source introuvable : {source}")
    if source.is_file() and source.suffix.lower() in EXTENSIONS_VIDEOS:
        return flux_video(detecteur, str(source), a, sortie / f"{source.stem}_resultat.mp4")
    images = [source] if source.is_file() else \
        sorted(p for p in source.iterdir() if p.suffix.lower() in EXTENSIONS_IMAGES)[:a.max]
    return lot_images(detecteur, images, a, sortie)


def lot_images(detecteur, images, a, sortie):
    if not images:
        raise SystemExit("Aucune image a traiter.")
    print(f"{detecteur.nom} : {len(images)} image(s), seuil {a.seuil}{', floutage' if a.flouter else ''}")
    total_visages, temps = 0, []
    with open(sortie / "detections.txt", "w", encoding="utf-8") as journal:
        for chemin in images:
            image = cv2.imread(str(chemin))
            if image is None:
                print(f"  Lecture impossible : {chemin.name}")
                continue
            t0 = time.perf_counter()
            resultat, dets = traiter(image, detecteur, a.seuil, a.flouter)
            temps.append(time.perf_counter() - t0)
            total_visages += len(dets)
            bandeau(resultat, f"{detecteur.nom} : {len(dets)} visage(s) - {temps[-1] * 1000:.0f} ms")
            cv2.imwrite(str(sortie / chemin.name), resultat)
            journal.write(f"\n--- {chemin.name} : {len(dets)} visage(s) ---\n")
            for i, (x1, y1, x2, y2, c) in enumerate(dets, 1):
                journal.write(f"Visage {i} : confiance {c:.2f}, boite ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})\n")
            print(f"  {chemin.name} : {len(dets)} visage(s) en {temps[-1] * 1000:.0f} ms")
            if not a.sans_affichage:
                cv2.imshow(f"{detecteur.nom} (q pour quitter)", pour_ecran(resultat))
                if cv2.waitKey(a.delai) & 0xFF == ord("q"):
                    break
    if not a.sans_affichage:
        cv2.destroyAllWindows()
    # la première image inclut l'initialisation du modèle : on l'écarte de la moyenne quand c'est possible
    moyenne = np.mean(temps[1:] if len(temps) > 1 else temps) * 1000 if temps else 0
    print(f"Termine : {total_visages} visage(s) au total, {moyenne:.1f} ms par image en moyenne. Resultats : {sortie}")


def flux_video(detecteur, source, a, fichier_sortie):
    """Webcam (source = numéro) ou fichier vidéo. Touche q pour quitter."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir la source video : {source}")
    ecrivain = None
    if fichier_sortie is not None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        taille = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        ecrivain = cv2.VideoWriter(str(fichier_sortie), cv2.VideoWriter_fourcc(*"mp4v"), fps, taille)
    print(f"{detecteur.nom} : lecture de {source} (q pour quitter)")
    fps_lisse = None
    while True:
        ok, image = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        resultat, dets = traiter(image, detecteur, a.seuil, a.flouter)
        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        fps_lisse = fps if fps_lisse is None else 0.9 * fps_lisse + 0.1 * fps     # moyenne glissante
        bandeau(resultat, f"{detecteur.nom} : {len(dets)} visage(s) - {fps_lisse:.0f} images/s")
        if ecrivain is not None:
            ecrivain.write(resultat)
        if not a.sans_affichage:
            cv2.imshow(f"{detecteur.nom} (q pour quitter)", pour_ecran(resultat))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if ecrivain is not None:
        ecrivain.release()
        print(f"Video enregistree : {fichier_sortie}")
    cv2.destroyAllWindows()
