"""
total_mobilenet.py : démonstration de floutage automatique des visages par lots (WIDER_test).
Reprend la logique de total.py, avec le choix du détecteur : ResNet-10, MobileNet fine-tuné, ou les deux côte à côte.

    python total_mobilenet.py                                   # comparaison ResNet-10 / MobileNet, 20 images
    python total_mobilenet.py --modele mobilenet --max 50       # MobileNet seul
    python total_mobilenet.py --dossier 5--Car_Accident
    python total_mobilenet.py --seuil 0.5                       # même seuil pour les deux modèles
    python total_mobilenet.py --sans-affichage                  # sans fenêtre (enregistre seulement les images)

Sans --seuil, chaque modèle utilise son seuil optimal mesuré sur WIDER_val (meilleur F1) :
SEUILS_PAR_DEFAUT = 0.15 pour ResNet-10, 0.25 pour MobileNet.

Touche « q » pour quitter la fenêtre d'affichage. Sorties dans sorties/demo/.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

import config

EXTENSIONS = (".png", ".jpg", ".jpeg")

# Seuils de confiance qui maximisent le F1 de chaque modèle sur WIDER_val (résultats de evaluate.py).
# Pour protéger la vie privée, rater un visage est pire que flouter une zone en trop : on privilégie le rappel.
SEUILS_PAR_DEFAUT = {"ResNet-10": 0.15, "MobileNet": 0.25}


def flouter(image, boite):
    """Floute la région du visage (modifie l'image). Le flou s'adapte à la taille du visage."""
    x1, y1, x2, y2 = [int(v) for v in boite]
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    k = max(3, (max(x2 - x1, y2 - y1) // 2) | 1)           # noyau impair, proportionnel au visage
    image[y1:y2, x1:x2] = cv2.GaussianBlur(image[y1:y2, x1:x2], (k, k), 0)


def traiter_image(image, detecteur, seuil, cadre=True):
    """Renvoie (image avec visages floutés, liste de (boîte, confiance))."""
    sortie = image.copy()
    dets = detecteur.detecter(image, seuil_min=seuil)
    dets = dets[dets[:, 4] >= seuil]
    for x1, y1, x2, y2, _ in dets:
        flouter(sortie, (x1, y1, x2, y2))
    if cadre:
        for x1, y1, x2, y2, _ in dets:
            cv2.rectangle(sortie, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return sortie, [(d[:4], float(d[4])) for d in dets]


def titre(image, texte):
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(image, texte, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


def hauteur_fixe(image, h=620):
    return cv2.resize(image, (int(image.shape[1] * h / image.shape[0]), h))


def executer(detecteurs, dossier_images, dossier_sortie, seuil=0.5, maximum=20, affichage=True, delai=1000, cadre=True):
    """detecteurs : dict nom_court -> détecteur. seuil : nombre commun, ou dict nom_court -> seuil.
    Traite les images du dossier et enregistre les résultats."""
    images = sorted(p for p in Path(dossier_images).iterdir() if p.suffix.lower() in EXTENSIONS)[:maximum]
    if not images:
        raise SystemExit(f"Aucune image dans {dossier_images}. Lancez 'python config.py' pour verifier les chemins.")
    dossier_sortie = Path(dossier_sortie)
    for nom in list(detecteurs) + (["comparaison"] if len(detecteurs) > 1 else []):
        (dossier_sortie / nom).mkdir(parents=True, exist_ok=True)

    seuils = seuil if isinstance(seuil, dict) else {nom: seuil for nom in detecteurs}
    print(f"Traitement de {len(images)} image(s) dans {dossier_images} (seuils de confiance : {seuils})")
    with open(dossier_sortie / "coordonnees.txt", "w", encoding="utf-8") as journal:
        for chemin in images:
            image = cv2.imread(str(chemin))
            if image is None:
                print(f"Erreur de lecture : {chemin.name}")
                continue
            journal.write(f"\n--- Fichier : {chemin.name} ---\n")
            vues = []
            for nom, det in detecteurs.items():
                resultat, faces = traiter_image(image, det, seuils[nom], cadre)
                cv2.imwrite(str(dossier_sortie / nom / chemin.name), resultat)
                for i, (b, s) in enumerate(faces, 1):
                    journal.write(f"[{nom}] Visage {i} - confiance {s:.2f} - Box(X:{int(b[0])},Y:{int(b[1])} -> X:{int(b[2])},Y:{int(b[3])})\n")
                vues.append(titre(hauteur_fixe(resultat), f"{nom} (seuil {seuils[nom]}) : {len(faces)} visage(s)"))
                print(f"[{chemin.name}] {nom} : {len(faces)} visage(s) floute(s)")
            affiche = np.hstack(vues) if len(vues) > 1 else vues[0]
            if len(vues) > 1:
                cv2.imwrite(str(dossier_sortie / "comparaison" / chemin.name), affiche)
            if affichage:
                cv2.imshow("Floutage des visages (q pour quitter)", affiche)
                if cv2.waitKey(delai) & 0xFF == ord("q"):
                    break
    if affichage:
        cv2.destroyAllWindows()
    print(f"Termine. Images floutees et coordonnees dans : {dossier_sortie}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modele", choices=["mobilenet", "resnet", "comparer"], default="comparer")
    p.add_argument("--dossier", default="0--Parade", help="sous-dossier de WIDER_test (ou chemin complet)")
    p.add_argument("--max", type=int, default=20, help="nombre maximal d'images")
    p.add_argument("--seuil", type=float, default=None, help="seuil commun aux deux modeles (defaut : seuil optimal de chacun)")
    p.add_argument("--poids", default=str(config.MODELE_MOBILENET))
    p.add_argument("--sans-affichage", action="store_true")
    p.add_argument("--sans-cadre", action="store_true", help="ne dessine pas le rectangle vert")
    p.add_argument("--delai", type=int, default=1000, help="duree d'affichage de chaque image (ms)")
    a = p.parse_args()

    dossier = Path(a.dossier)
    if not dossier.is_absolute() and not dossier.exists():
        dossier = config.IMAGES_TEST / a.dossier

    detecteurs = {}
    if a.modele in ("resnet", "comparer"):
        from detecteurs import DetecteurResNet
        detecteurs["ResNet-10"] = DetecteurResNet(config.PROTOTXT, config.CAFFEMODEL)
    if a.modele in ("mobilenet", "comparer"):
        from detecteurs import DetecteurMobileNet
        detecteurs["MobileNet"] = DetecteurMobileNet(a.poids)
    seuil = a.seuil if a.seuil is not None else {nom: SEUILS_PAR_DEFAUT[nom] for nom in detecteurs}
    executer(detecteurs, dossier, config.DOSSIER_DEMO, seuil, a.max, not a.sans_affichage, a.delai, not a.sans_cadre)


if __name__ == "__main__":
    main()
