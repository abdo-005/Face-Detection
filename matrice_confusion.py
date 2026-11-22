"""
matrice_confusion.py : matrice de confusion et visualisation des VP / FP / FN sur WIDER_val.

    python matrice_confusion.py                       # les deux modèles, toutes les images de WIDER_val
    python matrice_confusion.py --max-images 300      # essai rapide
    python matrice_confusion.py --modele mobilenet    # un seul modèle (mobilenet ou dnn)
    python matrice_confusion.py --seuil 0.25          # autre seuil de confiance

Rappel des notions (détection d'objets) :
  - VP (vrai positif)  : une détection qui recouvre un vrai visage (IoU >= 0,5) ;
  - FP (faux positif)  : une détection là où il n'y a pas de visage (ou un doublon) ;
  - FN (faux négatif)  : un vrai visage qu'aucune détection n'a trouvé ;
  - VN (vrai négatif)  : « une zone sans visage où le modèle n'a rien détecté ». En détection, ce nombre
    n'est PAS défini : une image contient une infinité de zones sans visage. C'est pour cela qu'on utilise
    la précision et le rappel (qui n'utilisent pas les VN) au lieu de l'exactitude (accuracy).
    La case VN de la matrice est donc affichée « non défini ».

Sorties dans sorties/confusion/ :
  - matrice_confusion.png     : matrice 2 x 2 de chaque modèle (nombres et pourcentages)
  - confusion_par_seuil.png   : VP, FP et FN selon le seuil de confiance
  - galerie_<modele>.png      : exemples de visages découpés : VP (vert), FP (rouge), FN (orange)
  - images/<modele>_*.jpg     : images complètes annotées (vert VP, rouge FP, orange FN, gris ignoré)
  - confusion.csv             : les nombres
"""
import argparse
import csv
import random
import re
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from metriques import apparier_image
from wider_utils import IDX_INVALID, lire_annotations

VERT, ROUGE, ORANGE, GRIS = (0, 200, 0), (0, 0, 255), (0, 140, 255), (160, 160, 160)   # couleurs BGR (OpenCV)
SEUILS_COURBE = np.round(np.arange(0.05, 0.96, 0.05), 2)


def charger_detecteurs(choix, poids, device):
    detecteurs = {}
    if choix in ("dnn", "les-deux"):
        from detecteurs import DetecteurResNet
        d = DetecteurResNet(config.PROTOTXT, config.CAFFEMODEL)
        detecteurs[d.nom] = d
    if choix in ("mobilenet", "les-deux"):
        from detecteurs import DetecteurMobileNet
        d = DetecteurMobileNet(poids, device)
        detecteurs[d.nom] = d
    return detecteurs


def classer_image(dets, boites, ignore, seuil, iou_min):
    """
    Classe les détections et les visages d'une image au seuil donné.
    Renvoie un dict de tableaux de boîtes : vp, fp, fn, ignores, et les scores bruts pour les courbes.
    """
    dets, statut, gt_score = apparier_image(dets, boites, ignore, iou_min)
    gardees = dets[:, 4] >= seuil if len(dets) else np.zeros(0, bool)
    return {
        "vp": dets[(statut == 1) & gardees, :5],
        "fp": dets[(statut == 0) & gardees, :5],
        "fn": boites[(~ignore) & (gt_score < seuil)],
        "ignores": boites[ignore],
        # pour les courbes selon le seuil : score de chaque détection non ignorée et score de chaque visage
        "scores_det": dets[statut >= 0, 4], "est_vp": statut[statut >= 0] == 1,
        "gt_score": gt_score[~ignore],
    }


def decouper(image, boite, cote=96, marge=0.25):
    """Découpe un visage (avec une marge) et le redimensionne en carré cote x cote."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in boite[:4]]
    mx, my = (x2 - x1) * marge, (y2 - y1) * marge
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return cv2.resize(image[y1:y2, x1:x2], (cote, cote))


def annoter(image, c):
    """Image complète : vert = VP, rouge = FP, orange = FN, gris = visage ignoré (invalide ou trop petit)."""
    sortie = image.copy()
    for boites, couleur, ep in [(c["ignores"], GRIS, 1), (c["vp"], VERT, 2), (c["fn"], ORANGE, 2), (c["fp"], ROUGE, 2)]:
        for b in boites:
            cv2.rectangle(sortie, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), couleur, ep)
    texte = f"VP {len(c['vp'])}  FP {len(c['fp'])}  FN {len(c['fn'])}"
    cv2.rectangle(sortie, (0, 0), (sortie.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(sortie, texte, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return sortie


def analyser(detecteurs, echantillons, seuil, taille_min, iou_min, n_galerie, n_images, dossier_images):
    """Parcourt les images, compte VP / FP / FN et garde des exemples pour les visualisations."""
    res = {nom: {"vp": 0, "fp": 0, "fn": 0, "scores_det": [], "est_vp": [], "gt_score": [],
                 "galerie": {"vp": [], "fp": [], "fn": []}, "images": 0} for nom in detecteurs}
    rng = random.Random(0)
    for k, e in enumerate(echantillons):
        image = cv2.imread(str(e.chemin))
        if image is None:
            continue
        larg, haut = e.boites[:, 2] - e.boites[:, 0], e.boites[:, 3] - e.boites[:, 1]
        ignore = (e.attrs[:, IDX_INVALID] == 1) | (np.minimum(larg, haut) < taille_min)
        for nom, det in detecteurs.items():
            c = classer_image(det.detecter(image), e.boites, ignore, seuil, iou_min)
            r = res[nom]
            for cle in ("vp", "fp", "fn"):
                r[cle] += len(c[cle])
                # échantillonnage aléatoire pour la galerie (les FP sont rares : on les garde plus souvent)
                proba = 0.8 if cle == "fp" else 0.15
                for b in c[cle]:
                    if len(r["galerie"][cle]) < n_galerie and rng.random() < proba:
                        crop = decouper(image, b)
                        if crop is not None:
                            r["galerie"][cle].append(crop)
            r["scores_det"].append(c["scores_det"])
            r["est_vp"].append(c["est_vp"])
            r["gt_score"].append(c["gt_score"])
            if r["images"] < n_images and len(c["fp"]) + len(c["fn"]) > 0 and len(e.boites) <= 40:
                court = re.sub(r"[^A-Za-z0-9]+", "_", nom.split(" (")[0]).strip("_")
                cv2.imwrite(str(dossier_images / f"{court}_{r['images'] + 1}_{e.evenement}.jpg"), annoter(image, c))
                r["images"] += 1
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(echantillons)} images", flush=True)
    for r in res.values():
        r["scores_det"] = np.concatenate(r["scores_det"]) if r["scores_det"] else np.zeros(0)
        r["est_vp"] = np.concatenate(r["est_vp"]) if r["est_vp"] else np.zeros(0, bool)
        r["gt_score"] = np.concatenate(r["gt_score"]) if r["gt_score"] else np.zeros(0)
    return res


def figure_matrices(res, seuil, chemin):
    """Matrice de confusion 2 x 2 par modèle : lignes = réalité, colonnes = prédiction."""
    noms = list(res)
    fig, axes = plt.subplots(1, len(noms), figsize=(6.2 * len(noms), 5.6), squeeze=False)
    for ax, nom in zip(axes[0], noms):
        r = res[nom]
        vp, fp, fn = r["vp"], r["fp"], r["fn"]
        n_visages = vp + fn
        precision = vp / max(vp + fp, 1)
        rappel = vp / max(n_visages, 1)
        # Couleur : part de la ligne (VP et FN en % des vrais visages ; FP en % des détections)
        valeurs = np.array([[vp / max(n_visages, 1), fn / max(n_visages, 1)],
                            [fp / max(vp + fp, 1), 0.0]])
        ax.imshow(valeurs, cmap="Blues", vmin=0, vmax=1)
        textes = [[f"VP\n{vp}\n({vp / max(n_visages, 1):.1%} des visages)",
                   f"FN\n{fn}\n({fn / max(n_visages, 1):.1%} des visages)"],
                  [f"FP\n{fp}\n({fp / max(vp + fp, 1):.1%} des détections)",
                   "VN\nnon défini\n(infinité de zones\nsans visage)"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, textes[i][j], ha="center", va="center", fontsize=11,
                        color="white" if valeurs[i, j] > 0.55 else "black")
        ax.set_xticks([0, 1], ["Visage", "Pas de visage"])
        ax.set_yticks([0, 1], ["Visage", "Pas de visage\n(fond)"])
        ax.set_xlabel("Prédiction du modèle")
        ax.set_ylabel("Réalité (annotation)")
        ax.set_title(f"{nom}\nprécision {precision:.3f} | rappel {rappel:.3f} | seuil {seuil}", fontsize=11)
    fig.suptitle("Matrice de confusion (WIDER_val, IoU >= 0,5)", fontsize=14)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def comptes_par_seuil(r):
    """VP, FP, FN pour chaque seuil de SEUILS_COURBE (même appariement, seuil différent)."""
    lignes = []
    for s in SEUILS_COURBE:
        m = r["scores_det"] >= s
        vp_det = int((r["est_vp"] & m).sum())
        fp = int((~r["est_vp"] & m).sum())
        fn = int((r["gt_score"] < s).sum())
        lignes.append((float(s), vp_det, fp, fn))
    return lignes


def figure_seuils(res, seuil, chemin):
    noms = list(res)
    fig, axes = plt.subplots(1, len(noms), figsize=(6.5 * len(noms), 4.8), squeeze=False)
    for ax, nom in zip(axes[0], noms):
        lignes = comptes_par_seuil(res[nom])
        s = [l[0] for l in lignes]
        ax.plot(s, [l[1] for l in lignes], "-o", color="green", label="VP (visages trouvés)")
        ax.plot(s, [l[2] for l in lignes], "-o", color="red", label="FP (fausses alertes)")
        ax.plot(s, [l[3] for l in lignes], "-o", color="darkorange", label="FN (visages ratés)")
        ax.axvline(seuil, color="gray", ls="--", label=f"seuil choisi ({seuil})")
        ax.set(xlabel="Seuil de confiance", ylabel="Nombre", title=nom)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Compromis : monter le seuil enlève des FP mais ajoute des FN", fontsize=13)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def figure_galerie(r, nom, n, chemin):
    """Grille de visages découpés : une ligne par catégorie (VP, FP, FN)."""
    categories = [("vp", "VP : visages trouvés", "green"), ("fp", "FP : fausses détections", "red"),
                  ("fn", "FN : visages ratés", "darkorange")]
    fig, axes = plt.subplots(3, n, figsize=(1.6 * n, 5.6))
    for i, (cle, titre, couleur) in enumerate(categories):
        crops = r["galerie"][cle]
        for j in range(n):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j < len(crops):
                ax.imshow(cv2.cvtColor(crops[j], cv2.COLOR_BGR2RGB))
                for cote in ax.spines.values():
                    cote.set_edgecolor(couleur)
                    cote.set_linewidth(3)
            else:
                ax.axis("off")
        axes[i, 0].set_ylabel(titre, color=couleur, fontsize=9, rotation=0, ha="right", va="center")
    fig.suptitle(f"Exemples - {nom}", fontsize=13)
    fig.tight_layout()
    fig.savefig(chemin, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Matrice de confusion et visualisation VP / FP / FN sur WIDER_val")
    p.add_argument("--modele", choices=["les-deux", "mobilenet", "dnn"], default="les-deux")
    p.add_argument("--poids", default=str(config.MODELE_MOBILENET))
    p.add_argument("--device", default=None, help="cuda ou cpu pour MobileNet (defaut : automatique)")
    p.add_argument("--max-images", type=int, default=None, help="limiter le nombre d'images (essai rapide)")
    p.add_argument("--seuil", type=float, default=0.5)
    p.add_argument("--taille-min", type=int, default=20, help="visages plus petits ignores (pixels)")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--galerie", type=int, default=10, help="exemples par categorie dans la galerie")
    p.add_argument("--images", type=int, default=6, help="images completes annotees par modele")
    p.add_argument("--sortie", default=str(config.SORTIES / "confusion"))
    a = p.parse_args()

    sortie = Path(a.sortie)
    (sortie / "images").mkdir(parents=True, exist_ok=True)

    echantillons = [e for e in lire_annotations(config.ANNOT_VAL, config.IMAGES_VAL) if e.chemin.exists()]
    if not echantillons:
        raise SystemExit(f"Aucune image trouvee (IMAGES_VAL = {config.IMAGES_VAL}). Lancez 'python config.py'.")
    random.Random(42).shuffle(echantillons)            # ordre aléatoire (reproductible) : exemples variés
    if a.max_images:
        echantillons = echantillons[:a.max_images]

    detecteurs = charger_detecteurs(a.modele, a.poids, a.device)
    print(f"Analyse de {len(echantillons)} images avec : {', '.join(detecteurs)} (seuil {a.seuil}, IoU {a.iou})")
    res = analyser(detecteurs, echantillons, a.seuil, a.taille_min, a.iou, a.galerie, a.images, sortie / "images")

    figure_matrices(res, a.seuil, sortie / "matrice_confusion.png")
    figure_seuils(res, a.seuil, sortie / "confusion_par_seuil.png")
    for nom, r in res.items():
        court = re.sub(r"[^A-Za-z0-9]+", "_", nom.split(" (")[0]).strip("_")
        figure_galerie(r, nom, a.galerie, sortie / f"galerie_{court}.png")

    with open(sortie / "confusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modele", "seuil", "VP", "FP", "FN", "VN", "precision", "rappel"])
        for nom, r in res.items():
            for s, vp, fp, fn in comptes_par_seuil(r):
                w.writerow([nom, s, vp, fp, fn, "non defini",
                            round(vp / max(vp + fp, 1), 4), round(vp / max(vp + fn, 1), 4)])

    print("\n" + "=" * 70)
    print(f"{'Modele':30s}{'VP':>8s}{'FP':>8s}{'FN':>8s}{'Precision':>11s}{'Rappel':>8s}")
    for nom, r in res.items():
        vp, fp, fn = r["vp"], r["fp"], r["fn"]
        print(f"{nom:30s}{vp:8d}{fp:8d}{fn:8d}{vp / max(vp + fp, 1):11.3f}{vp / max(vp + fn, 1):8.3f}")
    print("=" * 70)
    print(f"Resultats dans : {sortie}")


if __name__ == "__main__":
    main()
