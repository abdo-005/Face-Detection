"""
explore_dataset.py : analyse exploratoire de WIDER FACE (pour la partie « Dataset » de la présentation).

    python explore_dataset.py              # statistiques + figures (train et val)
    python explore_dataset.py --exemples 6 # + 6 images du train avec leurs boîtes dessinées

Sorties dans sorties/exploration/ : stats_train.json, stats_val.json, figure_train.png, figure_val.png, exemples/
"""
import argparse
import json
import random
from collections import Counter

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from wider_utils import NOMS_ATTRIBUTS, dessiner_boites, lire_annotations

LIBELLES = {
    "blur": {0: "net", 1: "flou moyen", 2: "flou fort"},
    "occlusion": {0: "aucune", 1: "partielle", 2: "forte"},
    "pose": {0: "typique", 1: "atypique"},
    "illumination": {0: "normale", 1: "extreme"},
    "expression": {0: "typique", 1: "exageree"},
}


def resumer(echantillons):
    """Calcule les statistiques d'un jeu annoté."""
    boites = [e.boites for e in echantillons if len(e.boites)]
    attrs = np.concatenate([e.attrs for e in echantillons if len(e.boites)])
    boites = np.concatenate(boites)
    valides = attrs[:, NOMS_ATTRIBUTS.index("invalid")] == 0
    haut = (boites[:, 3] - boites[:, 1])[valides]
    larg = (boites[:, 2] - boites[:, 0])[valides]
    par_image = np.array([len(e.boites) for e in echantillons])
    stats = {
        "images": len(echantillons),
        "images_sans_visage": int((par_image == 0).sum()),
        "evenements": len({e.evenement for e in echantillons}),
        "visages_annotes": int(len(boites)),
        "visages_invalides": int((~valides).sum()),
        "visages_valides": int(valides.sum()),
        "visages_par_image_moyenne": round(float(par_image.mean()), 2),
        "visages_par_image_mediane": float(np.median(par_image)),
        "visages_par_image_max": int(par_image.max()),
        "hauteur_px_p10_p50_p90": [round(float(x), 1) for x in np.percentile(haut, [10, 50, 90])],
        "part_visages_moins_16px": round(float((np.minimum(haut, larg) < 16).mean()), 3),
        "part_visages_moins_32px": round(float((np.minimum(haut, larg) < 32).mean()), 3),
        "attributs": {},
    }
    for nom, libelles in LIBELLES.items():
        col = attrs[valides, NOMS_ATTRIBUTS.index(nom)].astype(int)
        c = Counter(col.tolist())
        stats["attributs"][nom] = {libelles[k]: int(c.get(k, 0)) for k in libelles}
    return stats, par_image, haut


def figure(stats, par_image, haut, titre, chemin):
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    ax = ax.ravel()
    ax[0].hist(np.clip(par_image, 0, 60), bins=30, color="#0C0A9E")
    ax[0].set_title("Visages par image (tronque a 60)")
    ax[0].set_xlabel("Nombre de visages")
    ax[0].set_ylabel("Images")
    bins = np.logspace(np.log10(max(haut.min(), 1)), np.log10(haut.max()), 40)
    ax[1].hist(haut, bins=bins, color="#EB9109")
    ax[1].set_xscale("log")
    ax[1].axvline(16, color="red", ls="--", lw=1)
    ax[1].set_title("Hauteur des visages (pixels, echelle log)")
    ax[1].set_xlabel("Hauteur (px) - ligne rouge : 16 px")
    for k, nom in enumerate(["blur", "occlusion", "pose", "illumination"]):
        d = stats["attributs"][nom]
        ax[2 + k].bar(list(d.keys()), list(d.values()), color="#0C0A9E")
        ax[2 + k].set_title(f"Attribut : {nom}")
    ax[2].set_ylabel("Visages")
    fig.suptitle(f"WIDER FACE ({titre}) : {stats['images']} images, {stats['visages_valides']} visages valides", fontsize=13)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def exemples(echantillons, n, dossier):
    """Dessine les boîtes (vert : valide, rouge : invalide) sur n images au hasard."""
    dossier.mkdir(parents=True, exist_ok=True)
    random.Random(1).shuffle(echantillons)
    ecrites = 0
    for e in echantillons:
        if ecrites >= n:
            break
        if not e.chemin.exists() or len(e.boites) == 0:
            continue
        img = cv2.imread(str(e.chemin))
        if img is None:
            continue
        inv = e.attrs[:, NOMS_ATTRIBUTS.index("invalid")] == 1
        img = dessiner_boites(img, e.boites[~inv], (0, 255, 0))
        img = dessiner_boites(img, e.boites[inv], (0, 0, 255))
        cv2.imwrite(str(dossier / f"exemple_{ecrites + 1}_{e.evenement}.jpg"), img)
        ecrites += 1
    print(f"{ecrites} image(s) d'exemple ecrites dans {dossier}")
    if ecrites == 0:
        print("  -> Aucune image trouvee : verifiez IMAGES_TRAIN avec 'python config.py'.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annot-train", default=str(config.ANNOT_TRAIN))
    p.add_argument("--annot-val", default=str(config.ANNOT_VAL))
    p.add_argument("--images-train", default=str(config.IMAGES_TRAIN))
    p.add_argument("--images-val", default=str(config.IMAGES_VAL))
    p.add_argument("--sortie", default=str(config.SORTIES / "exploration"))
    p.add_argument("--exemples", type=int, default=0)
    a = p.parse_args()

    from pathlib import Path
    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    for nom, annot, images in (("train", a.annot_train, a.images_train), ("val", a.annot_val, a.images_val)):
        if not Path(annot).exists():
            print(f"[!] Fichier d'annotations introuvable : {annot}  (verifiez config.py)")
            continue
        ech = lire_annotations(annot, images)
        stats, par_image, haut = resumer(ech)
        with open(sortie / f"stats_{nom}.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        figure(stats, par_image, haut, nom, sortie / f"figure_{nom}.png")
        print(f"\n=== {nom} ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        if nom == "train" and a.exemples:
            exemples(ech, a.exemples, sortie / "exemples")
    print(f"\nResultats dans : {sortie}")


if __name__ == "__main__":
    main()
