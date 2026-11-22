"""
evaluate.py : compare ResNet-10 (OpenCV) et MobileNet fine-tuné sur WIDER_val (images annotées, jamais vues
pendant l'entraînement).

    python evaluate.py                    # évaluation complète (quelques minutes)
    python evaluate.py --max-images 300   # essai rapide
    python evaluate.py --seuil 0.4        # autre seuil de confiance pour précision / rappel

Sorties dans sorties/evaluation/ : resultats.json, resultats.csv, comparaison.png, echecs/*.jpg
"""
import argparse
import csv
import json
import random
import re
import time

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from detecteurs import DetecteurMobileNet, DetecteurResNet
from metriques import evaluer
from wider_utils import dessiner_boites, lire_annotations

COULEURS = ["#0C0A9E", "#EB9109", "#8208D5", "#2E8B57"]


def chronometrer(detecteur, images, n_chauffe=3):
    """Temps moyen (ms) par image, hors premières images (initialisation)."""
    temps = []
    for k, img in enumerate(images):
        t0 = time.perf_counter()
        detecteur.detecter(img)
        if k >= n_chauffe:
            temps.append(time.perf_counter() - t0)
    return round(float(np.mean(temps) * 1000), 2) if temps else None


def figures(resumes, courbes, vitesses, chemin):
    noms = list(resumes)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    for nom, c in zip(noms, COULEURS):                                   # courbes précision-rappel
        rap, pre = courbes[nom]
        ax[0, 0].plot(rap, pre, color=c, label=f"{nom} (AP = {resumes[nom]['AP']:.3f})")
    ax[0, 0].set(xlabel="Rappel", ylabel="Precision", title="Courbes precision-rappel (IoU 0.5)", xlim=(0, 1), ylim=(0, 1.02))
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].legend()

    seuil = resumes[noms[0]]["seuil"]
    metriques = [("AP", "AP"), ("precision", f"Precision\n(seuil {seuil})"), ("rappel", f"Rappel\n(seuil {seuil})"), ("f1", f"F1\n(seuil {seuil})")]
    larg = 0.8 / len(noms)
    for i, (nom, c) in enumerate(zip(noms, COULEURS)):
        vals = [resumes[nom][k] for k, _ in metriques]
        b = ax[0, 1].bar(np.arange(len(metriques)) + i * larg, vals, larg, color=c, label=nom)
        ax[0, 1].bar_label(b, fmt="%.2f", fontsize=8)
    ax[0, 1].set_xticks(np.arange(len(metriques)) + larg * (len(noms) - 1) / 2)
    ax[0, 1].set_xticklabels([l for _, l in metriques])
    ax[0, 1].set(ylim=(0, 1.1), title="Metriques globales")
    ax[0, 1].legend()

    tranches = list(resumes[noms[0]]["rappel_par_taille"])               # rappel selon la taille des visages
    for i, (nom, c) in enumerate(zip(noms, COULEURS)):
        vals = [(resumes[nom]["rappel_par_taille"][t]["rappel"] or 0) for t in tranches]
        b = ax[1, 0].bar(np.arange(len(tranches)) + i * larg, vals, larg, color=c, label=nom)
        ax[1, 0].bar_label(b, fmt="%.2f", fontsize=8)
    ax[1, 0].set_xticks(np.arange(len(tranches)) + larg * (len(noms) - 1) / 2)
    ax[1, 0].set_xticklabels([t.replace(" (", "\n(") for t in tranches])
    ax[1, 0].set(ylim=(0, 1.1), title=f"Rappel selon la taille des visages (seuil {seuil})")
    ax[1, 0].legend()

    etiquettes = list(vitesses)                                          # vitesse
    vals = [vitesses[k] for k in etiquettes]
    b = ax[1, 1].barh(etiquettes, vals, color=COULEURS[:len(etiquettes)])
    ax[1, 1].bar_label(b, fmt="%.1f ms", fontsize=9)
    ax[1, 1].set(xlabel="Millisecondes par image (moins = plus rapide)", title="Vitesse d'inference")
    ax[1, 1].set_xlim(0, max(vals) * 1.25)

    fig.suptitle("Comparaison sur WIDER FACE (jeu de validation)", fontsize=14)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def figure_analyse(resumes, chemin):
    """Seuil de confiance et rappel selon le flou, l'occlusion et la pose."""
    noms = list(resumes)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    for nom, c in zip(noms, COULEURS):                                   # rappel et precision selon le seuil
        t = resumes[nom]["par_seuil"]
        sv = [float(k) for k in t]
        ax[0, 0].plot(sv, [t[k]["rappel"] for k in t], "-o", color=c, label=f"{nom} - rappel")
        ax[0, 0].plot(sv, [t[k]["precision"] for k in t], "--s", color=c, alpha=0.6, label=f"{nom} - precision")
    ax[0, 0].set(xlabel="Seuil de confiance", ylabel="Valeur", ylim=(0, 1.05),
                 title="Effet du seuil de confiance (IoU 0.5)")
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)

    seuil = resumes[noms[0]]["seuil"]
    larg = 0.8 / len(noms)
    for k, nom_a in enumerate(["blur", "occlusion", "pose"]):
        a = ax.ravel()[1 + k]
        cats = list(resumes[noms[0]]["rappel_par_attribut"][nom_a])
        for i, (nom, c) in enumerate(zip(noms, COULEURS)):
            d = resumes[nom]["rappel_par_attribut"][nom_a]
            vals = [(d[x]["rappel"] or 0) for x in cats]
            b = a.bar(np.arange(len(cats)) + i * larg, vals, larg, color=c, label=nom)
            a.bar_label(b, fmt="%.2f", fontsize=8)
        n = resumes[noms[0]]["rappel_par_attribut"][nom_a]
        a.set_xticks(np.arange(len(cats)) + larg * (len(noms) - 1) / 2)
        a.set_xticklabels([f"{x}\n({n[x]['visages']} visages)" for x in cats])
        a.set(ylim=(0, 1.1), title=f"Rappel selon l'attribut : {nom_a} (seuil {seuil})")
        if k == 0:
            a.legend(fontsize=8)
    fig.suptitle("Analyse des echecs : seuil, flou, occlusion, pose", fontsize=14)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def sauver_echecs(donnees, echantillons, nom_modele, dossier, seuil, k=6):
    """Images où le modèle rate le plus de visages (>= 40 px). Vert : trouvé, orange : raté, rouge : faux positif."""
    dossier.mkdir(parents=True, exist_ok=True)
    sans_accent = re.sub(r"[^A-Za-z0-9]+", "_", nom_modele).strip("_")
    pires = sorted([d for d in donnees if d[1] > 0], key=lambda d: -d[1])[:k]
    for rang, (idx, manques, ignore, dets, statut, gt_score) in enumerate(pires, 1):
        e = echantillons[idx]
        img = cv2.imread(str(e.chemin))
        trouves = (~ignore) & (gt_score >= seuil)
        rates = (~ignore) & (gt_score < seuil)
        img = dessiner_boites(img, e.boites[trouves], (0, 200, 0))
        img = dessiner_boites(img, e.boites[rates], (0, 140, 255), 3)
        fp = (statut == 0) & (dets[:, 4] >= seuil) if len(dets) else np.zeros(0, bool)
        img = dessiner_boites(img, dets[fp, :4], (0, 0, 255), 1)
        cv2.putText(img, f"{nom_modele_court(nom_modele)} : {manques} visage(s) >= 40 px rates (orange)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(dossier / f"{sans_accent}_{rang}_{e.evenement}.jpg"), img)


def nom_modele_court(nom):
    return nom.split(" (")[0]


def executer(detecteurs, echantillons, sortie, seuil=0.5, taille_min=20, iou=0.5, vitesse_supp=None):
    sortie.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation de {len(detecteurs)} modele(s) sur {len(echantillons)} images "
          f"(visages ignores s'ils sont invalides ou < {taille_min} px ; IoU {iou} ; seuil {seuil})")
    resumes, courbes, brut = evaluer(detecteurs, echantillons, taille_min, iou, seuil)

    vitesses = {nom: r["ms_par_image"] for nom, r in resumes.items()}
    if vitesse_supp:                                                     # modèles chronométrés à part (ex. MobileNet sur CPU)
        images = [cv2.imread(str(e.chemin)) for e in echantillons[:40]]
        images = [i for i in images if i is not None]
        for nom, det in vitesse_supp.items():
            vitesses[nom] = chronometrer(det, images)

    with open(sortie / "resultats.json", "w", encoding="utf-8") as f:
        json.dump({"parametres": {"seuil": seuil, "iou": iou, "taille_min_px": taille_min, "images": len(echantillons)},
                   "modeles": resumes, "ms_par_image": vitesses}, f, indent=2, ensure_ascii=False)
    with open(sortie / "resultats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modele", "AP", "precision", "rappel", "f1", "meilleur_f1", "seuil_meilleur_f1",
                    "vrais_positifs", "faux_positifs", "faux_negatifs", "ms_par_image"])
        for nom, r in resumes.items():
            w.writerow([nom, r["AP"], r["precision"], r["rappel"], r["f1"], r["meilleur_f1"], r["seuil_meilleur_f1"],
                        r["vrais_positifs"], r["faux_positifs"], r["faux_negatifs"], r["ms_par_image"]])

    with open(sortie / "seuils.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modele", "seuil", "precision", "rappel", "f1", "faux_positifs", "visages_non_trouves"])
        for nom, r in resumes.items():
            for sv, v in r["par_seuil"].items():
                w.writerow([nom, sv, v["precision"], v["rappel"], v["f1"], v["faux_positifs"], v["visages_non_trouves"]])
    with open(sortie / "attributs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modele", "attribut", "valeur", "visages", "rappel"])
        for nom, r in resumes.items():
            for a, d in r["rappel_par_attribut"].items():
                for lib, v in d.items():
                    w.writerow([nom, a, lib, v["visages"], "" if v["rappel"] is None else round(v["rappel"], 4)])

    figures(resumes, courbes, vitesses, sortie / "comparaison.png")
    figure_analyse(resumes, sortie / "analyse_echecs.png")
    for nom in detecteurs:
        sauver_echecs(brut[nom], echantillons, nom, sortie / "echecs", seuil)

    print("\n" + "=" * 100)
    print(f"{'Modele':28s}{'AP':>7s}{'Prec.':>8s}{'Rappel':>8s}{'F1':>7s}{'meilleur F1 (seuil)':>22s}{'FP':>8s}{'FN':>8s}")
    for nom, r in resumes.items():
        print(f"{nom:28s}{r['AP']:7.3f}{r['precision']:8.3f}{r['rappel']:8.3f}{r['f1']:7.3f}"
              f"{r['meilleur_f1']:14.3f} ({r['seuil_meilleur_f1']:.2f}){r['faux_positifs']:8d}{r['faux_negatifs']:8d}")
    print("-" * 100)
    for nom, ms in vitesses.items():
        print(f"Vitesse {nom:30s}: {ms} ms/image ({1000 / ms:.0f} images/s)" if ms else f"Vitesse {nom}: n/a")
    print("=" * 100)
    for nom, r in resumes.items():
        print(f"Rappel par taille - {nom}: " + " | ".join(f"{t}: {v['rappel']:.2f}" if v['rappel'] is not None else f"{t}: n/a"
                                                          for t, v in r["rappel_par_taille"].items()))
    print("\nPrecision / rappel selon le seuil de confiance :")
    for nom, r in resumes.items():
        print(f"  {nom}")
        print("    seuil   : " + " ".join(f"{float(k):6.2f}" for k in r["par_seuil"]))
        print("    precision: " + " ".join(f"{v['precision']:6.2f}" for v in r["par_seuil"].values()))
        print("    rappel   : " + " ".join(f"{v['rappel']:6.2f}" for v in r["par_seuil"].values()))
    print("\nRappel selon les attributs (seuil choisi) :")
    for nom, r in resumes.items():
        morceaux = []
        for a, d in r["rappel_par_attribut"].items():
            morceaux.append(f"{a}: " + ", ".join(f"{lib} {v['rappel']:.2f}" for lib, v in d.items() if v["rappel"] is not None))
        print(f"  {nom} -> " + " | ".join(morceaux))
    print(f"\nResultats dans : {sortie}")
    return resumes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poids", default=str(config.MODELE_MOBILENET))
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--seuil", type=float, default=0.5)
    p.add_argument("--taille-min", type=int, default=20, help="plus petit visage evalue (pixels)")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default=None, help="cuda ou cpu pour MobileNet (defaut : automatique)")
    p.add_argument("--sans-cpu", action="store_true", help="ne mesure pas la vitesse de MobileNet sur CPU")
    p.add_argument("--sortie", default=str(config.DOSSIER_EVAL))
    a = p.parse_args()

    echantillons = [e for e in lire_annotations(config.ANNOT_VAL, config.IMAGES_VAL) if e.chemin.exists()]
    if not echantillons:
        raise SystemExit(f"Aucune image trouvee (IMAGES_VAL = {config.IMAGES_VAL}). Lancez 'python config.py'.")
    if a.max_images and len(echantillons) > a.max_images:
        random.Random(42).shuffle(echantillons)
        echantillons = echantillons[:a.max_images]

    resnet = DetecteurResNet(config.PROTOTXT, config.CAFFEMODEL)
    mobilenet = DetecteurMobileNet(a.poids, a.device)
    detecteurs = {resnet.nom: resnet, mobilenet.nom: mobilenet}
    supp = {}
    if mobilenet.device.type == "cuda" and not a.sans_cpu:
        cpu = DetecteurMobileNet(a.poids, "cpu")
        supp[cpu.nom] = cpu
    executer(detecteurs, echantillons, __import__("pathlib").Path(a.sortie), a.seuil, a.taille_min, a.iou, supp)


if __name__ == "__main__":
    main()
