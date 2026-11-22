"""
metriques.py : évaluation d'un détecteur de visages (numpy + OpenCV, sans PyTorch).

Protocole :
 - une détection est correcte (VP) si son IoU avec un visage non encore trouvé est >= 0.5 ;
 - une détection sans visage correspondant est un faux positif (FP) ; un visage non trouvé est un faux négatif (FN) ;
 - les visages INVALIDES ou plus petits que `taille_min` pixels sont IGNORÉS : ils ne comptent ni comme
   visages à trouver, ni comme erreurs si le modèle les détecte ;
 - AP (average precision) : aire sous la courbe précision-rappel, indépendante du seuil de confiance.
"""
import time

import cv2
import numpy as np

from wider_utils import IDX_INVALID, NOMS_ATTRIBUTS

SEUILS_TABLEAU = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
LIBELLES_ATTRIBUTS = {
    "blur": {0: "net", 1: "flou moyen", 2: "flou fort"},
    "occlusion": {0: "aucune", 1: "partielle", 2: "forte"},
    "pose": {0: "typique", 1: "atypique"},
    "illumination": {0: "normale", 1: "extreme"},
}


def iou_un_contre_tous(boite, boites):
    x1 = np.maximum(boite[0], boites[:, 0])
    y1 = np.maximum(boite[1], boites[:, 1])
    x2 = np.minimum(boite[2], boites[:, 2])
    y2 = np.minimum(boite[3], boites[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (boite[2] - boite[0]) * (boite[3] - boite[1])
    b = (boites[:, 2] - boites[:, 0]) * (boites[:, 3] - boites[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def apparier_image(dets, gt, ignore, iou_min=0.5):
    """
    dets : (N, 5) ; gt : (M, 4) ; ignore : (M,) booléens.
    Renvoie (dets triées par confiance décroissante, statut, gt_score) avec
      statut[i] : 1 = vrai positif, 0 = faux positif, -1 = ignorée
      gt_score[j] : confiance de la détection qui a trouvé le visage j (-1 si non trouvé)
    """
    if len(dets):
        dets = dets[np.argsort(-dets[:, 4])]
    statut = np.zeros(len(dets), int)
    pris = np.zeros(len(gt), bool)
    gt_score = np.full(len(gt), -1.0)
    if len(gt):
        for i, d in enumerate(dets):
            ious = iou_un_contre_tous(d[:4], gt)
            j = int(np.argmax(ious))
            if ious[j] >= iou_min:
                if ignore[j]:
                    statut[i] = -1
                elif not pris[j]:
                    statut[i], pris[j], gt_score[j] = 1, True, d[4]
    return dets, statut, gt_score


def courbe_pr(scores, tp, n_gt):
    """Courbe précision-rappel et AP (interpolation sur tous les points)."""
    if len(scores) == 0:
        return np.zeros(0), np.zeros(0), 0.0, np.zeros(0)
    ordre = np.argsort(-scores)
    tp = tp[ordre].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rappel = ctp / max(n_gt, 1)
    precision = ctp / np.maximum(ctp + cfp, 1)
    mrec = np.concatenate([[0.0], rappel, [rappel[-1]]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]          # enveloppe décroissante
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return rappel, precision, ap, scores[ordre]


TRANCHES = [("petits", None, 40), ("moyens", 40, 100), ("grands", 100, 1e9)]   # hauteur du visage (px)


def resumer_modele(r, seuil, taille_min):
    scores = np.concatenate(r["scores"]) if r["scores"] else np.zeros(0)
    tp = np.concatenate(r["tp"]).astype(bool) if r["tp"] else np.zeros(0, bool)
    n_gt = r["n_gt"]
    rap_c, pre_c, ap, sc_tries = courbe_pr(scores, tp, n_gt)

    m = scores >= seuil
    vp, fp = int(tp[m].sum()), int((~tp[m]).sum())
    precision, rappel = vp / max(vp + fp, 1), vp / max(n_gt, 1)
    f1 = 2 * precision * rappel / max(precision + rappel, 1e-9)

    if len(pre_c):
        f1_c = 2 * pre_c * rap_c / np.maximum(pre_c + rap_c, 1e-9)
        i = int(np.argmax(f1_c))
        meilleur_f1, seuil_f1 = float(f1_c[i]), float(sc_tries[i])
    else:
        meilleur_f1, seuil_f1 = 0.0, 0.0

    gt_score = np.concatenate(r["gt_score"]) if r["gt_score"] else np.zeros(0)
    gt_haut = np.concatenate(r["gt_haut"]) if r["gt_haut"] else np.zeros(0)
    tranches = {}
    for nom, lo, hi in TRANCHES:
        lo = taille_min if lo is None else lo
        sel = (gt_haut >= lo) & (gt_haut < hi)
        n = int(sel.sum())
        etiquette = f"{nom} (>{int(lo)} px)" if hi > 1e8 else f"{nom} ({int(lo)}-{int(hi)} px)"
        tranches[etiquette] = {
            "visages": n, "rappel": float((gt_score[sel] >= seuil).mean()) if n else None}

    par_seuil = {}
    for sv in SEUILS_TABLEAU:
        m_s = scores >= sv
        vp_s, fp_s = int(tp[m_s].sum()), int((~tp[m_s]).sum())
        p_s, r_s = vp_s / max(vp_s + fp_s, 1), vp_s / max(n_gt, 1)
        par_seuil[str(sv)] = {"precision": round(p_s, 4), "rappel": round(r_s, 4),
                              "f1": round(2 * p_s * r_s / max(p_s + r_s, 1e-9), 4),
                              "faux_positifs": fp_s, "visages_non_trouves": n_gt - vp_s}

    gt_attrs = np.concatenate(r["gt_attrs"]) if r["gt_attrs"] else np.zeros((0, 6))
    par_attribut = {}
    for nom_a, libelles in LIBELLES_ATTRIBUTS.items():
        col = gt_attrs[:, NOMS_ATTRIBUTS.index(nom_a)].astype(int) if len(gt_attrs) else np.zeros(0, int)
        par_attribut[nom_a] = {}
        for val, lib in libelles.items():
            sel = col == val
            n = int(sel.sum())
            par_attribut[nom_a][lib] = {"visages": n, "rappel": float((gt_score[sel] >= seuil).mean()) if n else None}

    temps = np.array(r["temps"])
    temps = temps[3:] if len(temps) > 6 else temps            # on écarte les premières images (initialisation)
    resume = {
        "AP": round(ap, 4), "seuil": seuil, "precision": round(precision, 4), "rappel": round(rappel, 4),
        "f1": round(f1, 4), "vrais_positifs": vp, "faux_positifs": fp, "faux_negatifs": n_gt - vp,
        "visages_evalues": int(n_gt), "meilleur_f1": round(meilleur_f1, 4), "seuil_meilleur_f1": round(seuil_f1, 3),
        "rappel_par_taille": tranches,
        "par_seuil": par_seuil,
        "rappel_par_attribut": par_attribut,
        "ms_par_image": round(float(temps.mean() * 1000), 2) if len(temps) else None,
    }
    return resume, (rap_c, pre_c)


def evaluer(detecteurs, echantillons, taille_min=20, iou_min=0.5, seuil=0.5, afficher_tous=200):
    """
    detecteurs : dict nom -> objet avec .detecter(image_bgr) -> (N, 5)
    Renvoie (résumés par modèle, courbes PR par modèle, données brutes par image pour l'analyse des échecs).
    """
    res = {nom: {"scores": [], "tp": [], "n_gt": 0, "gt_score": [], "gt_haut": [], "gt_attrs": [], "temps": [], "images": []}
           for nom in detecteurs}
    for k, e in enumerate(echantillons):
        img = cv2.imread(str(e.chemin))
        if img is None:
            continue
        larg, haut = e.boites[:, 2] - e.boites[:, 0], e.boites[:, 3] - e.boites[:, 1]
        ignore = (e.attrs[:, IDX_INVALID] == 1) | (np.minimum(larg, haut) < taille_min)
        for nom, det in detecteurs.items():
            t0 = time.perf_counter()
            dets = det.detecter(img)
            dt = time.perf_counter() - t0
            dets, statut, gt_score = apparier_image(dets, e.boites, ignore, iou_min)
            r = res[nom]
            garde = statut >= 0
            r["scores"].append(dets[garde, 4])
            r["tp"].append(statut[garde] == 1)
            r["n_gt"] += int((~ignore).sum())
            r["gt_score"].append(gt_score[~ignore])
            r["gt_haut"].append(haut[~ignore])
            r["gt_attrs"].append(e.attrs[~ignore])
            r["temps"].append(dt)
            manques = int(((~ignore) & (haut >= 40) & (gt_score < seuil)).sum())   # visages >= 40 px non trouvés
            r["images"].append((k, manques, ignore, dets, statut, gt_score))
        if afficher_tous and (k + 1) % afficher_tous == 0:
            print(f"  {k + 1}/{len(echantillons)} images evaluees", flush=True)

    resumes, courbes = {}, {}
    for nom, r in res.items():
        resumes[nom], courbes[nom] = resumer_modele(r, seuil, taille_min)
    return resumes, courbes, {nom: r["images"] for nom, r in res.items()}
