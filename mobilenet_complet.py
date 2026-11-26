"""
mobilenet_complet.py : TOUT le programme MobileNet dans un seul fichier.

Détecteur de visages SSDLite + MobileNetV3, fine-tuné sur WIDER FACE, avec :
  1. lecture et préparation du dataset WIDER FACE ;
  2. construction du modèle (poids COCO, nouvelle tête à 2 classes, ancres adaptées aux visages) ;
  3. entraînement (fine-tuning) ;
  4. évaluation sur WIDER_val et visualisation des métriques ;
  5. détection / floutage sur une image, un dossier, une vidéo ou la webcam.

Ce fichier ne dépend d'aucun autre script du projet. Il faut seulement :
    pip install opencv-python numpy matplotlib torch torchvision

UTILISATION (trois commandes) :

    python mobilenet_complet.py entrainer --test-rapide          # vérifie le pipeline (1 à 2 minutes)
    python mobilenet_complet.py entrainer                        # fine-tuning complet (~1 h sur GPU)

    python mobilenet_complet.py evaluer --max-images 300         # métriques + figures (~1 minute)
    python mobilenet_complet.py evaluer                          # tout WIDER_val (quelques minutes)

    python mobilenet_complet.py detecter                         # 20 images de WIDER_test
    python mobilenet_complet.py detecter --source 0              # webcam (q pour quitter)
    python mobilenet_complet.py detecter --source photo.jpg --flouter

Poids utilisés par « evaluer » et « detecter » : ceux de ce script (sorties/mobilenet_complet/meilleur.pt)
s'ils existent, sinon ceux déjà entraînés (sorties/modeles/meilleur.pt). Option --poids pour en choisir d'autres.
Le dataset est cherché dans « Wider Face/ » à côté de ce fichier, ou dans la variable d'environnement WIDER_FACE.
"""
import argparse
import csv
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# =====================================================================================================
# 1. CHEMINS
# =====================================================================================================
DOSSIER = Path(__file__).resolve().parent
SORTIE = DOSSIER / "sorties" / "mobilenet_complet"          # tout ce que produit ce script
POIDS_EXISTANTS = DOSSIER / "sorties" / "modeles" / "meilleur.pt"   # modèle déjà entraîné du projet


def _premier_existant(candidats, test=lambda p: p.exists()):
    for c in candidats:
        if test(c):
            return c
    return candidats[0]


def racine_wider():
    candidats = [DOSSIER / "Wider Face", Path.home() / "Downloads" / "Wider Face"]
    if os.environ.get("WIDER_FACE"):
        candidats.insert(0, Path(os.environ["WIDER_FACE"]))
    return _premier_existant(candidats)


def dossier_images(nom, dossier=None):
    """
    Dossier qui contient directement 0--Parade, 1--Handshaking... (gère les dossiers imbriqués).
    dossier : chemin explicite (ex. ".../Wider Face/WIDER_val"), sinon <racine WIDER>/<nom>.
    """
    d = Path(dossier) if dossier else racine_wider() / nom
    candidats = [d, d / "images", d / d.name / "images", d / "Dataset", d / d.name / "Dataset"]
    return _premier_existant(candidats, lambda p: (p / "0--Parade").exists())


def fichier_annotations(nom):
    r = racine_wider()
    return _premier_existant([r / "wider_face_split" / "wider_face_split" / nom, r / "wider_face_split" / nom])


def poids_par_defaut():
    propre = SORTIE / "meilleur.pt"
    return propre if propre.exists() else POIDS_EXISTANTS


# =====================================================================================================
# 2. DONNÉES WIDER FACE
# =====================================================================================================
# Format d'une ligne de visage : x y largeur hauteur blur expression illumination invalid occlusion pose
NOMS_ATTRIBUTS = ["blur", "expression", "illumination", "invalid", "occlusion", "pose"]
IDX_INVALID = NOMS_ATTRIBUTS.index("invalid")


@dataclass
class Echantillon:
    chemin: Path            # image
    boites: np.ndarray      # (N, 4) x_min, y_min, x_max, y_max en pixels d'origine
    attrs: np.ndarray       # (N, 6) attributs dans l'ordre de NOMS_ATTRIBUTS


def lire_annotations(fichier, dossier):
    """Lit un fichier d'annotations WIDER FACE. Une image sans visage est suivie d'une ligne factice de zéros."""
    with open(fichier, encoding="utf-8") as f:
        lignes = [l.strip() for l in f if l.strip()]
    resultat, i = [], 0
    while i < len(lignes):
        nom, n = lignes[i], int(lignes[i + 1])
        i += 2
        bloc = lignes[i:i + max(n, 1)]
        i += max(n, 1)
        if n == 0:
            boites, attrs = np.zeros((0, 4), np.float32), np.zeros((0, 6), np.float32)
        else:
            a = np.array([[float(v) for v in l.split()] for l in bloc], np.float32)
            boites = np.stack([a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]], axis=1)   # xywh -> xyxy
            attrs = a[:, 4:10]
        resultat.append(Echantillon(Path(dossier) / nom, boites, attrs))
    return resultat


def filtrer(echantillons, taille_min):
    """Garde seulement les visages valides et assez grands, et les images qui en ont au moins un."""
    garde = []
    for e in echantillons:
        l, h = e.boites[:, 2] - e.boites[:, 0], e.boites[:, 3] - e.boites[:, 1]
        m = (e.attrs[:, IDX_INVALID] == 0) & (l >= taille_min) & (h >= taille_min)
        if m.any():
            garde.append(Echantillon(e.chemin, e.boites[m], e.attrs[m]))
    return garde


def preparer(image_bgr, boites, taille, augmenter=False):
    """
    Image -> carré taille x taille en RGB ; boîtes mises à la même échelle.
    Augmentation (entraînement seulement) : retournement horizontal, contraste et luminosité aléatoires.
    """
    h, w = image_bgr.shape[:2]
    img = cv2.cvtColor(cv2.resize(image_bgr, (taille, taille), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    b = np.asarray(boites, np.float32).reshape(-1, 4) * np.array([taille / w, taille / h] * 2, np.float32)
    b = np.clip(b, 0, taille)
    b = b[((b[:, 2] - b[:, 0]) >= 1) & ((b[:, 3] - b[:, 1]) >= 1)]
    if augmenter:
        if random.random() < 0.5:
            img = img[:, ::-1]
            b = np.stack([taille - b[:, 2], b[:, 1], taille - b[:, 0], b[:, 3]], axis=1) if len(b) else b
        img = np.clip(img.astype(np.float32) * random.uniform(0.7, 1.3) + random.uniform(-25, 25), 0, 255)
        img = img.astype(np.uint8)
    return np.ascontiguousarray(img), b.astype(np.float32)


# =====================================================================================================
# 3. MODÈLE : SSDLite + MobileNetV3
# =====================================================================================================
# Taille relative des boîtes par défaut (ancres) sur les 6 cartes de caractéristiques.
# Avec une entrée de 640 px, la plus petite ancre fait 0,025 x 640 = 16 px : adaptée aux petits visages.
ECHELLES_ANCRES = [0.025, 0.06, 0.15, 0.30, 0.55, 0.80, 1.0]


def construire_modele(taille=640, pretrained=True):
    import torch.nn as nn
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
    from torchvision.models.detection import _utils as det_utils
    from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
    from torchvision.models.detection.ssdlite import SSDLiteHead

    poids = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None   # poids COCO
    modele = ssdlite320_mobilenet_v3_large(weights=poids, weights_backbone=None)

    ancres = DefaultBoxGenerator([[2, 3]] * 6, scales=ECHELLES_ANCRES)              # ancres pour visages
    modele.anchor_generator = ancres
    canaux = det_utils.retrieve_out_channels(modele.backbone, (320, 320))
    norm = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    modele.head = SSDLiteHead(canaux, ancres.num_anchors_per_location(), 2, norm)  # 2 classes : fond, visage

    modele.transform.min_size, modele.transform.max_size = (taille,), taille        # entrée taille x taille
    modele.transform.fixed_size = (taille, taille)
    modele.score_thresh = 0.01        # garde presque tout : le seuil de confiance se choisit à l'usage
    for m in modele.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps, m.momentum = 0.001, 0.03
    return modele


def sauvegarder(modele, chemin, taille, **infos):
    import torch
    chemin.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": modele.state_dict(), "taille": taille, **infos}, chemin)


def charger_modele(chemin, device):
    import torch
    ckpt = torch.load(chemin, map_location=device)
    modele = construire_modele(ckpt["taille"], pretrained=False)
    modele.load_state_dict(ckpt["model"])
    return modele.to(device).eval()


class Detecteur:
    """Enveloppe le modèle : image BGR (OpenCV) -> tableau (N, 5) x1, y1, x2, y2, confiance (pixels d'origine)."""

    def __init__(self, chemin_poids, device=None):
        import torch
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if not Path(chemin_poids).exists():
            raise SystemExit(f"Poids introuvables : {chemin_poids}. Lancez d'abord : python mobilenet_complet.py entrainer")
        self.modele = charger_modele(chemin_poids, self.device)
        self.taille = self.modele.transform.fixed_size[0]
        self.nom = f"MobileNet ({self.device.type.upper()})"

    def detecter(self, image_bgr, seuil_min=0.01):
        h, w = image_bgr.shape[:2]
        img, _ = preparer(image_bgr, np.zeros((0, 4)), self.taille)
        x = self.torch.from_numpy(img).permute(2, 0, 1).to(self.device).float().div_(255.0)
        with self.torch.no_grad():
            s = self.modele([x])[0]
        boites = s["boxes"].cpu().numpy() * np.array([w / self.taille, h / self.taille] * 2, np.float32)
        scores = s["scores"].cpu().numpy()
        garde = scores >= seuil_min
        boites, scores = boites[garde], scores[garde]
        boites[:, [0, 2]] = np.clip(boites[:, [0, 2]], 0, w)
        boites[:, [1, 3]] = np.clip(boites[:, [1, 3]], 0, h)
        ok = ((boites[:, 2] - boites[:, 0]) >= 1) & ((boites[:, 3] - boites[:, 1]) >= 1)
        return np.concatenate([boites[ok], scores[ok, None]], axis=1).astype(np.float32)


# =====================================================================================================
# 4. ENTRAÎNEMENT (fine-tuning)
# =====================================================================================================
class WiderDataset(Dataset):
    """Une image + ses boîtes de visages, au format attendu par torchvision (classe 1 = visage)."""

    def __init__(self, echantillons, taille, augmenter):
        self.e, self.taille, self.augmenter = echantillons, taille, augmenter

    def __len__(self):
        return len(self.e)

    def __getitem__(self, i):
        image = cv2.imread(str(self.e[i].chemin))
        img, b = preparer(image, self.e[i].boites, self.taille, self.augmenter)
        cible = {"boxes": torch.from_numpy(b).reshape(-1, 4), "labels": torch.ones(len(b), dtype=torch.int64)}
        return torch.from_numpy(img).permute(2, 0, 1).contiguous(), cible   # uint8 : converti sur le GPU


def regrouper(lot):
    """Chaque image a un nombre de visages différent : on garde des listes au lieu d'un tenseur unique."""
    return tuple(zip(*lot))


def entrainer(a):
    from torch.utils.data import DataLoader

    if a.test_rapide:
        a.max_images, a.epochs, a.batch, a.workers, a.gel = 64, 1, 8, 0, 0
    warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
    random.seed(a.graine)
    np.random.seed(a.graine)
    torch.manual_seed(a.graine)
    cv2.setNumThreads(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"            # précision mixte : plus rapide et moins de mémoire sur GPU
    print(f"Peripherique : {device}" + (f" ({torch.cuda.get_device_name(0)})" if amp else " - ATTENTION : tres lent sur CPU"))
    if amp:
        torch.backends.cudnn.benchmark = True

    # --- Données : WIDER_train, 95 % entraînement / 5 % mini-validation ------------------------------
    annot, images = fichier_annotations("wider_face_train_bbx_gt.txt"), dossier_images("WIDER_train")
    if not annot.exists():
        raise SystemExit(f"Annotations introuvables : {annot}. Placez le dataset dans '{DOSSIER / 'Wider Face'}'.")
    ech = [e for e in filtrer(lire_annotations(annot, images), a.taille_min) if e.chemin.exists()]
    if not ech:
        raise SystemExit(f"Aucune image trouvee dans {images}")
    rng = random.Random(a.graine)
    rng.shuffle(ech)
    if a.max_images:
        ech = ech[:a.max_images]
    n_val = max(1, int(len(ech) * 0.05))
    train, val = ech[n_val:], ech[:n_val]
    print(f"Entrainement : {len(train)} images | mini-validation : {len(val)} images")

    options = dict(batch_size=a.batch, num_workers=a.workers, collate_fn=regrouper, pin_memory=amp,
                   persistent_workers=a.workers > 0)
    l_train = DataLoader(WiderDataset(train, a.taille, True), shuffle=True, drop_last=True, **options)
    l_val = DataLoader(WiderDataset(val, a.taille, False), shuffle=False, **options)

    # --- Modèle, optimiseur, taux d'apprentissage --------------------------------------------------
    modele = construire_modele(a.taille, pretrained=True).to(device)
    optimiseur = torch.optim.AdamW([{"params": modele.backbone.parameters(), "lr": a.lr_base},
                                    {"params": modele.head.parameters(), "lr": a.lr_tete}], weight_decay=1e-4)
    total = max(1, len(l_train) * a.epochs)
    chauffe = min(200, total // 10 + 1)

    def facteur(it):                       # montée progressive (warm-up) puis décroissance en cosinus
        if it < chauffe:
            return (it + 1) / chauffe
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (it - chauffe) / max(1, total - chauffe)))

    planif = torch.optim.lr_scheduler.LambdaLR(optimiseur, facteur)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    @torch.no_grad()
    def perte_validation():
        modele.train()                     # en mode train, torchvision renvoie les pertes...
        for m in modele.modules():         # ...mais on fige les BatchNorm pour ne pas les modifier
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
        s, n = 0.0, 0
        for imgs, cibles in l_val:
            imgs = [x.to(device).float().div_(255.0) for x in imgs]
            cibles = [{k: v.to(device) for k, v in c.items()} for c in cibles]
            s += sum(modele(imgs, cibles).values()).item() * len(imgs)
            n += len(imgs)
        return s / max(n, 1)

    # le test rapide écrit à part, pour ne jamais remplacer un vrai modèle par un modèle à peine entraîné
    dossier = SORTIE / "test_rapide" if a.test_rapide else SORTIE
    dossier.mkdir(parents=True, exist_ok=True)
    journal, meilleure = [], float("inf")
    for epoque in range(1, a.epochs + 1):
        gele = epoque <= a.gel             # réseau de base gelé au début : seule la nouvelle tête apprend
        for p in modele.backbone.parameters():
            p.requires_grad = not gele
        modele.train()
        t0, somme, n = time.time(), 0.0, 0
        for it, (imgs, cibles) in enumerate(l_train, 1):
            imgs = [x.to(device, non_blocking=True).float().div_(255.0) for x in imgs]
            cibles = [{k: v.to(device, non_blocking=True) for k, v in c.items()} for c in cibles]
            with torch.autocast(device_type=device.type, enabled=amp):
                perte = sum(modele(imgs, cibles).values())          # classification + localisation
            optimiseur.zero_grad(set_to_none=True)
            scaler.scale(perte).backward()                          # rétropropagation
            scaler.unscale_(optimiseur)
            torch.nn.utils.clip_grad_norm_(modele.parameters(), 10.0)
            scaler.step(optimiseur)
            scaler.update()
            planif.step()
            somme, n = somme + perte.item() * len(imgs), n + len(imgs)
            if it % 50 == 0 or it == len(l_train):
                print(f"  epoque {epoque}/{a.epochs} - lot {it}/{len(l_train)} - perte {somme / n:.4f}", flush=True)

        p_train, p_val = somme / max(n, 1), perte_validation()
        duree = time.time() - t0
        print(f"Epoque {epoque}/{a.epochs} | perte train {p_train:.4f} | perte val {p_val:.4f} | {duree:.0f}s"
              + (" | reseau de base gele" if gele else ""))
        journal.append({"epoque": epoque, "perte_train": round(p_train, 5), "perte_val": round(p_val, 5),
                        "duree_s": round(duree, 1)})
        sauvegarder(modele, dossier / "dernier.pt", a.taille, epoque=epoque, perte_val=p_val)
        if p_val < meilleure:              # on garde le modèle qui généralise le mieux
            meilleure = p_val
            sauvegarder(modele, dossier / "meilleur.pt", a.taille, epoque=epoque, perte_val=p_val)
            print("  -> nouveau meilleur modele")
        with open(dossier / "journal_entrainement.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(journal[0]))
            w.writeheader()
            w.writerows(journal)
        figure_pertes(journal, dossier / "courbes_perte.png")
    print(f"Termine. Meilleur modele : {dossier / 'meilleur.pt'} (perte val {meilleure:.4f})")


# =====================================================================================================
# 5. MÉTRIQUES
# =====================================================================================================
TRANCHES = [("petits", 0, 40), ("moyens", 40, 100), ("grands", 100, 1e9)]      # hauteur du visage (px)
LIBELLES = {"blur": ["net", "flou moyen", "flou fort"], "occlusion": ["aucune", "partielle", "forte"],
            "pose": ["typique", "atypique"], "illumination": ["normale", "extreme"]}


def iou(boite, boites):
    """Intersection sur union entre une boîte et plusieurs boîtes (0 = disjointes, 1 = identiques)."""
    x1, y1 = np.maximum(boite[0], boites[:, 0]), np.maximum(boite[1], boites[:, 1])
    x2, y2 = np.minimum(boite[2], boites[:, 2]), np.minimum(boite[3], boites[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aire = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
    return inter / np.maximum(aire(boite) + aire(boites) - inter, 1e-9)


def apparier(dets, gt, ignore, iou_min=0.5):
    """
    Associe chaque détection (de la plus confiante à la moins confiante) au vrai visage le plus recouvert.
    statut : 1 = vrai positif, 0 = faux positif, -1 = ignorée (visage invalide ou trop petit).
    gt_score[j] : confiance de la détection qui a trouvé le visage j (-1 s'il n'a pas été trouvé).
    """
    dets = dets[np.argsort(-dets[:, 4])] if len(dets) else dets
    statut, pris, gt_score = np.zeros(len(dets), int), np.zeros(len(gt), bool), np.full(len(gt), -1.0)
    if len(gt):
        for i, d in enumerate(dets):
            ious = iou(d[:4], gt)
            j = int(np.argmax(ious))
            if ious[j] >= iou_min:
                if ignore[j]:
                    statut[i] = -1
                elif not pris[j]:                  # un visage ne peut être trouvé qu'une fois
                    statut[i], pris[j], gt_score[j] = 1, True, d[4]
    return dets, statut, gt_score


def courbe_pr(scores, vp, n_gt):
    """Courbe précision-rappel et AP (aire sous la courbe, enveloppe décroissante)."""
    if len(scores) == 0:
        return np.zeros(0), np.zeros(0), 0.0, np.zeros(0)
    ordre = np.argsort(-scores)
    vp = vp[ordre].astype(float)
    cvp, cfp = np.cumsum(vp), np.cumsum(1 - vp)
    rappel, precision = cvp / max(n_gt, 1), cvp / np.maximum(cvp + cfp, 1)
    mr = np.concatenate([[0.0], rappel, [rappel[-1]]])
    mp = np.maximum.accumulate(np.concatenate([[1.0], precision, [0.0]])[::-1])[::-1]
    idx = np.where(mr[1:] != mr[:-1])[0]
    return rappel, precision, float(np.sum((mr[idx + 1] - mr[idx]) * mp[idx + 1])), scores[ordre]


def comptes(scores, vp, gt_score, seuil):
    """VP, FP, FN au seuil de confiance donné."""
    m = scores >= seuil
    return int((vp & m).sum()), int((~vp & m).sum()), int((gt_score < seuil).sum())


# =====================================================================================================
# 6. ÉVALUATION + VISUALISATION
# =====================================================================================================
VERT, ROUGE, ORANGE, GRIS = (0, 200, 0), (0, 0, 255), (0, 140, 255), (160, 160, 160)   # BGR


def decouper(image, b, cote=96):
    h, w = image.shape[:2]
    mx, my = (b[2] - b[0]) * 0.25, (b[3] - b[1]) * 0.25
    x1, y1, x2, y2 = int(max(0, b[0] - mx)), int(max(0, b[1] - my)), int(min(w, b[2] + mx)), int(min(h, b[3] + my))
    return cv2.resize(image[y1:y2, x1:x2], (cote, cote)) if x2 - x1 > 1 and y2 - y1 > 1 else None


def evaluer(a):
    import matplotlib
    matplotlib.use("Agg")

    images = dossier_images("WIDER_val", a.val)
    annot = Path(a.annotations) if a.annotations else fichier_annotations("wider_face_val_bbx_gt.txt")
    if not (images / "0--Parade").exists():
        raise SystemExit(f"Images WIDER_val introuvables dans : {a.val or racine_wider() / 'WIDER_val'}")
    if not annot.exists():
        raise SystemExit(f"Annotations introuvables : {annot} (option --annotations)")
    print(f"Images      : {images}\nAnnotations : {annot}")
    ech = [e for e in lire_annotations(annot, images) if e.chemin.exists()]
    random.Random(42).shuffle(ech)
    if a.max_images:
        ech = ech[:a.max_images]
    det = Detecteur(a.poids or poids_par_defaut(), a.device)
    sortie = SORTIE / "evaluation"
    (sortie / "images").mkdir(parents=True, exist_ok=True)
    print(f"Evaluation de {det.nom} sur {len(ech)} images de WIDER_val (seuil {a.seuil}, IoU {a.iou}, "
          f"visages < {a.taille_min} px ou invalides ignores)")

    scores, vps, gt_scores, gt_hauts, gt_attrs, temps = [], [], [], [], [], []
    galerie = {"vp": [], "fp": [], "fn": []}
    n_images_annotees, rng = 0, random.Random(0)
    for k, e in enumerate(ech):
        image = cv2.imread(str(e.chemin))
        if image is None:
            continue
        l, h = e.boites[:, 2] - e.boites[:, 0], e.boites[:, 3] - e.boites[:, 1]
        ignore = (e.attrs[:, IDX_INVALID] == 1) | (np.minimum(l, h) < a.taille_min)
        t0 = time.perf_counter()
        dets = det.detecter(image)
        temps.append(time.perf_counter() - t0)
        dets, statut, gt_score = apparier(dets, e.boites, ignore, a.iou)

        garde = statut >= 0
        scores.append(dets[garde, 4])
        vps.append(statut[garde] == 1)
        gt_scores.append(gt_score[~ignore])
        gt_hauts.append(h[~ignore])
        gt_attrs.append(e.attrs[~ignore])

        # Exemples pour la visualisation, au seuil choisi
        au_seuil = dets[:, 4] >= a.seuil if len(dets) else np.zeros(0, bool)
        groupes = {"vp": dets[(statut == 1) & au_seuil, :4], "fp": dets[(statut == 0) & au_seuil, :4],
                   "fn": e.boites[(~ignore) & (gt_score < a.seuil)]}
        for cle, boites in groupes.items():
            for b in boites:
                if len(galerie[cle]) < a.galerie and rng.random() < (0.8 if cle == "fp" else 0.15):
                    c = decouper(image, b)
                    if c is not None:
                        galerie[cle].append(c)
        if n_images_annotees < 6 and len(groupes["fp"]) + len(groupes["fn"]) > 0 and len(e.boites) <= 40:
            vue = image.copy()
            for boites, coul in [(e.boites[ignore], GRIS), (groupes["vp"], VERT), (groupes["fn"], ORANGE),
                                 (groupes["fp"], ROUGE)]:
                for b in boites:
                    cv2.rectangle(vue, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), coul, 2)
            cv2.putText(vue, f"VP {len(groupes['vp'])}  FP {len(groupes['fp'])}  FN {len(groupes['fn'])}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            n_images_annotees += 1
            cv2.imwrite(str(sortie / "images" / f"exemple_{n_images_annotees}_{e.chemin.parent.name}.jpg"), vue)
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(ech)} images", flush=True)

    # --- Calcul des métriques ------------------------------------------------------------------------
    scores, vps = np.concatenate(scores), np.concatenate(vps).astype(bool)
    gt_score, gt_haut, gt_attr = np.concatenate(gt_scores), np.concatenate(gt_hauts), np.concatenate(gt_attrs)
    n_gt = len(gt_score)
    rappel_c, precision_c, ap, _ = courbe_pr(scores, vps, n_gt)
    vp, fp, fn = comptes(scores, vps, gt_score, a.seuil)
    precision, rappel = vp / max(vp + fp, 1), vp / max(n_gt, 1)
    f1 = 2 * precision * rappel / max(precision + rappel, 1e-9)
    seuils = np.round(np.arange(0.05, 0.96, 0.05), 2)
    par_seuil = []
    for s in seuils:
        v, f, n = comptes(scores, vps, gt_score, s)
        p, r = v / max(v + f, 1), v / max(n_gt, 1)
        par_seuil.append({"seuil": float(s), "VP": v, "FP": f, "FN": n, "precision": round(p, 4),
                          "rappel": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4)})
    meilleur = max(par_seuil, key=lambda d: d["f1"])
    par_taille = {}
    for nom, lo, hi in TRANCHES:
        sel = (gt_haut >= max(lo, a.taille_min)) & (gt_haut < hi)
        par_taille[nom] = {"visages": int(sel.sum()), "rappel": float((gt_score[sel] >= a.seuil).mean()) if sel.any() else 0.0}
    par_attribut = {}
    for attr, libs in LIBELLES.items():
        col = gt_attr[:, NOMS_ATTRIBUTS.index(attr)].astype(int)
        par_attribut[attr] = {lib: {"visages": int((col == v).sum()),
                                    "rappel": float((gt_score[col == v] >= a.seuil).mean()) if (col == v).any() else 0.0}
                              for v, lib in enumerate(libs)}
    ms = float(np.mean(temps[3:] if len(temps) > 6 else temps) * 1000)
    resultats = {"modele": det.nom, "images": len(ech), "seuil": a.seuil, "iou": a.iou, "AP": round(ap, 4),
                 "precision": round(precision, 4), "rappel": round(rappel, 4), "f1": round(f1, 4),
                 "VP": vp, "FP": fp, "FN": fn, "visages_evalues": n_gt, "meilleur_seuil_f1": meilleur,
                 "ms_par_image": round(ms, 2), "rappel_par_taille": par_taille, "rappel_par_attribut": par_attribut}
    with open(sortie / "resultats.json", "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)
    with open(sortie / "par_seuil.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(par_seuil[0]))
        w.writeheader()
        w.writerows(par_seuil)

    # --- Figures ---------------------------------------------------------------------------------------
    figure_tableau_de_bord(resultats, rappel_c, precision_c, par_seuil, sortie / "tableau_de_bord.png")
    figure_galerie(galerie, det.nom, a.galerie, sortie / "galerie_vp_fp_fn.png")
    journal = SORTIE / "journal_entrainement.csv"
    if not journal.exists():
        journal = POIDS_EXISTANTS.parent / "journal_entrainement.csv"
    if journal.exists():
        with open(journal, encoding="utf-8") as f:
            lignes = [{k: float(v) for k, v in l.items()} for l in csv.DictReader(f)]
        figure_pertes(lignes, sortie / "courbes_perte.png")

    print("\n" + "=" * 72)
    print(f"{det.nom} - {len(ech)} images, {n_gt} visages evalues")
    print(f"AP {ap:.3f} | precision {precision:.3f} | rappel {rappel:.3f} | F1 {f1:.3f}   (seuil {a.seuil})")
    print(f"VP {vp} | FP {fp} | FN {fn} | meilleur F1 {meilleur['f1']:.3f} au seuil {meilleur['seuil']}")
    print("Rappel par taille : " + " | ".join(f"{k} {v['rappel']:.2f}" for k, v in par_taille.items()))
    print(f"Vitesse : {ms:.1f} ms par image ({1000 / ms:.0f} images/s)")
    print("=" * 72)
    print(f"Figures et resultats dans : {sortie}")


def figure_pertes(journal, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cle = "epoque" if "epoque" in journal[0] else "epoch"
    ep = [l[cle] for l in journal]
    plt.figure(figsize=(7, 4))
    plt.plot(ep, [l["perte_train"] for l in journal], "o-", label="Entrainement")
    plt.plot(ep, [l["perte_val"] for l in journal], "o-", label="Mini-validation")
    plt.xlabel("Epoque")
    plt.ylabel("Perte (classification + localisation)")
    plt.title("Courbes de perte du fine-tuning")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    plt.close()


def figure_tableau_de_bord(r, rappel_c, precision_c, par_seuil, chemin):
    """Six graphiques : courbe PR, précision/rappel/F1 selon le seuil, matrice de confusion,
    VP/FP/FN selon le seuil, rappel selon la taille, rappel selon le flou / l'occlusion / la pose."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(19, 10.5))
    s = [d["seuil"] for d in par_seuil]

    a = ax[0, 0]                                                        # 1. courbe précision-rappel
    a.plot(rappel_c, precision_c, color="#1f4e9e", lw=2)
    a.fill_between(rappel_c, precision_c, alpha=0.15, color="#1f4e9e")
    a.scatter([r["rappel"]], [r["precision"]], color="red", zorder=3, label=f"seuil {r['seuil']}")
    a.set(xlabel="Rappel", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
          title=f"Courbe precision-rappel (AP = {r['AP']:.3f})")
    a.grid(alpha=0.3)
    a.legend()

    a = ax[0, 1]                                                        # 2. effet du seuil
    for cle, coul in [("precision", "#1f4e9e"), ("rappel", "#e07b00"), ("f1", "#2e8b57")]:
        a.plot(s, [d[cle] for d in par_seuil], "o-", color=coul, ms=4, label=cle.capitalize())
    a.axvline(r["meilleur_seuil_f1"]["seuil"], color="#2e8b57", ls=":", label="meilleur F1")
    a.axvline(r["seuil"], color="gray", ls="--", label="seuil choisi")
    a.set(xlabel="Seuil de confiance", ylim=(0, 1.02), title="Precision, rappel et F1 selon le seuil")
    a.grid(alpha=0.3)
    a.legend(fontsize=8)

    a = ax[0, 2]                                                        # 3. matrice de confusion
    vp, fp, fn = r["VP"], r["FP"], r["FN"]
    val = np.array([[vp / max(vp + fn, 1), fn / max(vp + fn, 1)], [fp / max(vp + fp, 1), 0.0]])
    a.imshow(val, cmap="Blues", vmin=0, vmax=1)
    txt = [[f"VP\n{vp}\n({val[0, 0]:.1%} des visages)", f"FN\n{fn}\n({val[0, 1]:.1%} des visages)"],
           [f"FP\n{fp}\n({val[1, 0]:.1%} des detections)", "VN\nnon defini\n(infinite de zones\nsans visage)"]]
    for i in range(2):
        for j in range(2):
            a.text(j, i, txt[i][j], ha="center", va="center", fontsize=10, color="white" if val[i, j] > 0.55 else "black")
    a.set_xticks([0, 1], ["Visage", "Pas de visage"])
    a.set_yticks([0, 1], ["Visage", "Fond"])
    a.set(xlabel="Prediction", ylabel="Realite", title=f"Matrice de confusion (seuil {r['seuil']}, IoU {r['iou']})")

    a = ax[1, 0]                                                        # 4. VP / FP / FN selon le seuil
    for cle, coul, lib in [("VP", "green", "VP (trouves)"), ("FP", "red", "FP (fausses alertes)"),
                           ("FN", "darkorange", "FN (rates)")]:
        a.plot(s, [d[cle] for d in par_seuil], "o-", color=coul, ms=4, label=lib)
    a.axvline(r["seuil"], color="gray", ls="--")
    a.set_yscale("log")                    # échelle log : les FP vont de quelques dizaines à des milliers
    a.set(xlabel="Seuil de confiance", ylabel="Nombre (echelle log)", title="VP, FP et FN selon le seuil")
    a.grid(alpha=0.3)
    a.legend(fontsize=8)

    a = ax[1, 1]                                                        # 5. rappel selon la taille
    t = r["rappel_par_taille"]
    noms = [f"{k}\n({v['visages']} visages)" for k, v in t.items()]
    b = a.bar(noms, [v["rappel"] for v in t.values()], color=["#9ecae1", "#4292c6", "#08519c"])
    a.bar_label(b, fmt="%.2f")
    a.set(ylim=(0, 1.1), ylabel="Rappel", title="Rappel selon la taille des visages\n(<40 px, 40-100 px, >100 px)")

    a = ax[1, 2]                                                        # 6. rappel selon les attributs
    x, etiquettes, couleurs = 0, [], {"blur": "#6a51a3", "occlusion": "#cb181d", "pose": "#238b45"}
    for attr in ["blur", "occlusion", "pose"]:
        for lib, v in r["rappel_par_attribut"][attr].items():
            b = a.bar(x, v["rappel"], color=couleurs[attr])
            a.bar_label(b, fmt="%.2f", fontsize=8)
            etiquettes.append(f"{attr}\n{lib}")
            x += 1
        x += 0.5
    pos = [i + 0.5 * (i >= 3) + 0.5 * (i >= 6) for i in range(len(etiquettes))]
    a.set_xticks(pos, etiquettes, fontsize=8)
    a.set(ylim=(0, 1.1), ylabel="Rappel", title="Rappel selon le flou, l'occlusion et la pose")

    fig.suptitle(f"{r['modele']} sur WIDER_val ({r['images']} images, {r['visages_evalues']} visages) - "
                 f"precision {r['precision']:.3f}, rappel {r['rappel']:.3f}, F1 {r['f1']:.3f}, "
                 f"{r['ms_par_image']:.0f} ms/image", fontsize=13)
    fig.tight_layout()
    fig.savefig(chemin, dpi=140)
    plt.close(fig)


def figure_galerie(galerie, nom, n, chemin):
    import matplotlib.pyplot as plt
    lignes = [("vp", "VP : trouves", "green"), ("fp", "FP : fausses\ndetections", "red"), ("fn", "FN : rates", "darkorange")]
    fig, ax = plt.subplots(3, n, figsize=(1.6 * n, 5.6))
    for i, (cle, titre, coul) in enumerate(lignes):
        for j in range(n):
            a = ax[i, j]
            a.set_xticks([])
            a.set_yticks([])
            if j < len(galerie[cle]):
                a.imshow(cv2.cvtColor(galerie[cle][j], cv2.COLOR_BGR2RGB))
                for c in a.spines.values():
                    c.set_edgecolor(coul)
                    c.set_linewidth(3)
            else:
                a.axis("off")
        ax[i, 0].set_ylabel(titre, color=coul, fontsize=9, rotation=0, ha="right", va="center")
    fig.suptitle(f"Exemples de visages - {nom}", fontsize=13)
    fig.tight_layout()
    fig.savefig(chemin, dpi=130, bbox_inches="tight")
    plt.close(fig)


# =====================================================================================================
# 7. DÉTECTION / FLOUTAGE (image, dossier, vidéo, webcam)
# =====================================================================================================
def dessiner(image, dets, flouter):
    sortie = image.copy()
    for x1, y1, x2, y2, conf in dets:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if flouter and x2 - x1 > 1 and y2 - y1 > 1:
            k = max(3, (max(x2 - x1, y2 - y1) // 2) | 1)               # noyau impair proportionnel au visage
            sortie[y1:y2, x1:x2] = cv2.GaussianBlur(sortie[y1:y2, x1:x2], (k, k), 0)
        cv2.rectangle(sortie, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(sortie, f"{conf:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return sortie


def bandeau(image, texte):
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(image, texte, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image


def pour_ecran(image, h_max=720):
    h = image.shape[0]
    return cv2.resize(image, (int(image.shape[1] * h_max / h), h_max)) if h > h_max else image


def detecter(a):
    det = Detecteur(a.poids or poids_par_defaut(), a.device)
    sortie = SORTIE / "detection"
    sortie.mkdir(parents=True, exist_ok=True)
    fenetre = f"{det.nom} (q pour quitter)"

    est_video = a.source is not None and (str(a.source).isdigit() or Path(a.source).suffix.lower() in
                                          (".mp4", ".avi", ".mov", ".mkv"))
    if est_video:                                                      # webcam ou fichier vidéo
        src = int(a.source) if str(a.source).isdigit() else a.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise SystemExit(f"Impossible d'ouvrir la source video : {a.source}")
        ecrivain = None
        if not str(a.source).isdigit():
            taille = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            ecrivain = cv2.VideoWriter(str(sortie / f"{Path(a.source).stem}_resultat.mp4"),
                                       cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 25, taille)
        fps = None
        while True:
            ok, image = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            dets = det.detecter(image, a.seuil)
            fps_i = 1 / max(time.perf_counter() - t0, 1e-6)
            fps = fps_i if fps is None else 0.9 * fps + 0.1 * fps_i       # moyenne glissante
            vue = bandeau(dessiner(image, dets, a.flouter), f"{det.nom} : {len(dets)} visage(s) - {fps:.0f} images/s")
            if ecrivain is not None:
                ecrivain.write(vue)
            if not a.sans_affichage:
                cv2.imshow(fenetre, pour_ecran(vue))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        if ecrivain is not None:
            ecrivain.release()
        cv2.destroyAllWindows()
        print(f"Termine. Resultats dans : {sortie}")
        return

    source = Path(a.source) if a.source else dossier_images("WIDER_test") / "0--Parade"
    if not source.exists():
        raise SystemExit(f"Source introuvable : {source}")
    images = [source] if source.is_file() else sorted(
        p for p in source.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))[:a.max]
    print(f"{det.nom} : {len(images)} image(s), seuil {a.seuil}{', floutage' if a.flouter else ''}")
    total, temps = 0, []
    with open(sortie / "detections.txt", "w", encoding="utf-8") as journal:
        for chemin in images:
            image = cv2.imread(str(chemin))
            if image is None:
                continue
            t0 = time.perf_counter()
            dets = det.detecter(image, a.seuil)
            temps.append(time.perf_counter() - t0)
            total += len(dets)
            vue = bandeau(dessiner(image, dets, a.flouter),
                          f"{det.nom} : {len(dets)} visage(s) - {temps[-1] * 1000:.0f} ms")
            cv2.imwrite(str(sortie / chemin.name), vue)
            journal.write(f"\n--- {chemin.name} : {len(dets)} visage(s) ---\n")
            for i, (x1, y1, x2, y2, c) in enumerate(dets, 1):
                journal.write(f"Visage {i} : confiance {c:.2f}, boite ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})\n")
            print(f"  {chemin.name} : {len(dets)} visage(s) en {temps[-1] * 1000:.0f} ms")
            if not a.sans_affichage:
                cv2.imshow(fenetre, pour_ecran(vue))
                if cv2.waitKey(a.delai) & 0xFF == ord("q"):
                    break
    cv2.destroyAllWindows()
    moyenne = np.mean(temps[1:] if len(temps) > 1 else temps) * 1000 if temps else 0
    print(f"Termine : {total} visage(s), {moyenne:.1f} ms par image en moyenne. Resultats dans : {sortie}")


# =====================================================================================================
# 8. LIGNE DE COMMANDE
# =====================================================================================================
def main():
    p = argparse.ArgumentParser(description="MobileNet (SSDLite) pour la detection de visages : tout-en-un")
    sous = p.add_subparsers(dest="commande", required=True)

    e = sous.add_parser("entrainer", help="fine-tuning sur WIDER_train")
    e.add_argument("--epochs", type=int, default=20)
    e.add_argument("--batch", type=int, default=16, help="images par lot (8 si la memoire GPU manque)")
    e.add_argument("--taille", type=int, default=640, help="taille d'entree du reseau (carre)")
    e.add_argument("--lr-tete", type=float, default=1e-3, help="taux d'apprentissage de la nouvelle tete")
    e.add_argument("--lr-base", type=float, default=2e-4, help="taux d'apprentissage du reseau de base")
    e.add_argument("--gel", type=int, default=2, help="epoques avec le reseau de base gele")
    e.add_argument("--taille-min", type=int, default=16, help="visages plus petits ignores (pixels)")
    e.add_argument("--max-images", type=int, default=None, help="sous-ensemble du train (plus rapide)")
    e.add_argument("--workers", type=int, default=2, help="processus de chargement (0 en cas d'erreur memoire)")
    e.add_argument("--graine", type=int, default=42)
    e.add_argument("--test-rapide", action="store_true", help="64 images, 1 epoque")

    v = sous.add_parser("evaluer", help="metriques et figures sur WIDER_val")
    v.add_argument("--val", default=str(DOSSIER / "Wider Face" / "WIDER_val"),
                   help="dossier WIDER_val (images annotees, jamais vues a l'entrainement)")
    v.add_argument("--annotations", default=None, help="fichier wider_face_val_bbx_gt.txt (defaut : trouve automatiquement)")
    v.add_argument("--poids", default=None)
    v.add_argument("--device", default=None, help="cuda ou cpu (defaut : automatique)")
    v.add_argument("--max-images", type=int, default=None)
    v.add_argument("--seuil", type=float, default=0.5, help="seuil de confiance pour precision / rappel")
    v.add_argument("--iou", type=float, default=0.5)
    v.add_argument("--taille-min", type=int, default=20, help="visages plus petits ignores (pixels)")
    v.add_argument("--galerie", type=int, default=10, help="exemples par categorie (VP, FP, FN)")

    d = sous.add_parser("detecter", help="detection / floutage : image, dossier, video ou webcam")
    d.add_argument("--source", default=None, help="image, dossier, video ou numero de webcam (0)")
    d.add_argument("--poids", default=None)
    d.add_argument("--device", default=None)
    d.add_argument("--seuil", type=float, default=0.25, help="0.25 = meilleur F1 mesure sur WIDER_val")
    d.add_argument("--flouter", action="store_true")
    d.add_argument("--max", type=int, default=20)
    d.add_argument("--delai", type=int, default=1500, help="duree d'affichage par image (ms)")
    d.add_argument("--sans-affichage", action="store_true")

    a = p.parse_args()
    {"entrainer": entrainer, "evaluer": evaluer, "detecter": detecter}[a.commande](a)


if __name__ == "__main__":      # obligatoire sous Windows (processus de chargement des images)
    main()
