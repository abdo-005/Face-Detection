"""
modele_mobilenet.py : SSDLite avec réseau de base MobileNetV3 (torchvision), adapté aux visages.

Principe du fine-tuning :
  - le réseau de base (MobileNetV3) et ses poids viennent d'un modèle déjà entraîné (COCO) ;
  - la tête de détection (classification + régression des boîtes) est recréée pour 2 classes
    (0 = fond, 1 = visage) et sera apprise sur WIDER FACE ;
  - les boîtes par défaut (anchor boxes) sont plus petites que celles de COCO, car les visages
    de WIDER FACE sont petits ;
  - l'image est donnée au réseau en taille x taille (par défaut 640) au lieu de 320, pour mieux
    voir les petits visages.
"""
from functools import partial

import torch
from torch import nn
from torchvision.models.detection import (SSDLite320_MobileNet_V3_Large_Weights,
                                          ssdlite320_mobilenet_v3_large)
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssdlite import SSDLiteHead

# Taille relative des boîtes par défaut pour les 6 cartes de caractéristiques (7 valeurs).
# Exemple avec une image 640 : la plus petite boîte fait 0.025 x 640 = 16 pixels.
ECHELLES_ANCRES = [0.025, 0.06, 0.15, 0.30, 0.55, 0.80, 1.0]


def _uniformiser_bn(modele):
    """Même epsilon / momentum pour toutes les couches BatchNorm (entraînement et inférence)."""
    for m in modele.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 0.001
            m.momentum = 0.03


def construire_modele(taille=640, pretrained=True):
    """
    Construit le détecteur de visages.
    pretrained=True  : télécharge les poids COCO (pour le fine-tuning, nécessite Internet).
    pretrained=False : poids aléatoires, à remplir ensuite avec load_state_dict (inférence).
    """
    poids = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    modele = ssdlite320_mobilenet_v3_large(weights=poids, weights_backbone=None)

    # 1) Boîtes par défaut adaptées aux visages
    ancres = DefaultBoxGenerator([[2, 3]] * 6, scales=ECHELLES_ANCRES)
    modele.anchor_generator = ancres

    # 2) Nouvelle tête : 2 classes (fond, visage)
    canaux = det_utils.retrieve_out_channels(modele.backbone, (320, 320))
    norm = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    modele.head = SSDLiteHead(canaux, ancres.num_anchors_per_location(), 2, norm)

    # 3) Taille d'entrée fixe taille x taille
    modele.transform.min_size = (taille,)
    modele.transform.max_size = taille
    modele.transform.fixed_size = (taille, taille)

    # 4) On garde presque toutes les détections : le seuil de confiance se règle à l'usage
    modele.score_thresh = 0.01

    _uniformiser_bn(modele)
    return modele


def sauvegarder(modele, chemin, taille, **infos):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": modele.state_dict(), "taille": taille, **infos}, chemin)


def charger_modele(chemin, device="cpu"):
    """Recharge un modèle fine-tuné sauvegardé par train_mobilenet.py."""
    ckpt = torch.load(chemin, map_location=device)
    modele = construire_modele(ckpt["taille"], pretrained=False)
    modele.load_state_dict(ckpt["model"])
    return modele.to(device).eval()
