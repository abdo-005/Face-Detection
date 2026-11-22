"""
detecteurs.py : deux détecteurs de visages avec la MÊME interface.

    detecteur.detecter(image_bgr, seuil_min=0.01) -> tableau (N, 5) : x_min, y_min, x_max, y_max, confiance
    (coordonnées en pixels de l'image d'origine)

 - DetecteurResNet    : SSD + ResNet-10 d'OpenCV (cv2.dnn), préentraîné, utilisé tel quel
 - DetecteurMobileNet : SSDLite + MobileNetV3, fine-tuné sur WIDER FACE (PyTorch)
"""
import cv2
import numpy as np


class DetecteurResNet:
    nom = "ResNet-10 (OpenCV)"

    def __init__(self, prototxt, caffemodel):
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))

    def detecter(self, image_bgr, seuil_min=0.01):
        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        sortie = self.net.forward()[0, 0]                         # (N, 7) : id, classe, confiance, x1, y1, x2, y2
        sortie = sortie[sortie[:, 2] >= seuil_min]
        boites = sortie[:, 3:7] * np.array([w, h, w, h], np.float32)   # coordonnées relatives -> pixels
        return _nettoyer(boites, sortie[:, 2], w, h)


class DetecteurMobileNet:
    nom = "MobileNet fine-tune"

    def __init__(self, chemin_modele, device=None):
        import torch
        from modele_mobilenet import charger_modele
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.modele = charger_modele(chemin_modele, self.device)
        self.taille = self.modele.transform.fixed_size[0]
        self.nom = f"MobileNet fine-tune ({self.device.type.upper()})"

    def detecter(self, image_bgr, seuil_min=0.01):
        torch = self.torch
        h, w = image_bgr.shape[:2]
        # Même prétraitement qu'à l'entraînement : redimensionnement carré, BGR -> RGB, valeurs dans [0, 1]
        img = cv2.cvtColor(cv2.resize(image_bgr, (self.taille, self.taille), interpolation=cv2.INTER_LINEAR),
                           cv2.COLOR_BGR2RGB)
        tenseur = torch.from_numpy(img).permute(2, 0, 1).to(self.device).float().div_(255.0)
        with torch.no_grad():
            sortie = self.modele([tenseur])[0]
        boites = sortie["boxes"].cpu().numpy() * np.array([w / self.taille, h / self.taille] * 2, np.float32)
        scores = sortie["scores"].cpu().numpy()
        garder = scores >= seuil_min
        return _nettoyer(boites[garder], scores[garder], w, h)


def _nettoyer(boites, scores, w, h):
    """Découpe les boîtes aux limites de l'image, supprime les boîtes vides, renvoie (N, 5)."""
    boites = np.asarray(boites, np.float32).reshape(-1, 4).copy()
    boites[:, [0, 2]] = np.clip(boites[:, [0, 2]], 0, w)
    boites[:, [1, 3]] = np.clip(boites[:, [1, 3]], 0, h)
    ok = ((boites[:, 2] - boites[:, 0]) >= 1) & ((boites[:, 3] - boites[:, 1]) >= 1)
    return np.concatenate([boites[ok], np.asarray(scores, np.float32)[ok][:, None]], axis=1)
