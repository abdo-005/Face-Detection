"""
detection_mobilenet.py : détection de visages avec UNIQUEMENT le modèle MobileNet fine-tuné (PyTorch).

Modèle : SSDLite + MobileNetV3, ré-entraîné sur WIDER_train par train_mobilenet.py
(poids dans sorties/modeles/meilleur.pt). Nécessite torch et torchvision. Utilise le GPU s'il est disponible.

    python detection_mobilenet.py                                  # 20 images de WIDER_test/0--Parade
    python detection_mobilenet.py --source photo.jpg               # une image
    python detection_mobilenet.py --source "Wider Face/WIDER_test/images/12--Group" --max 10
    python detection_mobilenet.py --source 0                       # webcam
    python detection_mobilenet.py --source video.mp4 --flouter     # vidéo, visages floutés
    python detection_mobilenet.py --device cpu                     # forcer le CPU

Seuil par défaut : 0.25, le seuil qui maximise le F1 de ce modèle sur WIDER_val (voir evaluate.py).
Résultats dans sorties/detection_mobilenet/.
"""
import config
from demo_commune import arguments, lancer
from detecteurs import DetecteurMobileNet


def main():
    p = arguments("Detection de visages avec MobileNet fine-tune", seuil_defaut=0.25,
                  nom_sortie="detection_mobilenet")
    p.add_argument("--poids", default=str(config.MODELE_MOBILENET), help="fichier .pt du modele fine-tune")
    p.add_argument("--device", default=None, help="cuda ou cpu (defaut : automatique)")
    a = p.parse_args()

    if not config.MODELE_MOBILENET.exists() and a.poids == str(config.MODELE_MOBILENET):
        raise SystemExit("Modele introuvable. Entrainez-le d'abord : python train_mobilenet.py")
    detecteur = DetecteurMobileNet(a.poids, a.device)
    lancer(detecteur, a)


if __name__ == "__main__":
    main()
