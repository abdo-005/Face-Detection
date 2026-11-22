"""
detection_dnn.py : détection de visages avec UNIQUEMENT le « Détecteur DNN » d'OpenCV.

Modèle : SSD + ResNet-10 au format Caffe, préentraîné et fourni avec OpenCV
(deploy.prototxt = architecture, res10_300x300_ssd_iter_140000.caffemodel = poids).
Il est exécuté par le module cv2.dnn : aucune dépendance à PyTorch, seulement opencv-python et numpy.

    python detection_dnn.py                                  # 20 images de WIDER_test/0--Parade
    python detection_dnn.py --source photo.jpg               # une image
    python detection_dnn.py --source 0                       # webcam
    python detection_dnn.py --source video.mp4 --flouter     # vidéo, visages floutés

Seuil par défaut : 0.5 (valeur habituelle pour ce modèle). Sur WIDER_val, son meilleur F1 est atteint
vers 0.15, mais avec plus de fausses détections.
Résultats dans sorties/detection_dnn/.
"""
import config
from demo_commune import arguments, lancer
from detecteurs import DetecteurResNet


def main():
    a = arguments("Detection de visages avec le detecteur DNN d'OpenCV (ResNet-10)", seuil_defaut=0.5,
                  nom_sortie="detection_dnn").parse_args()
    for f in (config.PROTOTXT, config.CAFFEMODEL):
        if not f.exists():
            raise SystemExit(f"Fichier du modele introuvable : {f}")
    detecteur = DetecteurResNet(config.PROTOTXT, config.CAFFEMODEL)
    detecteur.nom = "DNN OpenCV (ResNet-10)"
    lancer(detecteur, a)


if __name__ == "__main__":
    main()
