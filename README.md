# Détection de visages : DNN OpenCV (ResNet-10) vs MobileNet fine-tuné sur WIDER FACE

Projet universitaire. On compare deux détecteurs de visages :

- **Détecteur DNN d'OpenCV** : SSD + ResNet-10, préentraîné, utilisé tel quel via `cv2.dnn` ;
- **MobileNet fine-tuné** : SSDLite + MobileNetV3, ré-entraîné sur WIDER FACE avec PyTorch.

| Modèle (WIDER_val)      | AP        | Précision  | Rappel    | F1 (seuil 0,5) | Vitesse (GPU) |
|---                      |---        |---         |---        |---             |---            |
| DNN OpenCV (ResNet-10)  | 0,357     | 0,972      | 0,209     | 0,343          | 31 ms         |
| MobileNet fine-tuné     | **0,780** | 0,965      | **0,530** | **0,684**      | 43 ms         |

## Démarrage rapide

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python detection_dnn.py --source 0            # webcam, détecteur DNN d'OpenCV
python detection_mobilenet.py --source 0      # webcam, MobileNet fine-tuné
```

Le modèle fine-tuné (`sorties/modeles/meilleur.pt`) et le modèle ResNet-10 (`deploy.prototxt`,
`res10_300x300_ssd_iter_140000.caffemodel`) sont inclus dans le dépôt. Les deux scripts de détection
fonctionnent donc **sans le dataset** sur vos propres images, vidéos ou la webcam.

### Dataset WIDER FACE (pour l'entraînement et l'évaluation)

Le dataset (3,6 Go) n'est pas dans le dépôt. Téléchargez-le sur le site officiel de WIDER FACE
(`WIDER_train`, `WIDER_val`, `WIDER_test` et `wider_face_split`), puis décompressez-le dans un dossier
`Wider Face/` à la racine du projet. S'il est ailleurs, indiquez son emplacement avec la variable
d'environnement `WIDER_FACE` (PowerShell : `$env:WIDER_FACE = "D:\datasets\Wider Face"`).
Vérifiez ensuite avec `python config.py`.

- **WIDER_train** : le modèle **apprend** sur ces images (fine-tuning).
- **WIDER_val** : images **jamais vues** pendant l'entraînement, utilisées pour **mesurer** les performances.
- **WIDER_test** : sans annotations publiques, utilisé seulement pour la **démonstration**.

## MobileNet tout-en-un : `mobilenet_complet.py`

Tout le programme MobileNet dans **un seul fichier**, indépendant des autres scripts : données, modèle,
entraînement, métriques, visualisation et détection.

```
python mobilenet_complet.py entrainer --test-rapide        # vérification (1 à 3 minutes)
python mobilenet_complet.py entrainer                      # fine-tuning complet
python mobilenet_complet.py evaluer                        # métriques sur Wider Face/WIDER_val
python mobilenet_complet.py evaluer --val "D:\...\WIDER_val" --max-images 300
python mobilenet_complet.py detecter --source 0 --flouter  # webcam
```

`evaluer` produit dans `sorties/mobilenet_complet/evaluation/` :
- `tableau_de_bord.png` : courbe précision-rappel, précision/rappel/F1 selon le seuil, matrice de confusion,
  VP/FP/FN selon le seuil, rappel selon la taille et selon le flou, l'occlusion et la pose ;
- `galerie_vp_fp_fn.png`, `images/` (exemples annotés), `courbes_perte.png` ;
- `resultats.json` et `par_seuil.csv`.

Sur tout WIDER_val (3 226 images, 16 072 visages) : AP 0,780, précision 0,965, rappel 0,530 au seuil 0,5,
meilleur F1 0,794 au seuil 0,25.

Sans `--poids`, `evaluer` et `detecter` utilisent `sorties/mobilenet_complet/meilleur.pt` s'il existe,
sinon le modèle déjà entraîné `sorties/modeles/meilleur.pt`. `--test-rapide` écrit dans un sous-dossier
à part et ne remplace jamais un vrai modèle.

## Scripts de détection et matrice de confusion

| Commande | Rôle |
|---|---|
| `python detection_dnn.py [--source X] [--flouter]` | Détection avec le DNN OpenCV seul (sans PyTorch) |
| `python detection_mobilenet.py [--source X] [--flouter]` | Détection avec MobileNet seul |
| `python matrice_confusion.py` | Matrice de confusion, VP/FP/FN selon le seuil, galerie d'exemples (sur WIDER_val) |

`--source` accepte une image, un dossier, une vidéo ou un numéro de webcam (`0`). Les autres options sont
`--seuil`, `--max`, `--sans-affichage` et `--sortie`.
Les résultats sont enregistrés dans `sorties/detection_dnn/`, `sorties/detection_mobilenet/` et `sorties/confusion/`.

> En détection d'objets, la case **VN** (vrai négatif) de la matrice de confusion n'est pas définie : une image
> contient une infinité de zones sans visage. C'est pourquoi on utilise la précision, le rappel et l'AP.

## 1. Installer PyTorch avec le GPU (à faire une seule fois)

Sous Windows, `pip install torch` installe la version **CPU seulement**. Pour utiliser la RTX 4060 :

```
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install matplotlib
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

La dernière commande doit afficher `True` et le nom de la carte. Le téléchargement fait environ 2,5 à 3 Go.
`cu126` convient au pilote 566.36 (CUDA 12.7). Si pip ne trouve pas de version compatible avec Python 3.14,
créez un environnement avec Python 3.12 (ou 3.11), ou utilisez Google Colab.

## 2. Fichiers

| Fichier | Rôle |
|---|---|
| `config.py` | Chemins du projet (seul fichier à adapter) |
| `wider_utils.py` | Lecture des annotations, filtrage, préparation des images, dessin des boîtes |
| `modele_mobilenet.py` | Construction du modèle SSDLite + MobileNetV3 pour les visages |
| `train_mobilenet.py` | Fine-tuning |
| `explore_dataset.py` | Statistiques et figures du dataset (pour la présentation) |
| `detecteurs.py` | Deux détecteurs avec la même interface : ResNet-10 (OpenCV) et MobileNet fine-tuné |
| `metriques.py` | Appariement des boîtes (IoU), précision, rappel, AP |
| `evaluate.py` | Comparaison chiffrée des deux modèles sur WIDER_val |
| `total_mobilenet.py` | Démonstration : floutage par lots sur WIDER_test |
| `matrice_confusion.py` | Matrice de confusion + visualisation des VP / FP / FN |
| `detection_dnn.py` | Détection avec le DNN OpenCV (ResNet-10) uniquement |
| `detection_mobilenet.py` | Détection avec MobileNet fine-tuné uniquement |
| `demo_commune.py` | Code partagé par les deux scripts de détection (sources, dessin, floutage) |

## 3. Ordre d'exécution

```
python config.py                         # verifie que tous les chemins sont trouves (OK partout)
python explore_dataset.py --exemples 6      # statistiques, figures, images avec boîtes
python train_mobilenet.py --test-rapide     # vérification du pipeline (1 à 2 minutes)
python train_mobilenet.py --max-images 1500 --epochs 2 --gel 0   # mesure de la vitesse (4 à 5 minutes)
python train_mobilenet.py                   # entraînement complet
python train_mobilenet.py --reprendre       # reprend après une interruption (mêmes réglages)
python evaluate.py --max-images 300         # évaluation rapide (environ 1 minute)
python evaluate.py                          # évaluation complète sur WIDER_val -> sorties/evaluation/
python total_mobilenet.py                   # démonstration : ResNet-10 et MobileNet côte à côte
```

### Protocole d'évaluation
- Jeu : WIDER_val (3 226 images annotées, jamais utilisées pour l'entraînement).
- Une détection est correcte si son IoU avec un visage non encore trouvé est >= 0,5.
- Les visages invalides ou de moins de 20 pixels sont ignorés (ils ne comptent ni comme visages à trouver,
  ni comme erreurs s'ils sont détectés). Voir `--taille-min`.
- Métriques : AP (indépendante du seuil), précision, rappel et F1 au seuil de confiance choisi (0,5 par défaut),
  rappel selon la taille des visages, vitesse (ms par image).
- Sorties de `evaluate.py` : `comparaison.png`, `analyse_echecs.png` (effet du seuil, rappel selon flou / occlusion / pose),
  `resultats.csv`, `seuils.csv` (précision et rappel à 8 seuils), `attributs.csv`, `resultats.json`.
- `sorties/evaluation/echecs/` : images où chaque modèle rate le plus de visages (orange : raté, vert : trouvé,
  rouge : faux positif).

Résultats dans `sorties/` : `exploration/` (figures, JSON) et `modeles/` (`meilleur.pt`, `dernier.pt`,
`journal_entrainement.csv`, `courbes_perte.png`).

Si la mémoire du GPU est insuffisante : `python train_mobilenet.py --batch 8`.
Pour aller plus vite : `--taille 480` ou `--max-images 6000`.

## 4. Choix techniques

- **Données** : WIDER_train pour le fine-tuning (95 % entraînement, 5 % mini-validation de suivi).
  WIDER_val (annotée) pour l'évaluation chiffrée. WIDER_test (non annotée) pour la démonstration.
- **Modèle** : réseau de base MobileNetV3 préentraîné (COCO), tête de détection recréée pour 2 classes
  (fond / visage), boîtes par défaut plus petites que celles de COCO.
- **Entrée** : image redimensionnée en 640 x 640 (au lieu de 320) pour mieux voir les petits visages.
- **Augmentation** : retournement horizontal, variation de luminosité et de contraste.
- **Filtrage** : visages invalides exclus, visages plus petits que 16 pixels (taille d'origine) ignorés.
- **Optimisation** : AdamW, réseau de base gelé pendant 2 époques, puis dégelé ; précision mixte (AMP).

## 5. Problèmes fréquents (Windows)

- **`error code: <1455>`, `Memory allocation failure`, puis `CUDA error: unknown error`** : la mémoire virtuelle de
  Windows est trop petite pour les processus de chargement des images. Solutions, dans l'ordre :
  1. lancer avec `--workers 0` (plus lent, mais sans multiprocessing) ;
  2. agrandir le fichier d'échange : *Paramètres > Système > À propos > Paramètres système avancés > Performances >
     Paramètres > Avancé > Mémoire virtuelle > Modifier*, décocher la gestion automatique, choisir le disque C:,
     taille personnalisée (initiale 16384 Mo, maximale 32768 Mo), puis redémarrer le PC.
- **`CUDA out of memory`** : ajouter `--batch 8`.
