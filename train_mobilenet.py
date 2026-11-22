"""
train_mobilenet.py : fine-tuning de SSDLite-MobileNetV3 sur WIDER FACE (train).

Utilisation (dans le dossier du projet, venv activé) :
    python train_mobilenet.py --test-rapide     # vérifie que tout fonctionne (1 à 2 minutes)
    python train_mobilenet.py                   # entraînement complet
    python train_mobilenet.py --epochs 10 --taille 480 --batch 32   # exemple de réglages

Sorties dans sorties/modeles/ : meilleur.pt, dernier.pt, journal_entrainement.csv, courbes_perte.png
"""
import argparse
import csv
import math
import os
import random
import time
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config
from modele_mobilenet import construire_modele, sauvegarder
from wider_utils import filtrer_echantillons, lire_annotations, preparer_echantillon, separer

cv2.setNumThreads(0)   # evite que chaque worker lance ses propres threads OpenCV


class WiderDataset(Dataset):
    """Une image + ses boîtes de visages, au format attendu par torchvision."""

    def __init__(self, echantillons, taille, entrainement):
        self.echantillons = echantillons
        self.taille = taille
        self.entrainement = entrainement

    def __len__(self):
        return len(self.echantillons)

    def __getitem__(self, idx):
        e = self.echantillons[idx]
        image = cv2.imread(str(e.chemin))
        if image is None:
            raise FileNotFoundError(f"Image introuvable : {e.chemin}")
        img, boites = preparer_echantillon(image, e.boites, self.taille, self.entrainement)
        # uint8 (3, H, W) : 4 fois plus leger a transferer ; la conversion en [0, 1] se fait sur le GPU
        tenseur = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        cible = {
            "boxes": torch.from_numpy(boites).reshape(-1, 4),
            "labels": torch.ones((len(boites),), dtype=torch.int64),          # 1 = visage
        }
        return tenseur, cible


def collate(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def perte_validation(modele, loader, device):
    """Perte sur la mini-validation (les BatchNorm restent figées)."""
    modele.train()                                    # nécessaire pour que le modèle renvoie la perte
    for m in modele.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    total, n = 0.0, 0
    for images, cibles in loader:
        images = [im.to(device).float().div_(255.0) for im in images]
        cibles = [{k: v.to(device) for k, v in c.items()} for c in cibles]
        pertes = modele(images, cibles)
        total += sum(pertes.values()).item() * len(images)
        n += len(images)
    return total / max(n, 1)


def tracer_courbes(journal, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [l["epoch"] for l in journal]
    plt.figure(figsize=(7, 4))
    plt.plot(ep, [l["perte_train"] for l in journal], marker="o", label="Entrainement")
    plt.plot(ep, [l["perte_val"] for l in journal], marker="o", label="Mini-validation")
    plt.xlabel("Epoque")
    plt.ylabel("Perte (classification + localisation)")
    plt.title("Fine-tuning de SSDLite-MobileNetV3")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--taille", type=int, default=640, help="taille d'entree du reseau (carre)")
    p.add_argument("--lr-tete", type=float, default=1e-3)
    p.add_argument("--lr-base", type=float, default=2e-4, help="taux d'apprentissage du reseau de base")
    p.add_argument("--gel", type=int, default=2, help="epoques pendant lesquelles le reseau de base est gele")
    p.add_argument("--taille-min", type=int, default=16, help="plus petit visage utilise (pixels d'origine)")
    p.add_argument("--max-images", type=int, default=None, help="sous-ensemble du train (pour aller plus vite)")
    p.add_argument("--workers", type=int, default=2, help="processus de chargement des images (0 = aucun, le plus sur)")
    p.add_argument("--test-rapide", action="store_true", help="64 images, 1 epoque : verification du pipeline")
    p.add_argument("--graine", type=int, default=42)
    p.add_argument("--reprendre", action="store_true", help="reprend l'entrainement depuis sorties/modeles/reprise.pt")
    args = p.parse_args()

    if args.test_rapide:
        args.max_images, args.epochs, args.batch, args.workers, args.gel = 64, 1, 8, 0, 0

    warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")   # avertissement sans gravite (AMP)
    random.seed(args.graine)
    np.random.seed(args.graine)
    torch.manual_seed(args.graine)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Peripherique : {device}")
    if device.type == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("ATTENTION : PyTorch ne voit pas le GPU (version CPU installee ?). L'entrainement sera tres lent.")
    use_amp = device.type == "cuda"

    # --- Donnees -----------------------------------------------------------------------------
    echantillons = filtrer_echantillons(lire_annotations(config.ANNOT_TRAIN, config.IMAGES_TRAIN), args.taille_min)
    manquantes = [e for e in echantillons if not e.chemin.exists()]
    if len(manquantes) > 0.05 * len(echantillons):
        print(f"ERREUR : {len(manquantes)} images sur {len(echantillons)} introuvables, par exemple :")
        print(f"  {manquantes[0].chemin}")
        print(f"Dossier utilise (IMAGES_TRAIN) : {config.IMAGES_TRAIN}")
        print("Lancez 'python config.py' pour verifier les chemins.")
        raise SystemExit(1)
    if manquantes:
        print(f"Attention : {len(manquantes)} image(s) introuvable(s) ignoree(s).")
        echantillons = [e for e in echantillons if e.chemin.exists()]
    print(f"Images du train avec au moins un visage >= {args.taille_min}px : {len(echantillons)}")
    if args.max_images and len(echantillons) > args.max_images:
        random.Random(args.graine).shuffle(echantillons)
        echantillons = echantillons[:args.max_images]
    train, suivi = separer(echantillons, fraction=0.05, graine=args.graine)
    print(f"Entrainement : {len(train)} images | Mini-validation : {len(suivi)} images")

    loader_train = DataLoader(WiderDataset(train, args.taille, True), batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate, pin_memory=device.type == "cuda",
                              persistent_workers=args.workers > 0, drop_last=True)
    loader_suivi = DataLoader(WiderDataset(suivi, args.taille, False), batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, collate_fn=collate, pin_memory=device.type == "cuda",
                              persistent_workers=args.workers > 0)

    # --- Modele ------------------------------------------------------------------------------
    modele = construire_modele(args.taille, pretrained=True).to(device)
    optimiseur = torch.optim.AdamW([
        {"params": modele.backbone.parameters(), "lr": args.lr_base},
        {"params": modele.head.parameters(), "lr": args.lr_tete},
    ], weight_decay=1e-4)

    total_iters = max(1, len(loader_train) * args.epochs)
    chauffe = min(200, total_iters // 10 + 1)

    def facteur(it):                                   # montee progressive puis decroissance cosinus
        if it < chauffe:
            return (it + 1) / chauffe
        prog = (it - chauffe) / max(1, total_iters - chauffe)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))

    planificateur = torch.optim.lr_scheduler.LambdaLR(optimiseur, facteur)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config.DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)
    journal, meilleure, debut = [], float("inf"), 1
    chemin_reprise = config.DOSSIER_MODELES / "reprise.pt"
    if args.reprendre:
        if chemin_reprise.exists():
            etat = torch.load(chemin_reprise, map_location=device, weights_only=False)
            modele.load_state_dict(etat["model"])
            optimiseur.load_state_dict(etat["optimiseur"])
            planificateur.load_state_dict(etat["planificateur"])
            scaler.load_state_dict(etat["scaler"])
            journal, meilleure, debut = etat["journal"], etat["meilleure"], etat["epoch"] + 1
            print(f"Reprise de l'entrainement a l'epoque {debut} (gardez les memes reglages que le premier lancement)")
        else:
            print("Aucun fichier reprise.pt : l'entrainement demarre du debut.")

    # --- Entrainement ------------------------------------------------------------------------
    for epoch in range(debut, args.epochs + 1):
        geler = epoch <= args.gel
        for q in modele.backbone.parameters():
            q.requires_grad = not geler
        modele.train()
        t0, somme, n = time.time(), 0.0, 0

        for it, (images, cibles) in enumerate(loader_train, 1):
            images = [im.to(device, non_blocking=True).float().div_(255.0) for im in images]
            cibles = [{k: v.to(device, non_blocking=True) for k, v in c.items()} for c in cibles]

            with torch.autocast(device_type=device.type, enabled=use_amp):
                pertes = modele(images, cibles)
                perte = sum(pertes.values())

            optimiseur.zero_grad(set_to_none=True)
            scaler.scale(perte).backward()
            scaler.unscale_(optimiseur)
            torch.nn.utils.clip_grad_norm_(modele.parameters(), 10.0)
            scaler.step(optimiseur)
            scaler.update()
            planificateur.step()

            somme += perte.item() * len(images)
            n += len(images)
            if it % 50 == 0 or it == len(loader_train):
                vitesse = n / max(time.time() - t0, 1e-6)
                reste = (len(loader_train) - it) * args.batch / max(vitesse, 1e-6)
                print(f"  epoque {epoch}/{args.epochs} - lot {it}/{len(loader_train)} - perte {somme / n:.4f} - "
                      f"{vitesse:.0f} img/s - reste ~{reste / 60:.1f} min pour cette epoque", flush=True)

        perte_train = somme / max(n, 1)
        perte_val = perte_validation(modele, loader_suivi, device)
        duree = time.time() - t0
        print(f"Epoque {epoch}/{args.epochs} | perte train {perte_train:.4f} | perte val {perte_val:.4f} | "
              f"{duree:.0f}s{' | reseau de base gele' if geler else ''}")

        journal.append({"epoch": epoch, "perte_train": round(perte_train, 5), "perte_val": round(perte_val, 5),
                        "duree_s": round(duree, 1)})
        sauvegarder(modele, config.DOSSIER_MODELES / "dernier.pt", args.taille, epoch=epoch, perte_val=perte_val)
        if perte_val < meilleure:
            meilleure = perte_val
            sauvegarder(modele, config.DOSSIER_MODELES / "meilleur.pt", args.taille, epoch=epoch, perte_val=perte_val)
            print("  -> nouveau meilleur modele enregistre")
        if not args.test_rapide:
            torch.save({"model": modele.state_dict(), "optimiseur": optimiseur.state_dict(),
                        "planificateur": planificateur.state_dict(), "scaler": scaler.state_dict(),
                        "journal": journal, "meilleure": meilleure, "epoch": epoch}, chemin_reprise)
        print(f"  temps restant estime : ~{duree * (args.epochs - epoch) / 60:.0f} min "
              f"(la 1re epoque inclut l'initialisation du GPU : estimation prudente)", flush=True)

        with open(config.DOSSIER_MODELES / "journal_entrainement.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(journal[0].keys()))
            w.writeheader()
            w.writerows(journal)
        tracer_courbes(journal, config.DOSSIER_MODELES / "courbes_perte.png")

    print(f"Termine. Meilleur modele : {config.DOSSIER_MODELES / 'meilleur.pt'} (perte val {meilleure:.4f})")


if __name__ == "__main__":      # obligatoire sous Windows (multiprocessing des DataLoader)
    main()
