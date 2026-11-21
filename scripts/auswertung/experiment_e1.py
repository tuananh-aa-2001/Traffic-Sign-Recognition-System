"""
Experiment E1: synthetisch gegen echt beim Hauptschild.

Gleiche Architektur (ResNetGTSRB, 48x48), gleiche Vorverarbeitung (CLAHE in voller Größe, dann 48x48),
gleiche 43 Klassen (Synset-Label 0-42 = GTSRB-Klasse 0-42, geprüft am 20.11.2026). Es entsteht eine
2x2-Tabelle:

                       Test Synset-Val    Test GTSRB-Test
  trainiert auf Synset       ✓                  ✓          (wird hier trainiert)
  trainiert auf GTSRB        ✓                  ✓          (vorhandenes Baseline-Modell, 99,36 %)

Vorher:
  python scripts/daten/prepare_synset.py --split train
  python scripts/daten/prepare_synset.py --split validation
  GTSRB-Daten unter data/raw (python src/utils/download_dataset.py --data_dir data/raw), für die GTSRB-Spalte

Beispiele:
  python scripts/auswertung/experiment_e1.py                       # trainieren (falls noch kein Modell) und alles auswerten
  python scripts/auswertung/experiment_e1.py --nur-auswerten       # nur auswerten
  python scripts/auswertung/experiment_e1.py --smoke               # Probelauf auf der CPU mit wenigen Bildern

Ergebnis: results/e1/ergebnis.json (Accuracy, Makro-F1, Genauigkeit je Klasse, größte Abfälle)
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # Projektordner (src/)
import argparse
import csv
import json
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from src.data.augmentation import GTSRBAugmentation
from src.data.preprocessing import preprocess_image
from src.models.resnet_gtsrb import ResNetGTSRB
from src.utils.class_names import GTSRB_CLASSES

SYNSET_DIR = os.path.join("data", "synset")
OUT_DIR = os.path.join("results", "e1")
CKPT_DIR = os.path.join("models", "checkpoints", "e1_synset")
GTSRB_MODELL = os.path.join("models", "checkpoints", "best_model.pth")


class SynsetGTSRBKlassen(Dataset):
    """Synset-Bilder der Labels 0-42, verarbeitet wie die GTSRB-Bilder in src/data/data_loader.py."""

    def __init__(self, split, config, train, max_n=None):
        self.bilder = np.load(os.path.join(SYNSET_DIR, f"{split}_bilder.npy"), mmap_mode="r")
        with open(os.path.join(SYNSET_DIR, f"{split}_labels.csv"), encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if int(r["label"]) < 43]
        self.items = [(int(r["index"]), int(r["label"])) for r in rows][:max_n]
        self.size = config["data"]["image_size"]
        self.clahe = config["augmentation"]["apply_clahe"]
        self.transform = GTSRBAugmentation(config, is_training=train)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        idx, label = self.items[i]
        img = preprocess_image(np.array(self.bilder[idx]), size=(self.size, self.size),
                               apply_clahe=self.clahe, normalize=False)
        return self.transform(img), label


@torch.no_grad()
def auswerten(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        y_pred += model(x.to(device)).argmax(1).cpu().tolist()
        y_true += list(y.numpy() if torch.is_tensor(y) else y)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    je_klasse = {}
    for k in sorted(set(y_true.tolist())):
        m = y_true == k
        je_klasse[k] = {"name": GTSRB_CLASSES[k]["de"], "genauigkeit": round(float((y_pred[m] == k).mean()), 4),
                        "anzahl": int(m.sum())}
    return {"accuracy": float((y_true == y_pred).mean()) if len(y_true) else None,
            "makro_f1": float(f1_score(y_true, y_pred, average="macro", labels=sorted(set(y_true.tolist()))))
            if len(y_true) else None,
            "n": int(len(y_true)), "je_klasse": je_klasse}


def lade_modell(path, device):
    model = ResNetGTSRB(num_classes=43, pretrained=False).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model


def gtsrb_test_loader(config):
    from src.data.data_loader import GTSRBDataset
    try:
        ds = GTSRBDataset(config["data"]["root_dir"], split="test", config=config,
                          apply_clahe=config["augmentation"]["apply_clahe"])
    except FileNotFoundError as e:
        print(f"GTSRB-Test nicht verfügbar, Spalte bleibt leer: {str(e).splitlines()[0]}")
        return None
    return DataLoader(ds, batch_size=config["training"]["batch_size"], num_workers=config["training"]["num_workers"])


def main(args):
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out_dir, ckpt_dir = OUT_DIR, CKPT_DIR
    max_n = None
    if args.smoke:
        out_dir, ckpt_dir = OUT_DIR + "_smoke", CKPT_DIR + "_smoke"
        config["training"].update(num_epochs=1, batch_size=32, num_workers=0)
        max_n = 200
    config["data"]["num_classes"] = 43
    config["checkpoint"]["save_dir"] = ckpt_dir
    config["logging"]["tensorboard_dir"] = os.path.join(out_dir, "tensorboard")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    nw = config["training"]["num_workers"]

    syn_train_split = "validation" if args.smoke else "train"
    syn_val = SynsetGTSRBKlassen("validation", config, False, max_n)
    syn_val_loader = DataLoader(syn_val, batch_size=config["training"]["batch_size"], num_workers=nw)
    print(f"Synset-Validierung (Klassen 0-42): {len(syn_val)} Bilder")

    e1_modell = os.path.join(ckpt_dir, "best_model.pth")
    if not args.nur_auswerten and (args.neu or not os.path.exists(e1_modell)):
        from src.training.trainer import Trainer
        train = SynsetGTSRBKlassen(syn_train_split, config, True, max_n)
        print(f"Training auf Synset (Klassen 0-42): {len(train)} Bilder, {device}")
        train_loader = DataLoader(train, batch_size=config["training"]["batch_size"], shuffle=True,
                                  num_workers=nw, drop_last=True, pin_memory=device.type == "cuda")
        model = ResNetGTSRB(num_classes=43, pretrained=False, dropout=config["model"]["dropout"]).to(device)
        Trainer(model, train_loader, syn_val_loader, config, device).train()
        # Der Trainer schreibt die Lernkurve fest nach results/; zum Experiment verschieben
        if os.path.exists(os.path.join("results", "training_curves.png")):
            shutil.move(os.path.join("results", "training_curves.png"), os.path.join(out_dir, "training_curves.png"))

    gtsrb_test = None if args.smoke else gtsrb_test_loader(config)
    tabelle = {}
    for name, path in (("synset", e1_modell), ("gtsrb", GTSRB_MODELL)):
        if not os.path.exists(path):
            print(f"Modell fehlt: {path}")
            continue
        model = lade_modell(path, device)
        tabelle[f"{name}->synset_val"] = auswerten(model, syn_val_loader, device)
        if gtsrb_test is not None:
            tabelle[f"{name}->gtsrb_test"] = auswerten(model, gtsrb_test, device)

    print("\nE1: Accuracy / Makro-F1 auf 43 gemeinsamen Klassen")
    for k, v in tabelle.items():
        print(f"  {k:22} {v['accuracy']:.4f} / {v['makro_f1']:.4f}  (n = {v['n']})")

    # Größter Abfall je Klasse, wenn ein Modell die Domäne wechselt
    abfall = {}
    for quelle, eigen, fremd in (("synset", "synset->synset_val", "synset->gtsrb_test"),
                                 ("gtsrb", "gtsrb->gtsrb_test", "gtsrb->synset_val")):
        if eigen in tabelle and fremd in tabelle:
            d = [(k, tabelle[eigen]["je_klasse"][k]["genauigkeit"] - tabelle[fremd]["je_klasse"][k]["genauigkeit"])
                 for k in tabelle[fremd]["je_klasse"] if k in tabelle[eigen]["je_klasse"]]
            abfall[quelle] = [{"klasse": k, "name": GTSRB_CLASSES[k]["de"], "abfall": round(v, 4)}
                              for k, v in sorted(d, key=lambda x: -x[1])[:8]]
    ergebnis = {"tabelle": tabelle, "groesster_abfall": abfall}
    with open(os.path.join(out_dir, "ergebnis.json"), "w", encoding="utf-8") as f:
        json.dump(ergebnis, f, ensure_ascii=False, indent=1, default=str)
    print(f"Ergebnis: {os.path.join(out_dir, 'ergebnis.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Experiment E1: Synset gegen GTSRB beim Hauptschild")
    p.add_argument("--config", default=os.path.join("configs", "gtsrb.yaml"))
    p.add_argument("--neu", action="store_true", help="Synset-Modell neu trainieren, auch wenn es existiert")
    p.add_argument("--nur-auswerten", action="store_true")
    p.add_argument("--smoke", action="store_true", help="kurzer Probelauf auf der CPU")
    main(p.parse_args())
