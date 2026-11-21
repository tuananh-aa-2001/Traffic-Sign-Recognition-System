"""
Datensätze für Architektur A.

Beide liefern (Bild, Ziele). Ziele ist ein dict mit
  haupt (long, -1 = unbekannt), da (float), zusatz (float[K], Multi-Hot)
  und den Masken m_haupt, m_da, m_zusatz (1 = Label vorhanden).

SynsetDataset    vorbereitete Synset-Bilder (scripts/daten/prepare_synset.py): Hauptschild und "da"
RealCropDataset  echte Ausschnitte (crops/): "da", Zusatzzeichen und, wenn zuordenbar, Hauptschild
"""

import csv
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .preprocessing import _CLAHE
from .zusatz_klassen import MODELL_KLASSEN
from ..utils.zielgruppen import ist_gefahrzeichen

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
KLASSEN_JSON = os.path.join("configs", "synset_klassen.json")
SYNSET_DIR = os.path.join("data", "synset")


def load_synset_klassen(path=KLASSEN_JSON):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_transform(train: bool, size: int):
    """Augmentierung nur fürs Training. Kein Spiegeln: Pfeile und Richtungen würden falsch."""
    steps = [v2.ToImage()]
    if train:
        steps += [
            v2.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
        ]
    steps += [v2.Resize((size, size), antialias=True), v2.ToDtype(torch.float32, scale=True),
              v2.Normalize(MEAN, STD)]
    return v2.Compose(steps)


def letterbox(rgb: np.ndarray, size: int) -> np.ndarray:
    """Quadratisch auffüllen (Bildmittelwert) und skalieren, wie in scripts/daten/prepare_synset.py."""
    h, w = rgb.shape[:2]
    side = max(h, w)
    canvas = np.empty((side, side, 3), np.uint8)
    canvas[:] = rgb.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    y, x = (side - h) // 2, (side - w) // 2
    canvas[y:y + h, x:x + w] = rgb
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def _targets(haupt=-1, da=None, zusatz=None, gefahr=False):
    k = len(MODELL_KLASSEN)
    return {
        "haupt": torch.tensor(haupt if haupt is not None and haupt >= 0 else 0, dtype=torch.long),
        "m_haupt": torch.tensor(float(haupt is not None and haupt >= 0)),
        "da": torch.tensor(float(da or 0)),
        "m_da": torch.tensor(float(da is not None)),
        "zusatz": torch.as_tensor(zusatz if zusatz is not None else np.zeros(k), dtype=torch.float32),
        "m_zusatz": torch.tensor(float(zusatz is not None)),
        # Hauptschild ist ein Gefahrzeichen: nur für die Zielgruppen-Auswertung (Sinnbild = Hinweis)
        "gefahr": torch.tensor(float(gefahr)),
    }


class SynsetDataset(Dataset):
    def __init__(self, split: str, size: int, train: bool, clahe: bool = True, max_n: int = None):
        klassen = load_synset_klassen()
        self.haupt_index = {n: i for i, n in enumerate(klassen["hauptschilder"])}
        zusatz = set(klassen["zusatzzeichen"])
        self.bilder = np.load(os.path.join(SYNSET_DIR, f"{split}_bilder.npy"), mmap_mode="r")
        with open(os.path.join(SYNSET_DIR, f"{split}_labels.csv"), encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))[:max_n]
        self.labels = [(self.haupt_index[r["hauptschild"]], int(r["unten"] in zusatz), ist_gefahrzeichen(r["hauptschild"]))
                       for r in self.rows]
        self.transform, self.clahe = build_transform(train, size), clahe

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = np.array(self.bilder[int(self.rows[i]["index"])])
        if self.clahe:
            img = _CLAHE(img)
        haupt, da, gefahr = self.labels[i]
        return self.transform(img), _targets(haupt=haupt, da=da, gefahr=gefahr)


class RealCropDataset(Dataset):
    def __init__(self, records: list, size: int, train: bool, clahe: bool = True):
        klassen = load_synset_klassen()
        haupt_index = {n: i for i, n in enumerate(klassen["hauptschilder"])}
        index = {name: i for i, name in enumerate(MODELL_KLASSEN)}
        self.items = []
        for r in records:
            multi = np.zeros(len(MODELL_KLASSEN), np.float32)
            for k in r["klassen"]:
                if k in index:
                    multi[index[k]] = 1
            da = 0 if r["klassen"] == ["keins"] else 1
            gefahr = ist_gefahrzeichen(r["haupt_synset"] or r.get("mapillary_haupt", ""))
            self.items.append((r["crop"], haupt_index.get(r["haupt_synset"], -1), da, multi, gefahr))
        self.size, self.clahe = size, clahe
        self.transform = build_transform(train, size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, haupt, da, multi, gefahr = self.items[i]
        data = np.fromfile(path, dtype=np.uint8)               # Umlaute im Pfad unter Windows
        rgb = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if self.clahe:
            rgb = _CLAHE(rgb)
        return self.transform(letterbox(rgb, self.size)), _targets(haupt=haupt, da=da, zusatz=multi, gefahr=gefahr)
