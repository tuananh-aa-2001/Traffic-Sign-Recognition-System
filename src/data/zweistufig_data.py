"""
Datensätze für Architektur B.

SynsetSegDataset     Synset-Bild + Klassenkarte (prepare_synset.py --segmentierung) für das U-Net
ZusatzEinzelDataset  ein einzelnes Zusatzschild aus einem echten Ausschnitt, Box von Mapillary, eine Klasse
zusatz_einzel_items  baut die Einträge für ZusatzEinzelDataset aus den echten Ausschnitten
"""

import csv
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from .crop_records import SOURCE_LABELS, class_of
from .multitask_data import MEAN, STD, SYNSET_DIR, letterbox
from .preprocessing import _CLAHE
from .zusatz_klassen import MODELL_KLASSEN

BOXEN_JSON = os.path.join("real_world_mapillary", "boxen.json")


class SynsetSegDataset(Dataset):
    """Bild und Klassenkarte bekommen dieselbe geometrische Augmentierung (kein Spiegeln)."""

    def __init__(self, split, train, clahe=True, max_n=None):
        self.bilder = np.load(os.path.join(SYNSET_DIR, f"{split}_bilder.npy"), mmap_mode="r")
        self.karten = np.load(os.path.join(SYNSET_DIR, f"{split}_klassen.npy"), mmap_mode="r")
        self.n = min(len(self.bilder), max_n or len(self.bilder))
        self.clahe = clahe
        geo = [v2.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1))] if train else []
        farbe = [v2.ColorJitter(0.3, 0.3, 0.2)] if train else []
        self.transform = v2.Compose(geo + farbe + [v2.ToDtype({tv_tensors.Image: torch.float32, "others": None},
                                                              scale=True), v2.Normalize(MEAN, STD)])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = np.array(self.bilder[i])
        if self.clahe:
            img = _CLAHE(img)
        bild = tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1))
        karte = tv_tensors.Mask(torch.from_numpy(np.array(self.karten[i])).long())
        bild, karte = self.transform(bild, karte)
        return bild.as_subclass(torch.Tensor), karte.as_subclass(torch.Tensor)


def _detektionen():
    """Originalpfad (wie in labels.csv) -> Liste {wert, box (normiert)} aus labels.csv bzw. boxen.json."""
    cache = {}
    if os.path.exists(BOXEN_JSON):
        with open(BOXEN_JSON, encoding="utf-8") as f:
            cache = json.load(f)
    det = {}
    with open(SOURCE_LABELS, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for r in reader:
            if len(r) == 11:
                pfad, bild_id, erk = r[0], r[2], r[4]
            elif len(r) == 10:
                pfad, bild_id, erk = r[0], r[1], r[3]
            else:
                continue
            liste = [d for d in json.loads(erk or "[]") if "box" in d] or cache.get(bild_id, [])
            det[pfad] = [d for d in liste if "box" in d]
    return det


ZUSATZ_BOXEN_JSON = os.path.join("crops", "zusatz_boxen.json")


def _boxen_im_ausschnitt(records):
    """Ausschnitt -> [(Mapillary-Wert, Box in Ausschnitt-Koordinaten)].

    Wird aus den Originalfotos berechnet und in crops/zusatz_boxen.json gespeichert, damit auf
    Colab (ohne Originalfotos, nur crops/) dieselben Boxen verfügbar sind.
    """
    cache = {}
    if os.path.exists(ZUSATZ_BOXEN_JSON):
        with open(ZUSATZ_BOXEN_JSON, encoding="utf-8") as f:
            cache = json.load(f)
    fehlend = [r for r in records if r["crop"] not in cache and os.path.exists(r["original"])
               and r["mapillary_zusatz"] != "none"]
    if fehlend:
        det = _detektionen()
        for r in fehlend:
            w_img, h_img = Image.open(r["original"]).size
            x0, y0, x1, y1 = r["box"]
            werte = r["mapillary_zusatz"].split("|")
            liste = []
            for d in det.get(r["original_roh"], []):
                if d["wert"] not in werte:
                    continue
                bx = [d["box"][0] * w_img, d["box"][1] * h_img, d["box"][2] * w_img, d["box"][3] * h_img]
                if x0 <= (bx[0] + bx[2]) / 2 <= x1 and y0 <= (bx[1] + bx[3]) / 2 <= y1:
                    liste.append([d["wert"], [round(bx[0] - x0), round(bx[1] - y0), round(bx[2] - x0), round(bx[3] - y0)]])
            cache[r["crop"]] = liste
        with open(ZUSATZ_BOXEN_JSON, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    return cache


def zusatz_einzel_items(records):
    """Einzelne Zusatzschilder mit Klasse aus den Ausschnitt-Datensätzen.

    Box: Mapillary-Detektion des Zusatzzeichens (im Ausschnitt). Klasse: aus dem Mapillary-Wert;
    bei geprüften Ausschnitten muss sie zur Prüfung passen (bei einem einzelnen Schild gilt die
    geprüfte Klasse, auch wenn sie korrigiert wurde), sonst wird das Schild übersprungen.
    """
    boxen = _boxen_im_ausschnitt(records)
    index = {k: i for i, k in enumerate(MODELL_KLASSEN)}
    items = []
    for r in records:
        if r["klassen"] == ["keins"] or r["mapillary_zusatz"] == "none":
            continue
        werte = r["mapillary_zusatz"].split("|")
        for wert, box in boxen.get(r["crop"], []):
            klasse = class_of(wert)
            if r["geprueft"]:
                if len(werte) == 1 and len(r["klassen"]) == 1:
                    klasse = r["klassen"][0]
                elif klasse not in r["klassen"]:
                    continue
            if klasse in index:
                items.append({"bild": r["crop"], "box": tuple(box), "klasse": index[klasse],
                              "region": r["region"], "split": r["split"]})
    return items


class ZusatzEinzelDataset(Dataset):
    def __init__(self, items, size=96, train=False, clahe=True):
        self.items, self.size, self.clahe = items, size, clahe
        steps = [v2.ToImage()]
        if train:
            steps += [v2.RandomAffine(degrees=6, translate=(0.06, 0.06), scale=(0.85, 1.15)),
                      v2.ColorJitter(0.3, 0.3, 0.2), v2.RandomApply([v2.GaussianBlur(5, (0.1, 1.5))], p=0.3)]
        steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
        self.transform = v2.Compose(steps)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.cvtColor(cv2.imdecode(np.fromfile(it["bild"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return self.transform(ausschnitt(img, it["box"], self.size, self.clahe)), it["klasse"]


def ausschnitt(rgb, box, size, clahe=True, rand=0.1):
    """Box mit Rand ausschneiden, quadratisch auffüllen, skalieren (auch für die Pipeline genutzt)."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    p = rand * max(x1 - x0, y1 - y0)
    x0, y0 = int(max(0, x0 - p)), int(max(0, y0 - p))
    x1, y1 = int(min(w, x1 + p)), int(min(h, y1 + p))
    crop = rgb[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
    if clahe:
        crop = _CLAHE(np.ascontiguousarray(crop))
    return letterbox(crop, size)
