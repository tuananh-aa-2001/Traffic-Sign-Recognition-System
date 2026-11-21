"""
Architektur B trainieren und auswerten: U-Net -> Boxen der Zusatzschilder -> Klassifikator je Schild.

Vorher:
  python scripts/daten/prepare_synset.py --split train --segmentierung
  python scripts/daten/prepare_synset.py --split validation --segmentierung
  python scripts/daten/crop_signs.py mapillary   (und möglichst scripts/daten/review_crops.py)

Beispiele:
  python scripts/training/train_zweistufig.py unet               # Stufe 1 auf Synset trainieren
  python scripts/training/train_zweistufig.py klassifikator      # Stufe 2 auf echten Zusatzschildern trainieren
  python scripts/training/train_zweistufig.py auswerten          # ganze Pipeline auf dem echten Testsatz
  python scripts/training/train_zweistufig.py alles --smoke      # Probelauf auf der CPU

Ergebnisse: models/checkpoints/b_unet.pth, b_zusatz.pth und results/zweistufig/*.json
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # Projektordner (src/)
import argparse
import json
import os
import random
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, RandomSampler

from src.data.crop_records import build_records
from src.data.multitask_data import MEAN, STD, letterbox
from src.data.preprocessing import _CLAHE
from src.data.segmentation import KLASSEN as SEG_KLASSEN, zusatz_boxen
from src.data.zusatz_klassen import MODELL_KLASSEN
from src.data.zweistufig_data import SynsetSegDataset, ZusatzEinzelDataset, ausschnitt, zusatz_einzel_items
from src.models.unet import UNetResNet18, ZusatzKlassifikator
from src.utils.zielgruppen import ist_gefahrzeichen, regel, zielgruppe_gleich


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(rgb):
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
    return ((x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]).unsqueeze(0)


# ---------------------------------------------------------------- Stufe 1: U-Net

@torch.no_grad()
def iou_auswerten(model, loader, device, max_batches=None):
    model.eval()
    inter, union = np.zeros(4), np.zeros(4)
    for b, (x, y) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        p = model(x.to(device)).argmax(1).cpu()
        for k in range(4):
            inter[k] += ((p == k) & (y == k)).sum().item()
            union[k] += ((p == k) | (y == k)).sum().item()
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return {"miou": float(np.nanmean(iou)), "iou": {SEG_KLASSEN[k]: round(float(iou[k]), 4) for k in range(4)}}


def train_unet(cfg, device, smoke):
    d, u = cfg["data"], cfg["unet"]
    split_train = d["synset_val"] if smoke else d["synset_train"]
    max_n = 200 if smoke else None
    train = SynsetSegDataset(split_train, True, d["clahe"], max_n)
    val = SynsetSegDataset(d["synset_val"], False, d["clahe"], max_n)
    nw = 0 if smoke else cfg["training"]["num_workers"]
    n = 4 * u["batch_size"] if smoke else u["epoch_bilder"]
    tl = DataLoader(train, u["batch_size"], sampler=RandomSampler(train, replacement=True, num_samples=n), num_workers=nw)
    vl = DataLoader(val, u["batch_size"], num_workers=nw)
    model = UNetResNet18(4, u["pretrained"] and not smoke).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=u["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    epochs = 1 if smoke else u["num_epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gewichte = torch.tensor(u["klassen_gewichte"], device=device)
    best, verlauf = -1, []
    for e in range(1, epochs + 1):
        model.train()
        t0, summe, nb = time.time(), 0.0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y, weight=gewichte)
            loss.backward()
            opt.step()
            summe, nb = summe + loss.item(), nb + 1
        sched.step()
        res = iou_auswerten(model, vl, device, 3 if smoke else None)
        print(f"U-Net Epoche {e}/{epochs} ({time.time() - t0:.0f} s): Verlust {summe / nb:.3f}, mIoU {res['miou']:.3f}, {res['iou']}")
        verlauf.append({"epoche": e, "verlust": summe / nb, **res})
        if res["miou"] > best:
            best = res["miou"]
            torch.save({"model": model.state_dict(), "epoche": e, "val": res}, cfg["output"]["unet"])
    return verlauf


# ---------------------------------------------------------------- Stufe 2: Zusatz-Klassifikator

@torch.no_grad()
def acc_auswerten(model, loader, device):
    model.eval()
    c = n = 0
    for x, y in loader:
        c += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        n += len(y)
    return c / n if n else None


def train_klassifikator(cfg, device, smoke):
    d, k = cfg["data"], cfg["klassifikator"]
    items = zusatz_einzel_items(build_records(allow_unreviewed=d["ungepruefte_trainingsausschnitte"]))
    val_reg = set(d["real_val_regionen"])
    train = [i for i in items if i["split"] == "train" and i["region"] not in val_reg]
    val = [i for i in items if i["split"] == "train" and i["region"] in val_reg]
    if smoke:
        train, val = train[:64], val[:32]
    print(f"Zusatzschilder: Training {len(train)}, Validierung {len(val)}")
    nw = 0 if smoke else cfg["training"]["num_workers"]
    tl = DataLoader(ZusatzEinzelDataset(train, d["zusatz_size"], True, d["clahe"]), k["batch_size"], shuffle=True,
                    num_workers=nw)
    vl = DataLoader(ZusatzEinzelDataset(val, d["zusatz_size"], False, d["clahe"]), k["batch_size"], num_workers=nw)
    model = ZusatzKlassifikator(len(MODELL_KLASSEN), k["pretrained"] and not smoke).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=k["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    epochs = 1 if smoke else k["num_epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, verlauf = -1, []
    for e in range(1, epochs + 1):
        model.train()
        summe, nb = 0.0, 0
        for x, y in tl:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(device)), y.to(device), label_smoothing=k["label_smoothing"])
            loss.backward()
            opt.step()
            summe, nb = summe + loss.item(), nb + 1
        sched.step()
        acc = acc_auswerten(model, vl, device) if len(val) else 0.0
        print(f"Klassifikator Epoche {e}/{epochs}: Verlust {summe / max(nb, 1):.3f}, Val-Accuracy {acc:.3f}")
        verlauf.append({"epoche": e, "verlust": summe / max(nb, 1), "val_acc": acc})
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "epoche": e, "val_acc": acc, "klassen": MODELL_KLASSEN},
                       cfg["output"]["klassifikator"])
    return verlauf


# ---------------------------------------------------------------- Pipeline

@torch.no_grad()
def pipeline(unet, klassifikator, rgb, size, zusatz_size, clahe, device):
    """Ein Ausschnitt (RGB) -> Liste der Zusatzzeichen-Klassen (leer = keins)."""
    h, w = rgb.shape[:2]
    eingabe = _CLAHE(rgb) if clahe else rgb
    karte = unet(to_tensor(letterbox(eingabe, size)).to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8)
    # Box aus der Letterbox-Karte zurück in Ausschnitt-Koordinaten
    side = max(h, w)
    s, ox, oy = side / size, (side - w) / 2, (side - h) / 2
    klassen = []
    for x0, y0, x1, y1 in zusatz_boxen(karte):
        box = (x0 * s - ox, y0 * s - oy, x1 * s - ox, y1 * s - oy)
        x = to_tensor(ausschnitt(rgb, box, zusatz_size, clahe)).to(device)
        klassen.append(MODELL_KLASSEN[int(klassifikator(x).argmax(1))])
    return list(dict.fromkeys(klassen)), karte


def auswerten(cfg, device, smoke):
    d = cfg["data"]
    unet = UNetResNet18(4, False).to(device).eval()
    unet.load_state_dict(torch.load(cfg["output"]["unet"], map_location=device)["model"])
    kl = ZusatzKlassifikator(len(MODELL_KLASSEN), False).to(device).eval()
    kl.load_state_dict(torch.load(cfg["output"]["klassifikator"], map_location=device)["model"])
    test = [r for r in build_records() if r["split"] == "test" and r["geprueft"]]
    if smoke:
        test = test[:20]
    n = da_ok = exakt = n_rel = zg_rel = 0
    for r in test:
        rgb = cv2.cvtColor(cv2.imdecode(np.fromfile(r["crop"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        pred, _ = pipeline(unet, kl, rgb, d["image_size"], d["zusatz_size"], d["clahe"], device)
        wahr = [k for k in r["klassen"] if k != "keins"]
        n += 1
        da_ok += bool(pred) == bool(wahr)
        exakt += sorted(pred) == sorted(wahr)
        gefahr = ist_gefahrzeichen(r["haupt_synset"] or r["mapillary_haupt"])
        r_wahr = regel("", wahr or ["keins"], gefahr)
        if r_wahr["gilt_fuer"] != ["alle"] or r_wahr["ausgenommen"]:
            n_rel += 1
            zg_rel += zielgruppe_gleich(regel("", pred or ["keins"], gefahr), r_wahr)
    res = {"n": n, "da_acc": da_ok / n if n else None, "zusatz_exakt": exakt / n if n else None,
           "zielgruppe_korrekt_relevant": zg_rel / n_rel if n_rel else None, "n_zielgruppe_relevant": n_rel}
    print(f"Pipeline B auf dem echten Test (n = {n}): " + ", ".join(
        f"{k} {v:.3f}" for k, v in res.items() if isinstance(v, float)))
    return res


def main(args):
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.smoke:
        for k in ("unet", "klassifikator"):
            cfg["output"][k] = cfg["output"][k].replace(".pth", "_smoke.pth")
        cfg["output"]["results_dir"] += "_smoke"
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    os.makedirs(os.path.dirname(cfg["output"]["unet"]), exist_ok=True)
    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    print(f"Gerät: {device}")
    ergebnis = {}
    if args.schritt in ("unet", "alles"):
        ergebnis["unet"] = train_unet(cfg, device, args.smoke)
    if args.schritt in ("klassifikator", "alles"):
        ergebnis["klassifikator"] = train_klassifikator(cfg, device, args.smoke)
    if args.schritt in ("auswerten", "alles"):
        ergebnis["echt_test"] = auswerten(cfg, device, args.smoke)
    out = os.path.join(cfg["output"]["results_dir"], f"{args.schritt}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ergebnis, f, ensure_ascii=False, indent=1)
    print(f"Ergebnis: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Architektur B (zweistufig) trainieren und auswerten")
    p.add_argument("schritt", choices=["unet", "klassifikator", "auswerten", "alles"])
    p.add_argument("--config", default=os.path.join("configs", "zweistufig.yaml"))
    p.add_argument("--smoke", action="store_true", help="kurzer Probelauf auf der CPU")
    main(p.parse_args())
