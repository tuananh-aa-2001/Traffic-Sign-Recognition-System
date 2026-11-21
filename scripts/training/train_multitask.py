"""
Architektur A trainieren: ResNet-18 mit drei Köpfen auf Synset und echten Ausschnitten.

Vorher:
  python scripts/daten/prepare_synset.py --split train
  python scripts/daten/prepare_synset.py --split validation
  python scripts/daten/crop_signs.py mapillary        (und möglichst: python scripts/daten/review_crops.py)

Beispiele:
  python scripts/training/train_multitask.py                       # Training nach configs/multitask.yaml
  python scripts/training/train_multitask.py --smoke               # Probelauf auf der CPU mit wenigen Bildern
  python scripts/training/train_multitask.py --nur-auswerten       # bestes Modell auf Test und Synset-Validierung auswerten

Ergebnisse: models/checkpoints/multitask_best.pth und results/multitask/ergebnis.json
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # Projektordner (src/)
import argparse
import json
import os
import random
import sys
import time
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from src.data.crop_records import build_records
from src.data.multitask_data import RealCropDataset, SynsetDataset, load_synset_klassen
from src.data.zusatz_klassen import MODELL_KLASSEN
from src.models.multitask import MaskedMultiTaskLoss, MultiTaskResNet
from src.training.zusatz_metriken import zusatz_kennzahlen


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- Auswertung

@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """Kennzahlen je Kopf. Zusatzzeichen: Schwelle 0,5; 'keins' = keine Klasse über der Schwelle."""
    model.eval()
    k = len(MODELL_KLASSEN)
    n_h = c_h = n_d = c_d = 0
    preds, trues, gefahren = [np.zeros((0, k))], [np.zeros((0, k))], []
    for b, (x, t) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        out = model(x.to(device))
        t = {key: v.to(device) for key, v in t.items()}
        m = t["m_haupt"] > 0
        n_h += int(m.sum())
        c_h += int((out["haupt"][m].argmax(1) == t["haupt"][m]).sum())
        m = t["m_da"] > 0
        n_d += int(m.sum())
        c_d += int(((out["da"][m] > 0).float() == t["da"][m]).sum())
        m = t["m_zusatz"] > 0
        if m.any():
            preds.append((torch.sigmoid(out["zusatz"][m]) > 0.5).float().cpu().numpy())
            trues.append(t["zusatz"][m].cpu().numpy())
            gefahren.extend(t["gefahr"][m].cpu().numpy().tolist())
    res = {"haupt_acc": c_h / n_h if n_h else None, "n_haupt": n_h,
           "da_acc": c_d / n_d if n_d else None, "n_da": n_d}
    res.update(zusatz_kennzahlen(np.concatenate(preds), np.concatenate(trues), gefahren, MODELL_KLASSEN))
    return res


def fmt(res):
    parts = []
    for key in ("haupt_acc", "da_acc", "zusatz_exakt", "zusatz_macro_f1", "zielgruppe_korrekt_relevant"):
        if res.get(key) is not None:
            parts.append(f"{key} {res[key]:.3f}")
    return ", ".join(parts) or "keine Daten"


# ---------------------------------------------------------------- Daten

def begrenze_je_klasse(records, n, seed):
    """E4: höchstens n echte Trainingsausschnitte je Zusatzzeichen-Klasse ("keins" zählt als Klasse).

    Zufällige Reihenfolge (fester Seed), ein Ausschnitt wird nur genommen, wenn keine seiner Klassen
    das Limit schon erreicht hat. So bleibt jede Klasse <= n, seltene Klassen behalten alle Beispiele.
    """
    rng = random.Random(seed)
    reihe = list(records)
    rng.shuffle(reihe)
    zahl, auswahl = Counter(), []
    for r in reihe:
        if n > 0 and all(zahl[k] < n for k in r["klassen"]):
            auswahl.append(r)
            zahl.update(r["klassen"])
    return auswahl


def build_data(cfg, smoke):
    d = cfg["data"]
    size, clahe = d["image_size"], d["clahe"]
    records = build_records(allow_unreviewed=d["ungepruefte_trainingsausschnitte"])
    val_reg = set(d["real_val_regionen"])
    real = {
        "train": [r for r in records if r["split"] == "train" and r["region"] not in val_reg],
        "val": [r for r in records if r["split"] == "train" and r["region"] in val_reg],
        "test": [r for r in records if r["split"] == "test" and r["geprueft"]],
    }
    if d.get("echte_je_klasse") is not None:
        real["train"] = begrenze_je_klasse(real["train"], d["echte_je_klasse"], cfg["seed"])
        print(f"E4: höchstens {d['echte_je_klasse']} echte Trainingsausschnitte je Zusatzzeichen")
    print("Echte Ausschnitte: " + ", ".join(f"{k} {len(v)}" for k, v in real.items())
          + f" (davon geprüft im Training: {sum(r['geprueft'] for r in real['train'])})")

    max_n = 300 if smoke else None
    syn_train_split = d["synset_val"] if smoke else d["synset_train"]
    sets = {}
    if os.path.exists(os.path.join("data", "synset", f"{syn_train_split}_bilder.npy")):
        sets["syn_train"] = SynsetDataset(syn_train_split, size, True, clahe, max_n)
    else:
        print(f"Warnung: Synset '{syn_train_split}' fehlt (scripts/daten/prepare_synset.py), trainiere nur auf echten Ausschnitten")
    if os.path.exists(os.path.join("data", "synset", f"{d['synset_val']}_bilder.npy")):
        sets["syn_val"] = SynsetDataset(d["synset_val"], size, False, clahe, max_n)
    sets["real_train"] = RealCropDataset(real["train"], size, True, clahe)
    sets["real_val"] = RealCropDataset(real["val"], size, False, clahe)
    sets["real_test"] = RealCropDataset(real["test"], size, False, clahe)
    return sets


def train_loader(sets, cfg, smoke):
    d, t = cfg["data"], cfg["training"]
    parts = [s for s in (sets.get("syn_train"), sets["real_train"]) if s is not None and len(s)]
    concat = ConcatDataset(parts)
    if len(parts) == 2:
        n_syn, n_real = len(parts[0]), len(parts[1])
        a = d["real_anteil"]
        weights = [(1 - a) / n_syn] * n_syn + [a / n_real] * n_real
    else:
        weights = [1.0] * len(concat)
    n = 6 * t["batch_size"] if smoke else d["epoch_bilder"]
    sampler = WeightedRandomSampler(weights, num_samples=n, replacement=True)
    return DataLoader(concat, batch_size=t["batch_size"], sampler=sampler,
                      num_workers=0 if smoke else t["num_workers"], pin_memory=torch.cuda.is_available())


def eval_loader(ds, cfg, smoke):
    return DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                      num_workers=0 if smoke else cfg["training"]["num_workers"])


# ---------------------------------------------------------------- Ablauf

def main(args):
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.smoke:
        cfg["training"].update(batch_size=16, num_epochs=1)
        cfg["model"]["pretrained"] = False      # kein Download im Probelauf
        cfg["output"]["checkpoint"] = os.path.join("models", "checkpoints", "multitask_smoke.pth")
        cfg["output"]["results_dir"] = os.path.join("results", "multitask_smoke")
    if args.epochen:
        cfg["training"]["num_epochs"] = args.epochen
    if args.echte_je_klasse is not None:
        cfg["data"]["echte_je_klasse"] = args.echte_je_klasse
    if args.ausgabe:
        cfg["output"]["results_dir"] = args.ausgabe
        cfg["output"]["checkpoint"] = os.path.join(args.ausgabe, "modell.pth")
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    print(f"Gerät: {device}")

    sets = build_data(cfg, args.smoke)
    n_haupt = len(load_synset_klassen()["hauptschilder"])
    model = MultiTaskResNet(n_haupt, len(MODELL_KLASSEN), cfg["model"]["pretrained"], cfg["model"]["dropout"]).to(device)
    ckpt_path = cfg["output"]["checkpoint"]
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    max_eval = 3 if args.smoke else None

    if not args.nur_auswerten:
        loss_fn = MaskedMultiTaskLoss(tuple(cfg["loss"]["gewichte"]), cfg["loss"]["label_smoothing"],
                                      cfg["loss"]["focal_gamma"])
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                weight_decay=cfg["training"]["weight_decay"])
        epochs = cfg["training"]["num_epochs"]
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        use_amp = device.type == "cuda"
        scaler = GradScaler("cuda") if use_amp else None
        loader = train_loader(sets, cfg, args.smoke)
        best, history = -1.0, []
        for epoch in range(1, epochs + 1):
            model.train()
            t0, summe, n = time.time(), {"haupt": 0.0, "da": 0.0, "zusatz": 0.0}, 0
            for x, t in loader:
                x = x.to(device)
                t = {k: v.to(device) for k, v in t.items()}
                opt.zero_grad(set_to_none=True)
                with autocast(device.type, enabled=use_amp):
                    loss, teile = loss_fn(model(x), t)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                for k in summe:
                    summe[k] += teile[k]
                n += 1
            sched.step()
            syn = evaluate(model, eval_loader(sets["syn_val"], cfg, args.smoke), device, max_eval) if "syn_val" in sets else {}
            real = evaluate(model, eval_loader(sets["real_val"], cfg, args.smoke), device, max_eval) if len(sets["real_val"]) else {}
            # Auswahl: Mittel aus Hauptschild-Genauigkeit (Synset) und exakter Zusatz-Treffer (echt)
            werte = [v for v in (syn.get("haupt_acc"), real.get("zusatz_exakt")) if v is not None]
            score = float(np.mean(werte)) if werte else 0.0
            print(f"Epoche {epoch}/{epochs} ({time.time() - t0:.0f} s) Verlust "
                  + ", ".join(f"{k} {v / max(n, 1):.3f}" for k, v in summe.items()))
            print(f"  Synset-Val: {fmt(syn)} | Echt-Val: {fmt(real)} | Auswahlwert {score:.3f}")
            history.append({"epoche": epoch, "verlust": {k: v / max(n, 1) for k, v in summe.items()},
                            "synset_val": syn, "echt_val": real, "auswahlwert": score})
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "config": cfg, "epoche": epoch,
                            "haupt_klassen": load_synset_klassen()["hauptschilder"],
                            "zusatz_klassen": MODELL_KLASSEN}, ckpt_path)
                print(f"  bestes Modell gespeichert: {ckpt_path}")
        with open(os.path.join(cfg["output"]["results_dir"], "verlauf.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=1)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    ergebnis = {"epoche": state["epoche"], "echte_je_klasse": cfg["data"].get("echte_je_klasse"),
                "n_echt_train": len(sets["real_train"]),
                "echt_test": evaluate(model, eval_loader(sets["real_test"], cfg, args.smoke), device, max_eval)
                if len(sets["real_test"]) else {},
                "synset_val": evaluate(model, eval_loader(sets["syn_val"], cfg, args.smoke), device, max_eval)
                if "syn_val" in sets else {}}
    print(f"\nTest (echt, Aachen/Eifel/Köln): {fmt(ergebnis['echt_test'])}")
    print(f"Synset-Validierung: {fmt(ergebnis['synset_val'])}")
    out = os.path.join(cfg["output"]["results_dir"], "ergebnis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ergebnis, f, ensure_ascii=False, indent=1)
    print(f"Ergebnis: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Architektur A (Multi-Task) trainieren")
    p.add_argument("--config", default=os.path.join("configs", "multitask.yaml"))
    p.add_argument("--smoke", action="store_true", help="kurzer Probelauf auf der CPU")
    p.add_argument("--epochen", type=int, default=None)
    p.add_argument("--nur-auswerten", action="store_true")
    p.add_argument("--echte-je-klasse", type=int, default=None,
                   help="E4: höchstens so viele echte Trainingsausschnitte je Zusatzzeichen (0 = nur Synset)")
    p.add_argument("--ausgabe", default=None, help="Ergebnisordner (Checkpoint landet darin)")
    main(p.parse_args())
