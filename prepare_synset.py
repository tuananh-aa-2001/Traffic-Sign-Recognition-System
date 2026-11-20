"""
Synset Signset Germany für das Multi-Task-Training vorbereiten.

Liest die Parquet-Datei Zeilengruppe für Zeilengruppe (lokal oder direkt von Hugging Face,
ohne sie ganz zu laden), bringt jedes Bild quadratisch auf eine feste Größe und speichert:

  data/synset/<split>_bilder.npy    uint8-Array (N, S, S, 3), RGB
  data/synset/<split>_labels.csv    Index, Hauptschild, Schild darunter, Bedingungen
  configs/synset_klassen.json       alle Hauptschild-Namen und welche lower_sign-Werte Zusatzzeichen sind

Masken und Segmentierung werden nur mit --segmentierung geladen (für Architektur B):
  data/synset/<split>_klassen.npy   uint8-Array (N, S, S): 0 Hintergrund, 1 Hauptschild,
                                    2 andere Schilder, 3 Mast (src/data/segmentation.py)

Beispiele:
  python prepare_synset.py --split validation --max 2000        # kleiner Ausschnitt zum Testen
  python prepare_synset.py --split train                         # komplett, auf Colab (ca. 7,4 GB lesen)
  python prepare_synset.py --split train --quelle /content/cycles_train.parquet
"""

import argparse
import csv
import io
import json
import os
import time

import numpy as np
from PIL import Image

HF_URL = "https://huggingface.co/datasets/FraunhoferIOSB/Synset-Signset-Germany/resolve/main/cycles/{split}.parquet"
OUT_DIR = os.path.join("data", "synset")
KLASSEN_JSON = os.path.join("configs", "synset_klassen.json")
SPALTEN = ["image", "label", "class_name", "lower_sign", "upper_sign", "environment_day_time",
           "environment_weather", "environment_contrast", "global_blur_std_dev"]


def open_parquet(quelle):
    import pyarrow.parquet as pq
    if quelle.startswith("http"):
        import fsspec
        return pq.ParquetFile(fsspec.filesystem("https").open(quelle, block_size=2 ** 22))
    return pq.ParquetFile(quelle)


def letterbox(img, size):
    """Quadratisch auffüllen (Rand in Bildmittelwert) und skalieren."""
    w, h = img.size
    side = max(w, h)
    mean = tuple(int(v) for v in np.asarray(img).reshape(-1, 3).mean(axis=0))
    canvas = Image.new("RGB", (side, side), mean)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.BILINEAR)


# Zusatzzeichen, die Synset zugleich als eigene Hauptklasse führt (stehen in class_name und lower_sign)
ZUSATZ_AUCH_KLASSE = {"BeiNaesse", "CarsharingfahrzeugeFrei", "MehrfachbesetzePersonenkraftwagenFrei"}


def ist_zusatz_klasse(name):
    return name in ZUSATZ_AUCH_KLASSE or name.startswith("VerlaufVorfahrtsstrasse")


def letterbox_karte(karte, size):
    """Wie letterbox(), aber für Klassenkarten: Rand = Hintergrund (0), Skalierung ohne Interpolation."""
    w, h = karte.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(karte, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.NEAREST)


def write_klassen(pf):
    """Klassennamen aus allen Zeilengruppen lesen (nur die kleinen Textspalten)."""
    haupt, unten, label_namen = set(), set(), {}
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=["label", "class_name", "lower_sign"])
        haupt.update(t.column("class_name").to_pylist())
        unten.update(t.column("lower_sign").to_pylist())
        label_namen.update(zip(t.column("label").to_pylist(), t.column("class_name").to_pylist()))
    unten.discard("None")
    beides = {u for u in unten & haupt if ist_zusatz_klasse(u)}
    data = {
        "hauptschilder": sorted(haupt),
        # lower_sign enthält auch Hauptschilder (z. B. Tempo 30 unter einem Gefahrzeichen);
        # "zusatzzeichen" = alles, was als Zusatzschild zählt
        "zusatzzeichen": sorted((unten - haupt) | beides),
        "hauptschilder_unten": sorted((unten & haupt) - beides),
        # Synset-Label (0-210) -> Klassenname; 0-42 entsprechen laut Datensatzkarte den GTSRB-Klassen
        "label_namen": {str(k): v for k, v in sorted(label_namen.items())},
    }
    os.makedirs(os.path.dirname(KLASSEN_JSON), exist_ok=True)
    with open(KLASSEN_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"{KLASSEN_JSON}: {len(data['hauptschilder'])} Hauptschilder, "
          f"{len(data['zusatzzeichen'])} Zusatzzeichen-Werte")


def main(args):
    quelle = args.quelle or HF_URL.format(split=args.split)
    pf = open_parquet(quelle)
    n_total = pf.metadata.num_rows
    n = min(n_total, args.max) if args.max else n_total
    print(f"{quelle}: {n_total} Zeilen in {pf.metadata.num_row_groups} Zeilengruppen, nehme {n}")
    if args.split == "train" or not os.path.exists(KLASSEN_JSON):
        write_klassen(pf)

    os.makedirs(OUT_DIR, exist_ok=True)
    npy = os.path.join(OUT_DIR, f"{args.split}_bilder.npy")
    bilder = np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=(n, args.groesse, args.groesse, 3))
    karten = None
    if args.segmentierung:
        from src.data.segmentation import seg_zu_klassen
        karten = np.lib.format.open_memmap(os.path.join(OUT_DIR, f"{args.split}_klassen.npy"), mode="w+",
                                           dtype=np.uint8, shape=(n, args.groesse, args.groesse))
    spalten = SPALTEN + (["mask", "segmentation"] if args.segmentierung else [])
    rows, k, t0 = [], 0, time.time()
    # Bei --max nur wenige, gleichmäßig verteilte Zeilengruppen lesen (je bis 200 Bilder),
    # damit ein Testlauf nicht die ganze Datei herunterlädt und trotzdem mehrere Klassen enthält
    groups = list(range(pf.metadata.num_row_groups))
    if args.max:
        n_groups = min(len(groups), int(np.ceil(n / 200)))
        groups = sorted({int(round(x)) for x in np.linspace(0, len(groups) - 1, n_groups)})
    per_group = int(np.ceil(n / len(groups)))
    for g in groups:
        if k >= n:
            break
        t = pf.read_row_group(g, columns=spalten).to_pylist()
        if args.max:
            t = t[:per_group]
        for r in t:
            if k >= n:
                break
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            bilder[k] = np.asarray(letterbox(img, args.groesse))
            if karten is not None:
                seg = np.asarray(Image.open(io.BytesIO(r["segmentation"]["bytes"])).convert("RGB"))
                maske = np.asarray(Image.open(io.BytesIO(r["mask"]["bytes"])).convert("L"))
                karte = Image.fromarray(seg_zu_klassen(seg, maske))
                karten[k] = np.asarray(letterbox_karte(karte, args.groesse))
            rows.append([k, r["label"], r["class_name"], r["lower_sign"], r["upper_sign"],
                         r["environment_day_time"], r["environment_weather"],
                         r["environment_contrast"], round(r["global_blur_std_dev"] or 0, 3)])
            k += 1
        print(f"  Zeilengruppe {g + 1}/{len(groups)}: {k}/{n} Bilder, {time.time() - t0:.0f} s", flush=True)
    bilder.flush()
    if karten is not None:
        karten.flush()
    with open(os.path.join(OUT_DIR, f"{args.split}_labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "label", "hauptschild", "unten", "oben", "tageszeit", "wetter", "kontrast", "unschaerfe"])
        w.writerows(rows)
    print(f"Fertig: {k} Bilder in {npy} ({os.path.getsize(npy) / 1e9:.2f} GB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Synset Signset Germany vorbereiten")
    p.add_argument("--split", choices=["train", "validation"], default="validation")
    p.add_argument("--quelle", default=None, help="lokale Parquet-Datei statt Download von Hugging Face")
    p.add_argument("--max", type=int, default=None, help="höchstens so viele Bilder (zum Testen)")
    p.add_argument("--groesse", type=int, default=128, help="Kantenlänge in Pixeln")
    p.add_argument("--segmentierung", action="store_true",
                   help="auch Klassenkarten für das U-Net speichern (Architektur B, lädt mehr Daten)")
    main(p.parse_args())
