"""
Hauptschild nachlabeln für Ausschnitte, bei denen Mapillary das Hauptschild nicht erkannt hat
(mapillary_haupt = 'unbekannt', alle aus der Rettung über die Farbe).

Ohne Label wird der Hauptschild-Kopf für diese Ausschnitte maskiert, und die Hauptschild-Genauigkeit
auf dem Test zählt nur 490 von 651 Bildern. Jede Entscheidung wird sofort an
crops/hauptschild_nachgelabelt.csv angehängt (letzter Eintrag je Ausschnitt gilt); crop_records.py
übernimmt sie beim nächsten Training.

Tasten:
  1-9, a-i     Synset-Klasse aus der Liste (die häufigsten Hauptschilder dieser Ausschnitte)
  n            andere Synset-Klasse im Terminal eintippen (Teil des Namens genügt)
  k            keine passende Synset-Klasse (z. B. Umleitung U5, Verbot für Krafträder und Kraftwagen)
  z / Rücktaste  zurück
  q / Esc      beenden

  python scripts/daten/hauptschild_nachlabeln.py               # erst Test (138), dann Training (420)
  python scripts/daten/hauptschild_nachlabeln.py --split test
  python scripts/daten/hauptschild_nachlabeln.py --statistik
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # Projektordner (src/)
import argparse
import collections
import csv
import json
import os
import time

import cv2
import numpy as np

from review_crops import context_view, fit, imread, put
from src.data.crop_records import NACHGELABELT, build_records, load_nachgelabelt

KANDIDATEN = ["Parken", "BeginnFussgaengerzone", "EndeFussgaengerzone", "GemeinsamerGehUndRadweg", "Gehweg",
              "Radweg", "GetrennterRadUndGehwegRadwegRechts", "VerbotDerEinfahrt", "VerbotFahrzeugeAllerArt",
              "VerbotKraftfahrzeuge", "VerbotRadverkehr", "BeginnVerkehrsberuhigterBereich", "BeginnZone30",
              "EingeschraenktesHalteverbot", "AbsolutesHalteverbot", "BeginnFahrradstrasse", "Sackgasse",
              "EinbahnstrasseRechtsweisend"]
TASTEN = "123456789abcdefghi"
KEINE = "-"


def synset_namen():
    with open(os.path.join("configs", "synset_klassen.json"), encoding="utf-8") as f:
        k = json.load(f)
    werte = list(k.values())[-1] if isinstance(k, dict) else k
    return sorted(werte.values()) if isinstance(werte, dict) else sorted(werte)


def items(split):
    recs = [r for r in build_records(allow_unreviewed=True) if r["mapillary_haupt"] == "unbekannt"
            and (split == "alle" or r["split"] == split)]
    return sorted(recs, key=lambda r: (r["split"] != "test", r["crop"]))


def render(item, wahl, idx, total):
    crop = imread(item["crop"])
    x0, y0, x1, y1 = item["box"]
    ctx = context_view({"original": item["original"], "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    left = np.vstack([fit(crop, 420, 380) if crop is not None else np.full((380, 420, 3), 40, np.uint8), ctx])
    panel = np.full((600, 520, 3), 28, np.uint8)
    put(panel, f"{idx + 1}/{total}  {item['split'].upper()}  {item['region']}", 14, 26, 0.6, (0, 220, 255), 2)
    put(panel, "Zusatz: " + " + ".join(item["klassen"]), 14, 52, 0.48)
    put(panel, "Auswahl: " + (wahl if wahl is not None else "-"), 14, 80, 0.6,
        (80, 220, 80) if wahl else (60, 180, 255), 2)
    for i, name in enumerate(KANDIDATEN):
        put(panel, f"{TASTEN[i]}  {name}", 14, 112 + i * 23, 0.46, (80, 220, 80) if name == wahl else (210, 210, 210))
    put(panel, "n andere Klasse tippen | k keine Synset-Klasse", 14, 545, 0.42, (180, 180, 180))
    put(panel, "z zurueck | q beenden", 14, 568, 0.42, (180, 180, 180))
    return np.hstack([left, panel])


def eintippen(namen):
    while True:
        s = input("Synset-Klasse (Teil des Namens, leer = abbrechen): ").strip().lower()
        if not s:
            return None
        treffer = [n for n in namen if s in n.lower()]
        if len(treffer) == 1:
            return treffer[0]
        exakt = [n for n in treffer if n.lower() == s]
        if exakt:
            return exakt[0]
        print(f"{len(treffer)} Treffer: " + ", ".join(treffer[:15]))


def statistik():
    done = load_nachgelabelt()
    for split in ("test", "train"):
        it = items(split)
        werte = collections.Counter(done[r["crop"]] or "keine Klasse" for r in it if r["crop"] in done)
        print(f"{split}: {sum(r['crop'] in done for r in it)} von {len(it)} nachgelabelt  {werte.most_common(8)}")


def main(args):
    if args.statistik:
        return statistik()
    namen, done = synset_namen(), load_nachgelabelt()
    liste = items(args.split)
    i = next((k for k, r in enumerate(liste) if r["crop"] not in done), len(liste))
    neu = not os.path.exists(NACHGELABELT)
    with open(NACHGELABELT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if neu:
            w.writerow(["crop", "haupt_synset", "zeit"])
        cv2.namedWindow("Hauptschild", cv2.WINDOW_AUTOSIZE)
        while 0 <= i < len(liste):
            item = liste[i]
            cv2.imshow("Hauptschild", render(item, done.get(item["crop"]), i, len(liste)))
            key = cv2.waitKey(0) & 0xFF
            ch = chr(key) if key < 128 else ""
            wahl = None
            if key in (27,) or ch == "q":
                break
            if key == 8 or ch == "z":
                i = max(0, i - 1)
                continue
            if ch and ch in TASTEN[:len(KANDIDATEN)]:
                wahl = KANDIDATEN[TASTEN.index(ch)]
            elif ch == "k":
                wahl = KEINE
            elif ch == "n":
                wahl = eintippen(namen)
            if wahl is None:
                continue
            w.writerow([item["crop"], "" if wahl == KEINE else wahl, time.strftime("%Y-%m-%d %H:%M:%S")])
            f.flush()
            done[item["crop"]] = "" if wahl == KEINE else wahl
            i += 1
    cv2.destroyAllWindows()
    statistik()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hauptschild für Ausschnitte mit 'unbekannt' nachlabeln")
    p.add_argument("--split", choices=["test", "train", "alle"], default="alle")
    p.add_argument("--statistik", action="store_true")
    main(p.parse_args())
