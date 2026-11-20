"""
Prüfwerkzeug für die ausgeschnittenen Schildkombinationen (crops/labels.csv).

Zeigt jeden Ausschnitt mit dem vorgeschlagenen Zusatzzeichen (aus configs/mapillary_to_synset.csv)
und dem Foto drumherum. Mit einer Taste bestätigen, korrigieren oder verwerfen. Jede Entscheidung
wird sofort an crops/geprueft.csv angehängt (letzter Eintrag je Ausschnitt gilt), daher kann man
jederzeit abbrechen und weitermachen. crops/labels.csv wird nie verändert.

Split nach Ort: Ausschnitte aus Aachen, Eifel und Köln sind Test, alles andere Training.

Tasten:
  Enter / Leertaste   Vorschlag stimmt
  1-9, a-t            anderes Zusatzzeichen (ersetzt den Vorschlag)
  +  dann Taste       weiteres Zusatzzeichen anhängen (gestapelte Schilder)
  h                   Hauptschild als falsch markieren (an/aus)
  x                   verwerfen (Box falsch, unlesbar, kein Verkehrszeichen)
  Rücktaste / z       zurück zum vorigen Ausschnitt
  q / Esc             beenden

Beispiele:
  python review_crops.py                    # erst Test, dann Training; Ausschnitte mit Zusatzzeichen
  python review_crops.py --split test
  python review_crops.py --mit-none         # auch Hauptschilder ohne Zusatzzeichen prüfen
  python review_crops.py --statistik        # Zählung je Klasse und Split, ohne Fenster
  python review_crops.py --nachpruefen "andere,Länge einer Strecke"
                                            # schon geprüfte Ausschnitte dieser Klassen erneut zeigen
"""

import argparse
import collections
import csv
import os
import sys
import time

import cv2
import numpy as np

from crop_signs import imread
from src.data.zusatz_klassen import KLASSEN
from src.data.crop_records import (LABELS, REVIEW, class_of, load_locations, load_mapping,
                                   load_reviews, split_of, test_boxes)

REVIEW_HEADER = ["crop", "split", "region", "hauptschild", "hauptschild_falsch",
                 "zusatz_klassen", "zusatz_synset", "status", "zeit"]

TASTEN = "123456789abcdefgijklmnoprst"       # ohne h, q, x, z
assert len(TASTEN) == len(KLASSEN)


# ---------------------------------------------------------------- Daten

def load_items(args, mapping):
    with open(LABELS, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["crop"] and r["methode"].startswith("mapillary")]
    loc = load_locations()
    boxes = test_boxes()
    rank = {"unsicher": 0, "kein_gegenstueck": 1, "wahrscheinlich": 2, "sicher": 3}
    items = []
    for r in rows:
        if r["zusatzzeichen"] == "none" and not args.mit_none:
            continue
        region, lon, lat = loc.get(r["original"], ("", "", ""))
        split = split_of(region, lon, lat, boxes)
        if args.split != "alle" and split != args.split:
            continue
        values = r["zusatzzeichen"].split("|")
        sicherheit = min((rank.get(mapping.get(v, {}).get("sicherheit"), 1) for v in values), default=3)
        items.append({**r, "region": region or "?", "split": split, "werte": values,
                      "vorschlag": [class_of(v) for v in values], "rang": sicherheit})
    # Test zuerst; innerhalb unsichere Zuordnungen zuerst; Hauptschilder ohne Zusatz zuletzt
    items.sort(key=lambda it: (it["split"] != "test", it["zusatzzeichen"] == "none", it["rang"], it["crop"]))
    return items


# ---------------------------------------------------------------- Anzeige

def fit(img, w, h):
    s = min(w / img.shape[1], h / img.shape[0])
    out = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.full((h, w, 3), 40, np.uint8)
    y, x = (h - out.shape[0]) // 2, (w - out.shape[1]) // 2
    canvas[y:y + out.shape[0], x:x + out.shape[1]] = out
    return canvas


def put(img, text, x, y, scale=0.5, color=(235, 235, 235), bold=1):
    # cv2.putText kann keine Umlaute
    text = text.translate(str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}))
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)


def render(item, state, idx, total, synset_of_class):
    crop = imread(item["crop"])
    left = np.vstack([fit(crop, 420, 380) if crop is not None else np.full((380, 420, 3), 40, np.uint8),
                      context_view(item)])
    panel = np.full((600, 560, 3), 28, np.uint8)
    put(panel, f"{idx + 1}/{total}  {item['split'].upper()}  {item['region']}", 14, 26, 0.6, (0, 220, 255), 2)
    put(panel, "Haupt: " + item["hauptschild"].replace("regulatory--", "").replace("warning--", "")
        .replace("information--", ""), 14, 52, 0.48)
    put(panel, "Mapillary: " + " | ".join(v.replace("complementary--", "") for v in item["werte"]), 14, 74, 0.48)
    color = (80, 220, 80) if state["klassen"] == item["vorschlag"] else (60, 180, 255)
    put(panel, "Auswahl: " + " + ".join(state["klassen"]), 14, 102, 0.6, color, 2)
    if state["haupt_falsch"]:
        put(panel, "Hauptschild als FALSCH markiert", 14, 126, 0.5, (60, 60, 255), 2)
    if state["plus"]:
        put(panel, "+ : naechste Klasse wird angehaengt", 14, 126, 0.5, (60, 180, 255), 1)
    y = 154
    for i, (name, _) in enumerate(KLASSEN):
        col, row = i // 14, i % 14
        mark = name in state["klassen"]
        c = (80, 220, 80) if mark else (210, 210, 210)
        syn = synset_of_class.get(name, "")
        put(panel, f"{TASTEN[i]}  {name}", 14 + col * 275, y + row * 27, 0.46, c, 2 if mark else 1)
        if syn:
            put(panel, syn[:30], 34 + col * 275, y + row * 27 + 12, 0.33, (150, 150, 150))
    put(panel, "Enter ok | Taste andere Klasse | + anhaengen | h Haupt falsch", 14, 545, 0.42, (180, 180, 180))
    put(panel, "x verwerfen | z zurueck | q beenden", 14, 568, 0.42, (180, 180, 180))
    return np.hstack([left, panel])


def context_view(item):
    """Ausschnitt mit Umgebung aus dem Originalfoto, damit man Mast und Nachbarschilder sieht."""
    img = imread(item["original"]) if os.path.exists(item["original"]) else None
    if img is None:
        return np.full((220, 420, 3), 40, np.uint8)
    x0, y0, x1, y1 = (int(item[k]) for k in ("x0", "y0", "x1", "y1"))
    w, h = x1 - x0, y1 - y0
    cx, cy, r = (x0 + x1) // 2, (y0 + y1) // 2, int(max(w, h) * 1.6)
    H, W = img.shape[:2]
    a0, b0, a1, b1 = max(0, cx - 2 * r), max(0, cy - r), min(W, cx + 2 * r), min(H, cy + r)
    view = img[b0:b1, a0:a1].copy()
    cv2.rectangle(view, (x0 - a0, y0 - b0), (x1 - a0, y1 - b0), (0, 220, 255), 2)
    return fit(view, 420, 220)


# ---------------------------------------------------------------- Ablauf

def save(writer, f, item, state, status, synset_of_class):
    writer.writerow([item["crop"], item["split"], item["region"], item["hauptschild"],
                     int(state["haupt_falsch"]), "|".join(state["klassen"]),
                     "|".join(synset_of_class.get(k, "") for k in state["klassen"]),
                     status, time.strftime("%Y-%m-%d %H:%M:%S")])
    f.flush()


def review(args, keys=None):
    mapping = load_mapping()
    synset_of_class = {}
    for name, vals in KLASSEN:
        synset_of_class[name] = next((mapping[v]["synset_wert"] for v in vals
                                      if v in mapping and mapping[v]["synset_wert"]), "")
    synset_of_class["keins"] = "None"
    items = load_items(args, mapping)
    done = load_reviews()
    if args.nachpruefen:
        wanted = {k.strip() for k in args.nachpruefen.split(",")}
        unknown = wanted - {name for name, _ in KLASSEN}
        if unknown:
            sys.exit(f"Unbekannte Klasse(n): {', '.join(sorted(unknown))}")
        todo = [i for i, it in enumerate(items) if it["crop"] in done
                and wanted & set(done[it["crop"]]["zusatz_klassen"].split("|"))]
        print(f"{len(todo)} geprüfte Ausschnitte der Klassen {', '.join(sorted(wanted))} zum Nachprüfen.")
    else:
        todo = [i for i, it in enumerate(items) if it["crop"] not in done]
        print(f"{len(items)} Ausschnitte, davon {len(items) - len(todo)} schon geprüft.")
    if not todo:
        return

    new = not os.path.exists(REVIEW)
    f = open(REVIEW, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new:
        writer.writerow(REVIEW_HEADER)
    key_iter = iter(keys) if keys is not None else None
    win = "Ausschnitte pruefen"
    pos, history = 0, []
    with f:
        while pos < len(todo):
            item = items[todo[pos]]
            prev = done.get(item["crop"])
            klassen = prev["zusatz_klassen"].split("|") if prev else list(item["vorschlag"])
            if prev and klassen == ["andere"] and item["vorschlag"] != ["andere"]:
                klassen = list(item["vorschlag"])   # neue Klasse in der Liste passt jetzt
            state = {"klassen": klassen,
                     "haupt_falsch": bool(prev and prev["hauptschild_falsch"] == "1"), "plus": False}
            while True:
                frame = render(item, state, todo[pos], len(items), synset_of_class)
                if key_iter is None:
                    cv2.imshow(win, frame)
                    k = cv2.waitKey(0) & 0xFF
                else:
                    k = next(key_iter, ord("q"))
                    if isinstance(k, str):
                        k = ord(k)
                    args.letztes_bild = frame
                ch = chr(k) if 32 <= k < 127 else ""
                if k in (ord("q"), 27):
                    if key_iter is None:
                        cv2.destroyAllWindows()
                    print(f"Beendet bei {pos}/{len(todo)} offenen Ausschnitten.")
                    return
                if k in (13, 32):
                    status = "ok" if state["klassen"] == item["vorschlag"] and not state["haupt_falsch"] else "korrigiert"
                    if prev and status == "ok":
                        status = prev["status"]        # beim Nachprüfen unverändert lassen
                    break
                if ch == "x":
                    status = "verworfen"
                    break
                if k in (8, ord("z")):
                    if history:
                        pos = history.pop()
                        done = load_reviews()
                    status = None
                    break
                if ch == "h":
                    state["haupt_falsch"] = not state["haupt_falsch"]
                elif ch == "+":
                    state["plus"] = True
                elif ch and ch in TASTEN:
                    name = KLASSEN[TASTEN.index(ch)][0]
                    if state["plus"] and name not in state["klassen"]:
                        state["klassen"] = [c for c in state["klassen"] if c != "keins"] + [name]
                    else:
                        state["klassen"] = [name]
                    state["plus"] = False
            if status is None:
                continue
            save(writer, f, item, state, status, synset_of_class)
            done[item["crop"]] = {"zusatz_klassen": "|".join(state["klassen"]), "status": status,
                                  "hauptschild_falsch": str(int(state["haupt_falsch"]))}
            history.append(pos)
            pos += 1
    if key_iter is None:
        cv2.destroyAllWindows()
    print("Alle Ausschnitte dieser Auswahl sind geprüft.")


def statistik():
    done = load_reviews()
    if not done:
        print("Noch nichts geprüft.")
        return
    status = collections.Counter(r["status"] for r in done.values())
    print("Status:", dict(status))
    counts = collections.Counter()
    for r in done.values():
        if r["status"] == "verworfen":
            continue
        for k in r["zusatz_klassen"].split("|"):
            counts[(k, r["split"])] += 1
    print(f"\n{'Klasse':30} {'train':>6} {'test':>6}")
    for name, _ in KLASSEN:
        tr, te = counts[(name, "train")], counts[(name, "test")]
        if tr or te:
            print(f"{name:30} {tr:6} {te:6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ausgeschnittene Schildkombinationen prüfen")
    parser.add_argument("--split", choices=["test", "train", "alle"], default="alle")
    parser.add_argument("--mit-none", action="store_true", help="auch Hauptschilder ohne Zusatzzeichen")
    parser.add_argument("--statistik", action="store_true", help="nur Zählung ausgeben")
    parser.add_argument("--nachpruefen", default=None,
                        help="kommagetrennte Klassen: schon geprüfte Ausschnitte dieser Klassen erneut zeigen")
    args = parser.parse_args()
    if not os.path.exists(LABELS):
        sys.exit(f"{LABELS} fehlt. Zuerst: python crop_signs.py mapillary")
    statistik() if args.statistik else review(args)
