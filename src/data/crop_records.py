"""
Echte Ausschnitte (crops/) mit Split und Labels, gemeinsam genutzt von review_crops.py und dem Training.

Split nach Ort: Ausschnitte aus Aachen, Eifel und Köln sind Test, alles andere Training.
Labels: die von Hand geprüften aus crops/geprueft.csv; auf Wunsch für noch ungeprüfte
Trainingsausschnitte die Vorschläge aus der Mapillary-Zuordnung (Test immer nur geprüft).
"""

import csv
import os

from .zusatz_klassen import KLASSEN

LABELS = os.path.join("crops", "labels.csv")
REVIEW = os.path.join("crops", "geprueft.csv")
MAPPING = os.path.join("configs", "mapillary_to_synset.csv")
SOURCE_LABELS = os.path.join("real_world_mapillary", "labels.csv")
TEST_REGIONEN = ["aachen", "eifel_monschau", "koeln"]

VALUE_TO_CLASS = {v: name for name, vals in KLASSEN for v in vals}

# Mapillary-Werte, die auf deutschen Fotos gar kein Zusatzzeichen sind (geprüft am 20.11.2026):
# "dangerous-or-pollutant-good" = Hinweiszeichen 354 "Wasserschutzgebiet". Solche Ausschnitte werden übersprungen.
IGNORIERTE_WERTE = {"complementary--dangerous-or-pollutant-good--g1"}


def class_of(value):
    if value in ("none", ""):
        return "keins"
    return VALUE_TO_CLASS.get(value, "andere")


def load_mapping(path=MAPPING):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["mapillary_wert"]: r for r in csv.DictReader(f)}


def load_locations(path=SOURCE_LABELS):
    """Originalbild -> (Region, Länge, Breite) aus real_world_mapillary/labels.csv."""
    loc = {}
    if not os.path.exists(path):
        return loc
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for r in reader:
            if len(r) == 11:
                loc[r[0]] = (r[1], r[5], r[6])
            elif len(r) == 10:                       # ältere Zeilen ohne Spalte region
                loc[r[0]] = ("", r[4], r[5])
    return loc


def test_boxes():
    from scrape_mapillary import MAPILLARY_REGIONS, region_bbox
    return [region_bbox(*MAPILLARY_REGIONS[r]) for r in TEST_REGIONEN]


def split_of(region, lon, lat, boxes):
    if region in TEST_REGIONEN:
        return "test"
    try:
        lon, lat = float(lon), float(lat)
    except ValueError:
        return "train"
    return "test" if any(w <= lon <= e and s <= lat <= n for w, s, e, n in boxes) else "train"


def load_reviews(path=REVIEW):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done[r["crop"]] = r                  # letzter Eintrag gilt
    return done


def build_records(allow_unreviewed=False):
    """Alle nutzbaren echten Ausschnitte als Liste von dicts.

    Felder: crop, split, region, haupt_synset ('' = unbekannt), klassen (Liste), geprueft (bool).
    """
    if not os.path.exists(LABELS):
        return []
    mapping, loc, boxes, reviews = load_mapping(), load_locations(), test_boxes(), load_reviews()
    with open(LABELS, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["crop"] and r["methode"].startswith("mapillary")]
    records = []
    for r in rows:
        if set(r["zusatzzeichen"].split("|")) <= IGNORIERTE_WERTE:
            continue
        region, lon, lat = loc.get(r["original"], ("", "", ""))
        split = split_of(region, lon, lat, boxes)
        rev = reviews.get(r["crop"])
        if rev:
            if rev["status"] == "verworfen":
                continue
            klassen = list(dict.fromkeys(rev["zusatz_klassen"].split("|")))   # doppelte Klassen einmal
            haupt_falsch = rev["hauptschild_falsch"] == "1"
        elif allow_unreviewed and split == "train":
            klassen = list(dict.fromkeys(class_of(v) for v in r["zusatzzeichen"].split("|")))
            haupt_falsch = False
        else:
            continue
        m = mapping.get(r["hauptschild"], {})
        haupt = m.get("synset_wert", "") if m.get("sicherheit") in ("sicher", "wahrscheinlich") else ""
        # Pfade wurden unter Windows mit "\" geschrieben; "/" funktioniert auf Windows und Linux (Colab)
        records.append({"crop": r["crop"].replace("\\", "/"), "original": r["original"].replace("\\", "/"),
                        "split": split,
                        "region": region or "?", "haupt_synset": "" if haupt_falsch else haupt,
                        "klassen": klassen, "geprueft": rev is not None,
                        "mapillary_haupt": r["hauptschild"], "mapillary_zusatz": r["zusatzzeichen"],
                        # Ausschnitt im Originalfoto (Pixel) und Originalpfad wie in den Labeldateien (für Boxen)
                        "box": tuple(int(r[k]) for k in ("x0", "y0", "x1", "y1")), "original_roh": r["original"]})
    return records
