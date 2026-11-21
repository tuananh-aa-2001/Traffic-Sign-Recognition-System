"""
Echte Fotos von Verkehrszeichen-Kombinationen von Mapillary laden.

Drei Suchen:
  --quelle mapillary     21 Regionen in NRW (Großstädte, Eifel, Sauerland, Münsterland, OWL, Niederrhein):
                         alle Fotos, in denen Mapillary ein Zusatzzeichen ("complementary--...") erkannt hat
  --quelle zielgruppen   gezielt 14 Zusatzzeichen mit Zielgruppe (Rad frei, LKW, Bus, ...) in 66 Städten
                         in ganz Deutschland; die Testregionen Aachen, Eifel, Köln sind ausgeschlossen
  --quelle zielgruppen --testregionen   dieselben Zeichen nur in den Testregionen (Testsatz)

Zu jedem Foto werden alle erkannten Schilder mit Umriss gespeichert (Haupt- und Zusatzzeichen).
Ablage: real_world_mapillary/ mit labels.csv (Urheber, Lizenz CC BY-SA 4.0, Quelle je Foto;
in Folien und Bericht nennen).

Zugangsschlüssel (kostenlos): https://www.mapillary.com/dashboard/developers -> "Register application"
-> Client Token in die Datei mapillary_token.txt schreiben (oder $env:MAPILLARY_TOKEN = "MLY|...").

Aufruf:
  python scripts/daten/scrape_mapillary.py --quelle mapillary --max 600
  python scripts/daten/scrape_mapillary.py --quelle mapillary --regionen aachen,koeln --max 50
  python scripts/daten/scrape_mapillary.py --quelle zielgruppen
  python scripts/daten/scrape_mapillary.py --quelle zielgruppen --testregionen
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # Projektordner (src/)
import argparse
import base64
import csv
import json
import math
import os
import random
import re
import time

import requests

# ---------------------------------------------------------------- Einstellungen

# Mapillary: Regionen in NRW als (Breite, Länge, Radius in km): Großstädte
# plus ländliche Gebiete, damit Stadt, Land und Mittelgebirge vertreten sind.
# Mittelpunkte sind gerundet; Radius bei Bedarf anpassen.
MAPILLARY_REGIONS = {
    # Großstädte
    "aachen":          (50.776, 6.084, 5),
    "koeln":           (50.938, 6.960, 7),
    "duesseldorf":     (51.227, 6.774, 6),
    "dortmund":        (51.514, 7.468, 6),
    "essen":           (51.456, 7.012, 6),
    "duisburg":        (51.434, 6.762, 5),
    "bochum":          (51.482, 7.216, 5),
    "wuppertal":       (51.256, 7.150, 5),
    "bielefeld":       (52.021, 8.532, 5),
    "bonn":            (50.737, 7.098, 5),
    "muenster":        (51.962, 7.626, 5),
    "moenchengladbach": (51.185, 6.442, 4),
    "gelsenkirchen":   (51.517, 7.085, 4),
    "krefeld":         (51.339, 6.586, 4),
    "paderborn":       (51.719, 8.754, 4),
    "siegen":          (50.875, 8.024, 4),
    # Ländlich und Mittelgebirge
    "eifel_monschau":  (50.555, 6.240, 6),
    "sauerland_winterberg": (51.195, 8.533, 6),
    "muensterland_coesfeld": (51.946, 7.168, 6),
    "owl_detmold":     (51.936, 8.878, 5),
    "niederrhein_kleve": (51.788, 6.138, 5),
}
MAPILLARY_TILE = 0.01          # Kachelgröße in Grad; die API erlaubt nur kleine Gebiete pro Anfrage
MAPILLARY_FEATURE_LIMIT = 2000 # Antwortlimit pro Kachel; volle Kacheln werden geviertelt
MAPILLARY_IMAGES_PER_SIGN = 2  # Fotos pro erkanntem Zusatzschild (aus verschiedenen Blickwinkeln)
MAPILLARY_MAX_IMAGES = 600     # insgesamt, gleichmäßig auf die Regionen verteilt
MAPILLARY_DIR = "real_world_mapillary"

MAPILLARY_API = "https://graph.mapillary.com"
MAPILLARY_TOKEN_FILE = "mapillary_token.txt"     # steht in .gitignore


def mapillary_token():
    """Token aus der Umgebungsvariable oder aus mapillary_token.txt.

    Die Datei darf den nackten Token oder die ganze PowerShell-Zeile
    $env:MAPILLARY_TOKEN = "MLY|..." enthalten.
    """
    text = os.environ.get("MAPILLARY_TOKEN", "")
    if not text and os.path.exists(MAPILLARY_TOKEN_FILE):
        with open(MAPILLARY_TOKEN_FILE, encoding="utf-8-sig") as f:
            text = f.read()
    match = re.search(r"MLY\|[^\s\"']+", text)
    return match.group(0) if match else None

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "TrafficSignStudentProject/1.0 python-requests"


# ---------------------------------------------------------------- Gemeinsam

def get(url, params=None, headers=None, max_tries=5):
    """GET mit Wartezeit bei 429/5xx. Gibt die Response oder None zurück."""
    for attempt in range(max_tries):
        try:
            resp = SESSION.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as e:
            print(f"  Netzwerkfehler: {e}")
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = int(resp.headers.get("Retry-After", 10 * (attempt + 1)))
            print(f"  HTTP {resp.status_code}, warte {wait} s ...")
            time.sleep(wait)
            continue
        print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    return None


def get_json(url, params=None, headers=None):
    resp = get(url, params, headers)
    if resp is None:
        return None
    if "json" not in resp.headers.get("content-type", ""):
        print(f"  Keine JSON-Antwort: {resp.text[:150]}")
        return None
    return resp.json()


def save(url, out_path):
    """Lädt eine Datei, falls sie noch nicht existiert. True bei Erfolg."""
    if os.path.exists(out_path):
        return True
    resp = get(url)
    if resp is None:
        return False
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return True


def open_csv(path, header):
    """Hängt an eine CSV an; schreibt den Kopf nur bei einer neuen Datei."""
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(header)
    return f, writer


# ---------------------------------------------------------------- Mapillary

def region_bbox(lat, lon, radius_km):
    """Quadrat um einen Mittelpunkt als (West, Süd, Ost, Nord) in Grad."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def mapillary_tiles(bbox, step):
    west, south, east, north = bbox
    lat = south
    while lat < north:
        lon = west
        while lon < east:
            yield (round(lon, 5), round(lat, 5), round(min(lon + step, east), 5), round(min(lat + step, north), 5))
            lon += step
        lat += step


def mapillary_tile_features(headers, tile, depth=0):
    """Kartenobjekte einer Kachel. Ist die Antwort voll, wird die Kachel geviertelt,
    damit in dichten Innenstädten keine Schilder abgeschnitten werden."""
    data = get_json(f"{MAPILLARY_API}/map_features", {
        "fields": "id,object_value,geometry,images",
        "bbox": ",".join(map(str, tile)),
        "limit": MAPILLARY_FEATURE_LIMIT,
    }, headers)
    time.sleep(0.3)
    if data is None:
        return []
    feats = data.get("data", [])
    if len(feats) >= MAPILLARY_FEATURE_LIMIT and depth < 2:
        w, s, e, n = tile
        mx, my = (w + e) / 2, (s + n) / 2
        feats = []
        for sub in ((w, s, mx, my), (mx, s, e, my), (w, my, mx, n), (mx, my, e, n)):
            feats += mapillary_tile_features(headers, tuple(round(v, 6) for v in sub), depth + 1)
    return feats


def mapillary_sign_features(token, bbox):
    """Alle Kartenobjekte im Gebiet, deren Wert mit 'complementary--' beginnt (= Zusatzzeichen).
    Die Kacheln werden in zufälliger Reihenfolge abgefragt, damit bei einem
    Bildlimit nicht nur der Südwesten einer Region vorkommt."""
    headers = {"Authorization": f"OAuth {token}"}
    seen = set()
    tiles = list(mapillary_tiles(bbox, MAPILLARY_TILE))
    random.Random(42).shuffle(tiles)
    for i, tile in enumerate(tiles, 1):
        for feat in mapillary_tile_features(headers, tile):
            value = feat.get("object_value", "")
            if value.startswith("complementary--") and feat["id"] not in seen:
                seen.add(feat["id"])
                yield feat
        if i % 10 == 0 or i == len(tiles):
            print(f"  Kachel {i}/{len(tiles)}: {len(seen)} Zusatzzeichen bisher")


def mapillary_detection_boxes(token, image_id):
    """Erkannte Objekte eines Bildes als Liste von {wert, box}.

    Mapillary liefert die Umrisse als Base64-kodierte Vektorkachel (Ausdehnung 4096).
    Die Box ist normiert auf 0..1 (x0, y0, x1, y1). Ohne das Paket
    'mapbox-vector-tile' wird nur der Wert gespeichert.
    """
    headers = {"Authorization": f"OAuth {token}"}
    data = get_json(f"{MAPILLARY_API}/{image_id}/detections", {"fields": "value,geometry"}, headers)
    if data is None:
        return []
    try:
        import mapbox_vector_tile
    except ImportError:
        mapbox_vector_tile = None
    result = []
    for det in data.get("data", []):
        entry = {"wert": det.get("value")}
        if mapbox_vector_tile and det.get("geometry"):
            tile = mapbox_vector_tile.decode(base64.b64decode(det["geometry"]),
                                             default_options={"y_coord_down": True})
            points = []
            for layer in tile.values():
                extent = layer.get("extent", 4096)
                for feat in layer["features"]:
                    geom = feat["geometry"]
                    rings = geom["coordinates"] if geom["type"] == "Polygon" else \
                        [r for poly in geom["coordinates"] for r in poly]
                    points += [(x / extent, y / extent) for ring in rings for x, y in ring]
            if points:
                xs, ys = zip(*points)
                entry["box"] = [round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)]
        result.append(entry)
    return result


def scrape_mapillary_region(token, region, bbox, quota, writer, seen_images):
    """Lädt bis zu `quota` Fotos mit Zusatzzeichen aus einer Region."""
    headers = {"Authorization": f"OAuth {token}"}
    target_dir = os.path.join(MAPILLARY_DIR, region)
    os.makedirs(target_dir, exist_ok=True)
    count = 0
    for feat in mapillary_sign_features(token, bbox):
        for image_id in (feat.get("images", {}).get("data", []) or [])[:MAPILLARY_IMAGES_PER_SIGN]:
            if count >= quota:
                return count
            image_id = image_id["id"]
            if image_id in seen_images:            # dasselbe Foto kann mehrere Zusatzschilder zeigen
                continue
            seen_images.add(image_id)
            if mapillary_download(token, image_id, target_dir, writer, region, feat["object_value"]):
                count += 1
    return count


def mapillary_download(token, image_id, target_dir, writer, region, value):
    """Lädt ein Foto samt Schild-Umrissen und schreibt eine Zeile in labels.csv. True bei Erfolg."""
    out_path = os.path.join(target_dir, f"{image_id}.jpg")
    if os.path.exists(out_path):
        return True
    headers = {"Authorization": f"OAuth {token}"}
    info = get_json(f"{MAPILLARY_API}/{image_id}",
                    {"fields": "id,thumb_2048_url,captured_at,creator,geometry"}, headers)
    if not info or not info.get("thumb_2048_url"):
        return False
    if not save(info["thumb_2048_url"], out_path):
        return False
    lon, lat = info.get("geometry", {}).get("coordinates", [None, None])
    captured = info.get("captured_at")
    captured = time.strftime("%Y-%m-%d", time.gmtime(captured / 1000)) if captured else ""
    writer.writerow([out_path, region, image_id, value,
                     json.dumps(mapillary_detection_boxes(token, image_id), ensure_ascii=False),
                     lon, lat, captured, info.get("creator", {}).get("username", ""),
                     "CC BY-SA 4.0", f"https://www.mapillary.com/app/?pKey={image_id}"])
    time.sleep(0.3)
    return True


def scrape_mapillary(max_images, regions):
    token = mapillary_token()
    if not token:
        print(f"Mapillary übersprungen: kein Token (Umgebungsvariable MAPILLARY_TOKEN "
              f"oder Datei {MAPILLARY_TOKEN_FILE}, siehe Anleitung oben).")
        return
    os.makedirs(MAPILLARY_DIR, exist_ok=True)
    f, writer = open_csv(os.path.join(MAPILLARY_DIR, "labels.csv"), MAPILLARY_HEADER)
    # Gleichmäßig verteilen; was eine Region nicht ausschöpft, geht an die nächsten
    remaining, seen_images, total = max_images, set(), 0
    with f:
        for i, region in enumerate(regions):
            quota = math.ceil(remaining / (len(regions) - i))
            lat, lon, radius = MAPILLARY_REGIONS[region]
            print(f"Mapillary '{region}' ({radius} km um {lat}, {lon}), Ziel {quota} Fotos")
            got = scrape_mapillary_region(token, region, region_bbox(lat, lon, radius), quota,
                                          writer, seen_images)
            f.flush()
            print(f"Mapillary '{region}': {got} Fotos.")
            remaining -= got
            total += got
    print(f"Mapillary gesamt: {total} Fotos aus {len(regions)} Regionen.")


# ---------------------------------------------------------------- Mapillary: Zielgruppen in ganz Deutschland

MAPILLARY_HEADER = ["datei", "region", "bild_id", "zusatzzeichen", "alle_erkennungen", "laenge",
                    "breite", "aufgenommen", "urheber", "lizenz", "quelle"]

# Mapillary-Werte von Zusatzzeichen, die eine Zielgruppe nennen (gilt für / ausgenommen).
# Gefunden in Stichproben aus 15 deutschen Städten am 18.11.2026.
ZIELGRUPPEN_WERTE = [
    "complementary--except-bicycles--g1",          # Radfahrer frei
    "complementary--trucks--g1",                   # LKW
    "complementary--trucks-buses-trailers--g1",    # LKW, Bus, PKW mit Anhänger
    "complementary--trucks-and-trailers--g1",
    "complementary--buses--g1",                    # Bus
    "complementary--except-buses--g1",
    "complementary--motorcycles--g2",              # Krafträder
    "complementary--vehicles--g1",                 # PKW (Sinnbild)
    "complementary--vehicles--g2",
    "complementary--except-vehicles--g1",          # PKW frei
    "complementary--except-trailers--g1",
    "complementary--except-tractors--g1",          # Landwirtschaftlicher Verkehr frei
    "complementary--except-trams--g1",
    "complementary--disabled-persons--g1",         # Schwerbehinderte
]
# Nicht mehr gesucht: "dangerous-or-pollutant-good" ist auf deutschen Fotos das Hinweiszeichen 354
# "Wasserschutzgebiet", kein Zusatzzeichen (geprüft an 8 Fotos, 20.11.2026).
ZIELGRUPPEN_MAX_PER_WERT = 150     # Fotos je Wert
ZIELGRUPPEN_KACHEL = 0.05          # Grad; größere Kacheln sind bei Mapillary sehr langsam
ZIELGRUPPEN_RADIUS_KM = 6          # Suchgebiet um jede Stadt
ZIELGRUPPEN_TESTREGIONEN = ["aachen", "eifel_monschau", "koeln"]   # nie ins Training

# Städte über ganz Deutschland verteilt (Mittelpunkte gerundet, aus dem Gedächtnis).
# Auf dem Land sind fast alle Kacheln leer; in Städten stehen die meisten Zusatzzeichen.
ZIELGRUPPEN_STAEDTE = {
    "berlin": (52.520, 13.405), "hamburg": (53.550, 9.990), "muenchen": (48.137, 11.575),
    "frankfurt": (50.110, 8.680), "stuttgart": (48.776, 9.180), "duesseldorf": (51.227, 6.774),
    "leipzig": (51.340, 12.370), "dortmund": (51.514, 7.468), "essen": (51.456, 7.012),
    "bremen": (53.080, 8.800), "dresden": (51.050, 13.740), "hannover": (52.375, 9.730),
    "nuernberg": (49.450, 11.080), "duisburg": (51.434, 6.762), "bochum": (51.482, 7.216),
    "wuppertal": (51.256, 7.150), "bielefeld": (52.021, 8.532), "bonn": (50.737, 7.098),
    "muenster": (51.962, 7.626), "mannheim": (49.488, 8.466), "karlsruhe": (49.007, 8.400),
    "augsburg": (48.370, 10.898), "wiesbaden": (50.080, 8.240), "moenchengladbach": (51.185, 6.442),
    "gelsenkirchen": (51.517, 7.085), "kiel": (54.320, 10.135), "chemnitz": (50.830, 12.920),
    "halle": (51.480, 11.970), "magdeburg": (52.130, 11.630), "freiburg": (47.999, 7.842),
    "krefeld": (51.339, 6.586), "mainz": (49.990, 8.247), "luebeck": (53.866, 10.686),
    "erfurt": (50.978, 11.029), "rostock": (54.090, 12.140), "kassel": (51.312, 9.480),
    "hagen": (51.360, 7.470), "potsdam": (52.390, 13.065), "saarbruecken": (49.234, 6.996),
    "hamm": (51.680, 7.820), "oldenburg": (53.140, 8.214), "osnabrueck": (52.280, 8.047),
    "heidelberg": (49.400, 8.690), "darmstadt": (49.870, 8.650), "regensburg": (49.013, 12.100),
    "ingolstadt": (48.766, 11.425), "wuerzburg": (49.790, 9.950), "wolfsburg": (52.420, 10.787),
    "ulm": (48.400, 9.990), "heilbronn": (49.140, 9.220), "braunschweig": (52.270, 10.520),
    "goettingen": (51.540, 9.930), "jena": (50.930, 11.590), "trier": (49.750, 6.640),
    "koblenz": (50.360, 7.590), "paderborn": (51.719, 8.754), "siegen": (50.875, 8.024),
    "bremerhaven": (53.540, 8.580), "konstanz": (47.660, 9.170), "passau": (48.570, 13.430),
    "flensburg": (54.780, 9.440), "cottbus": (51.760, 14.330), "schwerin": (53.630, 11.410),
    "solingen": (51.170, 7.080), "leverkusen": (51.030, 6.980), "oberhausen": (51.470, 6.850),
}


def zielgruppen_tile(headers, tile, values, depth=0):
    """Kartenobjekte der gesuchten Werte in einer Kachel. Bricht die API wegen
    zu vieler Daten ab, wird die Kachel geviertelt."""
    # Eigene Anfrage ohne lange Wartezeiten: Zeitüberschreitung oder Serverfehler -> sofort vierteln
    data = None
    try:
        resp = SESSION.get(f"{MAPILLARY_API}/map_features", params={
            "fields": "id,object_value,geometry,images",
            "bbox": ",".join(f"{v:.5f}" for v in tile),
            "object_values": ",".join(values),
            "limit": 500,
        }, headers=headers, timeout=20)
        if resp.status_code == 429:
            time.sleep(int(resp.headers.get("Retry-After", 30)))
            return zielgruppen_tile(headers, tile, values, depth)
        if resp.status_code == 200 and "json" in resp.headers.get("content-type", ""):
            data = resp.json()
    except requests.RequestException:
        pass
    time.sleep(0.3)
    if data is not None and len(data.get("data", [])) < 500:
        return data["data"]
    if depth >= 2:
        return data.get("data", []) if data else []
    w, s, e, n = tile
    mx, my = (w + e) / 2, (s + n) / 2
    feats = []
    for sub in ((w, s, mx, my), (mx, s, e, my), (w, my, mx, n), (mx, my, e, n)):
        feats += zielgruppen_tile(headers, sub, values, depth + 1)
    return feats


def scrape_zielgruppen(max_per_value, values, max_tiles=None, test=False):
    """Sucht Zusatzzeichen mit Zielgruppe in Städten ganz Deutschlands, bis jeder Wert sein Kontingent hat.

    Die Kacheln kommen reihum aus allen Städten, damit die Fotos über das Land verteilt sind.
    Schilder in den Testregionen werden übersprungen, Fotos aus früheren Läufen nicht doppelt geladen.
    Mit test=True umgekehrt: nur die Testregionen (Aachen, Eifel, Köln), Ablage in zielgruppen_test/.
    """
    token = mapillary_token()
    if not token:
        print(f"Mapillary übersprungen: kein Token (Datei {MAPILLARY_TOKEN_FILE}).")
        return
    headers = {"Authorization": f"OAuth {token}"}
    test_boxes = [region_bbox(*MAPILLARY_REGIONS[r]) for r in ZIELGRUPPEN_TESTREGIONEN]

    def in_test(lon, lat):
        return any(w <= lon <= e and s <= lat <= n for w, s, e, n in test_boxes)

    os.makedirs(MAPILLARY_DIR, exist_ok=True)
    labels = os.path.join(MAPILLARY_DIR, "labels.csv")
    seen_images = set()
    if os.path.exists(labels):
        with open(labels, encoding="utf-8") as lf:
            for row in csv.reader(lf):
                if len(row) >= 10:
                    seen_images.add(row[2] if len(row) == 11 else row[1])
    base = os.path.join(MAPILLARY_DIR, "zielgruppen_test" if test else "zielgruppen")
    count = {}
    for v in values:
        d = os.path.join(base, v.split("--")[1] + "_" + v.split("--")[2])
        os.makedirs(d, exist_ok=True)
        count[v] = len([x for x in os.listdir(d) if x.endswith(".jpg")])

    # Kacheln reihum aus allen Städten, damit früh viele Städte vertreten sind
    city_tiles = {}
    if test:
        for c in ZIELGRUPPEN_TESTREGIONEN:
            city_tiles[c] = list(mapillary_tiles(region_bbox(*MAPILLARY_REGIONS[c]), ZIELGRUPPEN_KACHEL))
    else:
        for c, (lat, lon) in ZIELGRUPPEN_STAEDTE.items():
            ts = [t for t in mapillary_tiles(region_bbox(lat, lon, ZIELGRUPPEN_RADIUS_KM), ZIELGRUPPEN_KACHEL)
                  if not in_test((t[0] + t[2]) / 2, (t[1] + t[3]) / 2)]
            random.Random(c).shuffle(ts)
            city_tiles[c] = ts
    order = list(city_tiles)
    random.Random(7).shuffle(order)
    tiles = [(c, city_tiles[c][k]) for k in range(max(map(len, city_tiles.values())))
             for c in order if k < len(city_tiles[c])]
    if max_tiles:
        tiles = tiles[:max_tiles]
    # keine Stadt füllt eine Klasse allein (im Testmodus zählt jedes Schild)
    per_city = max_per_value if test else max(5, max_per_value // 10)
    city_count = {}
    print(f"Zielgruppen: {len(tiles)} Kacheln in {len(order)} Städten, {len(values)} Werte, "
          f"Ziel {max_per_value} Fotos je Wert, höchstens {per_city} je Stadt und Wert")

    f, writer = open_csv(labels, MAPILLARY_HEADER)
    with f:
        for i, (city, tile) in enumerate(tiles, 1):
            open_values = [v for v in values if count[v] < max_per_value]
            if not open_values:
                break
            for feat in zielgruppen_tile(headers, tile, open_values):
                v = feat.get("object_value")
                if v not in open_values or count[v] >= max_per_value:
                    continue
                if city_count.get((city, v), 0) >= per_city:
                    continue
                lon, lat = feat.get("geometry", {}).get("coordinates", [0, 0])
                if in_test(lon, lat) != test:
                    continue
                d = os.path.join(base, v.split("--")[1] + "_" + v.split("--")[2])
                for img in (feat.get("images", {}).get("data", []) or [])[:MAPILLARY_IMAGES_PER_SIGN]:
                    if count[v] >= max_per_value or img["id"] in seen_images:
                        continue
                    seen_images.add(img["id"])
                    if city_count.get((city, v), 0) >= per_city:
                        break
                    if mapillary_download(token, img["id"], d, writer, city if test else f"zg_{city}", v):
                        count[v] += 1
                        city_count[(city, v)] = city_count.get((city, v), 0) + 1
            f.flush()
            if i % 50 == 0 or i == len(tiles):
                stand = ", ".join(f"{v.split('--', 1)[1]} {c}" for v, c in count.items())
                print(f"  Kachel {i}/{len(tiles)}: {stand}", flush=True)
    print("Zielgruppen fertig: " + ", ".join(f"{v.split('--', 1)[1]} {c}" for v, c in count.items()))


# ---------------------------------------------------------------- Start

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Echte Fotos von Schildkombinationen laden")
    parser.add_argument("--quelle", choices=["mapillary", "zielgruppen"], default="mapillary",
                        help="'mapillary' = 21 Regionen in NRW; 'zielgruppen' = Zusatzzeichen mit Zielgruppe in ganz Deutschland")
    parser.add_argument("--werte", default=None,
                        help="zielgruppen: kommagetrennte Mapillary-Werte statt der Standardliste")
    parser.add_argument("--testregionen", action="store_true",
                        help="zielgruppen: nur in den Testregionen Aachen, Eifel, Köln suchen (Testsatz)")
    parser.add_argument("--max-kacheln", type=int, default=None,
                        help="zielgruppen: höchstens so viele Kacheln abfragen (zum Testen)")
    parser.add_argument("--max", type=int, default=None,
                        help=f"mapillary: Fotos insgesamt (Standard {MAPILLARY_MAX_IMAGES}); "
                             f"zielgruppen: Fotos je Wert (Standard {ZIELGRUPPEN_MAX_PER_WERT})")
    parser.add_argument("--regionen", default="alle",
                        help="Mapillary: kommagetrennt, z. B. 'aachen,koeln,eifel_monschau'; "
                             f"Standard alle {len(MAPILLARY_REGIONS)} Regionen in NRW. "
                             f"Verfügbar: {', '.join(MAPILLARY_REGIONS)}")
    args = parser.parse_args()

    if args.regionen == "alle":
        regions = list(MAPILLARY_REGIONS)
    else:
        regions = [r.strip() for r in args.regionen.split(",") if r.strip()]
        unknown = [r for r in regions if r not in MAPILLARY_REGIONS]
        if unknown:
            parser.error(f"Unbekannte Region(en): {', '.join(unknown)}")

    if args.quelle == "mapillary":
        scrape_mapillary(args.max or MAPILLARY_MAX_IMAGES, regions)
    if args.quelle == "zielgruppen":
        values = [v.strip() for v in args.werte.split(",")] if args.werte else ZIELGRUPPEN_WERTE
        scrape_zielgruppen(args.max or ZIELGRUPPEN_MAX_PER_WERT, values, args.max_kacheln, args.testregionen)
    print("\nFertig.")
