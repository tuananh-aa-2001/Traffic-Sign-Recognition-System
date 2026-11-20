"""
Schildkombinationen (Hauptschild + Zusatzzeichen darunter) aus Gesamtfotos ausschneiden.

Zwei Modi:

  mapillary   Automatisch. Nutzt die Objekt-Erkennungen von Mapillary (Umriss jedes Schilds).
              Fehlen die Boxen in real_world_mapillary/labels.csv, werden sie mit dem
              Mapillary-Token nachgeladen und in real_world_mapillary/boxen.json zwischengespeichert.
              Braucht:  pip install mapbox-vector-tile   und den Token in mapillary_token.txt
              Zusatzzeichen, über denen Mapillary kein Hauptschild erkannt hat, werden "gerettet":
              das Hauptschild wird per Farbe (rot/blau/gelb) direkt darüber gesucht
              -> crops/mapillary_gerettet, Hauptschild "unbekannt". Mit --geschaetzt auch ohne
              Farbtreffer (geschätztes Fenster, viele Fehltreffer), mit --no-retten gar nicht.

  markieren   Halbautomatisch für Fotos ohne Boxen (Wikimedia, eigene Fotos).
              Rote/blaue Schilder werden per Farbe vorgeschlagen und nach unten um ein weißes
              Zusatzschild verlängert. Im Fenster: Vorschläge an-/abwählen oder selbst Boxen ziehen.
              Mit --auto ohne Fenster: alle Vorschläge werden gespeichert (später durchsehen!).

Ergebnis: crops/<quelle>/*.jpg und crops/labels.csv mit Originaldatei, Box in Pixeln,
Hauptschild, Zusatzzeichen ("none" = keins) und Methode (mapillary / auto / manuell).
Die Box wird gespeichert, damit man später mit anderem Rand neu ausschneiden kann.

Beispiele:
  python crop_signs.py mapillary --kontrolle 30
  python crop_signs.py markieren real_world_wikimedia
  python crop_signs.py markieren eigene_fotos --auto
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

OUT_DIR = "crops"
LABELS = os.path.join(OUT_DIR, "labels.csv")
HEADER = ["crop", "original", "x0", "y0", "x1", "y1", "hauptschild", "zusatzzeichen", "methode"]
PAD = 0.10                       # Rand um die Kombination, wie bei GTSRB (ca. 10 %)
SIGN_PREFIXES = ("regulatory--", "warning--", "information--", "complementary--")
IMG_EXT = (".jpg", ".jpeg", ".png")
NOT_ZUSATZ = ("chevron",)        # Leitbaken führt Mapillary als "complementary", sind aber keine Zusatzzeichen


# ---------------------------------------------------------------- Allgemein

def load_done():
    """Schon verarbeitete Originale (zum Fortsetzen nach Abbruch)."""
    if not os.path.exists(LABELS):
        return set()
    with open(LABELS, encoding="utf-8") as f:
        return {row["original"] for row in csv.DictReader(f)}


def open_labels():
    os.makedirs(OUT_DIR, exist_ok=True)
    new = not os.path.exists(LABELS)
    f = open(LABELS, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new:
        writer.writerow(HEADER)
    return f, writer


def imread(path):
    """cv2.imread kann unter Windows keine Umlaute im Pfad lesen."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path, img):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        buf.tofile(path)
    return ok


def padded(box, w_img, h_img, pad=PAD):
    x0, y0, x1, y1 = box
    p = pad * max(x1 - x0, y1 - y0)
    return (int(max(0, x0 - p)), int(max(0, y0 - p)),
            int(min(w_img, x1 + p)), int(min(h_img, y1 + p)))


def union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def save_crop(writer, img, original, box, sub, name, haupt, zusatz, methode):
    h_img, w_img = img.shape[:2]
    x0, y0, x1, y1 = padded(box, w_img, h_img)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return False
    out = os.path.join(OUT_DIR, sub, name)
    if not imwrite(out, img[y0:y1, x0:x1]):
        return False
    writer.writerow([out, original, x0, y0, x1, y1, haupt, zusatz, methode])
    return True


# ---------------------------------------------------------------- Modus mapillary

def read_mapillary_labels(path):
    """labels.csv enthält ältere Zeilen ohne Spalte 'region' (10 statt 11 Felder)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for r in reader:
            if len(r) == 11:
                datei, bild_id, erk = r[0], r[2], r[4]
            elif len(r) == 10:
                datei, bild_id, erk = r[0], r[1], r[3]
            else:
                continue
            rows.append((datei.replace("\\", os.sep), bild_id, json.loads(erk or "[]")))
    return rows


def group_combinations(signs, w_img, h_img, min_px):
    """Ordnet jedem Hauptschild die Zusatzzeichen direkt darunter zu.

    signs: Liste (wert, (x0, y0, x1, y1)) in Pixeln. Zusatzzeichen gehören zu einem
    Hauptschild, wenn sie sich horizontal mit ihm überlappen und lückenlos darunter
    hängen (Abstand höchstens 0,6 Schildhöhen, auch mehrere untereinander).
    """
    main = [(v, b) for v, b in signs if not v.startswith("complementary--")]
    extra = sorted([(v, b) for v, b in signs if v.startswith("complementary--")], key=lambda s: s[1][1])
    used, combos = set(), []
    for value, mb in main:
        mw, mh = mb[2] - mb[0], mb[3] - mb[1]
        if mh < min_px:
            continue
        bottom, attached = mb[3], []
        for i, (zv, zb) in enumerate(extra):
            if i in used:
                continue
            overlap = min(mb[2], zb[2]) - max(mb[0], zb[0])
            if overlap < 0.5 * min(mw, zb[2] - zb[0]):
                continue
            if zb[1] < mb[1] + 0.5 * mh or zb[1] - bottom > 0.6 * mh:
                continue
            attached.append(i)
            bottom = zb[3]
        used.update(attached)
        combos.append((value, [extra[i] for i in attached], union([mb] + [extra[i][1] for i in attached])))
    return combos, [extra[i] for i in range(len(extra)) if i not in used]


def chain_orphans(orphans):
    """Übereinander hängende verwaiste Zusatzzeichen zu Stapeln zusammenfassen."""
    stacks = []
    for zv, zb in sorted(orphans, key=lambda s: s[1][1]):
        for st in stacks:
            top = union([b for _, b in st])
            w = zb[2] - zb[0]
            if min(top[2], zb[2]) - max(top[0], zb[0]) > 0.5 * w and 0 <= zb[1] - top[3] < 0.5 * w:
                st.append((zv, zb))
                break
        else:
            stacks.append([(zv, zb)])
    return stacks


def sign_mask(hsv, yellow=False):
    red = cv2.bitwise_or(cv2.inRange(hsv, (0, 90, 60), (10, 255, 255)),
                         cv2.inRange(hsv, (168, 90, 60), (180, 255, 255)))
    mask = cv2.bitwise_or(red, cv2.inRange(hsv, (100, 120, 50), (128, 255, 255)))
    if yellow:                                   # Vorfahrtstraße
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, (18, 120, 120), (35, 255, 255)))
    return mask


def rescue_main(hsv, signs, stack):
    """Sucht über einem verwaisten Zusatzzeichen-Stapel ein Hauptschild per Farbe.

    Rückgabe (box, methode) oder None. 'None' bei: Hauptschild liegt darunter
    (z. B. Straßennummer über dem Vorfahrtschild), über dem Zusatzzeichen ist ein
    anderes erkanntes Schild im Weg, oder das Bild endet oben.
    """
    h_img, w_img = hsv.shape[:2]
    sb = union([b for _, b in stack])
    x0, y0, x1, y1 = sb
    w = x1 - x0
    # Gelbe Schilder ohne Hauptschild sind Wegweiser/Umleitungen, keine Zusatzzeichen
    zb = stack[0][1]
    patch = hsv[int(zb[1]):int(zb[3]), int(zb[0]):int(zb[2])]
    if patch.size and (cv2.inRange(patch, (15, 70, 80), (35, 255, 255)) > 0).mean() > 0.3:
        return None
    search = (max(0, x0 - 0.2 * w), y0 - 1.4 * w, min(w_img, x1 + 0.2 * w), y0 + 0.05 * w)
    if search[1] < 0:
        return None
    for v, b in signs:
        if v.startswith("complementary--"):
            continue
        overlap_x = min(b[2], x1) - max(b[0], x0) > 0
        if overlap_x and 0 <= b[1] - y1 < 0.6 * w:                       # Hauptschild darunter
            return None
        if overlap_x and b[3] > search[1] and b[1] < y0:                 # anderes Schild im Weg
            return None
    rx0, ry0, rx1, ry1 = map(int, search)
    mask = cv2.morphologyEx(sign_mask(hsv[ry0:ry1, rx0:rx1], yellow=True),
                            cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        if not 0.45 * w <= cw <= 1.3 * w or not 0.6 <= cw / ch <= 1.6:
            continue
        bottom = ry0 + cy + ch
        if abs(y0 - bottom) > 0.4 * w:
            continue
        if best is None or cw * ch > best[2] * best[3]:
            best = (cx, cy, cw, ch)
    if best:
        cx, cy, cw, ch = best
        main = (rx0 + cx, ry0 + cy, rx0 + cx + cw, ry0 + cy + ch)
        return union([main, sb]), "mapillary+farbe"
    # nichts gefunden: Fenster in Breite des Zusatzzeichens schätzen
    return union([(x0, y0 - 1.05 * w, x1, y0), sb]), "mapillary_geschaetzt"


def fetch_boxes(bild_id, token):
    from scrape_mapillary import mapillary_detection_boxes
    return mapillary_detection_boxes(token, bild_id)


def run_mapillary(args):
    src = os.path.join("real_world_mapillary", "labels.csv")
    if not os.path.exists(src):
        sys.exit(f"{src} fehlt. Zuerst: python scrape_mapillary.py --quelle mapillary")
    cache_path = os.path.join("real_world_mapillary", "boxen.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    rows = read_mapillary_labels(src)
    need = [b for _, b, erk in rows if b not in cache and not any("box" in d for d in erk)]
    from scrape_mapillary import mapillary_token
    token = mapillary_token()
    if need:
        try:
            import mapbox_vector_tile  # noqa: F401
        except ImportError:
            sys.exit("Für die Schild-Umrisse fehlt ein Paket:  pip install mapbox-vector-tile")
        if not token:
            sys.exit(f"Für {len(need)} Fotos fehlen die Boxen. Token fehlt (mapillary_token.txt).")
        print(f"Lade Schild-Umrisse für {len(need)} Fotos nach ...")

    done = load_done()
    f, writer = open_labels()
    n_crops = n_none = n_orphan = n_kontrolle = n_kontrolle_r = 0
    n_rescued = {}
    with f:
        for k, (datei, bild_id, erk) in enumerate(rows, 1):
            if datei in done or not os.path.exists(datei):
                continue
            if not any("box" in d for d in erk):
                if bild_id not in cache:
                    cache[bild_id] = fetch_boxes(bild_id, token)
                    if k % 25 == 0:
                        with open(cache_path, "w", encoding="utf-8") as cf:
                            json.dump(cache, cf)
                        print(f"  {k}/{len(rows)} Fotos")
                erk = cache[bild_id]
            img = imread(datei)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]
            signs = [(d["wert"], (d["box"][0] * w_img, d["box"][1] * h_img,
                                  d["box"][2] * w_img, d["box"][3] * h_img))
                     for d in erk if "box" in d and d["wert"].startswith(SIGN_PREFIXES)
                     and not any(n in d["wert"] for n in NOT_ZUSATZ)]
            combos, orphans = group_combinations(signs, w_img, h_img, args.min_px)
            stem = os.path.splitext(os.path.basename(datei))[0]
            rescued = []
            if args.retten and orphans:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                for stack in chain_orphans(orphans):
                    if (stack[0][1][3] - stack[0][1][1]) < args.min_px / 2:
                        n_orphan += len(stack)
                        continue
                    found = rescue_main(hsv, signs, stack)
                    if found is None:
                        n_orphan += len(stack)
                        continue
                    box, methode = found
                    if methode == "mapillary_geschaetzt" and not args.geschaetzt:
                        n_orphan += len(stack)
                        continue
                    rescued.append((box, methode))
                    z = "|".join(v for v, _ in stack)
                    if save_crop(writer, img, datei, box, "mapillary_gerettet",
                                 f"{stem}_r{len(rescued)}.jpg", "unbekannt", z, methode):
                        n_rescued[methode] = n_rescued.get(methode, 0) + 1
            else:
                n_orphan += len(orphans)
            if rescued and n_kontrolle_r < args.kontrolle:
                draw_control(img, signs, [("", [], b) for b, _ in rescued],
                             os.path.join(OUT_DIR, "_kontrolle_gerettet", f"{stem}.jpg"))
                n_kontrolle_r += 1
            for j, (haupt, zusatz, box) in enumerate(combos):
                if not zusatz and not args.ohne_zusatz:
                    continue
                z = "|".join(v for v, _ in zusatz) or "none"
                if save_crop(writer, img, datei, box, "mapillary", f"{stem}_{j}.jpg", haupt, z, "mapillary"):
                    n_crops += 1
                    n_none += not zusatz
            if n_kontrolle < args.kontrolle and combos:
                draw_control(img, signs, combos, os.path.join(OUT_DIR, "_kontrolle", f"{stem}.jpg"))
                n_kontrolle += 1
    with open(cache_path, "w", encoding="utf-8") as cf:
        json.dump(cache, cf)
    print(f"{n_crops} Ausschnitte gespeichert ({n_crops - n_none} mit Zusatzzeichen, {n_none} ohne).")
    if n_rescued:
        print("Gerettet (Hauptschild 'unbekannt', bitte durchsehen): "
              + ", ".join(f"{v} {k}" for k, v in n_rescued.items())
              + f"  -> {os.path.join(OUT_DIR, 'mapillary_gerettet')}")
    print(f"{n_orphan} Zusatzzeichen ohne Hauptschild darüber übersprungen.")
    if n_kontrolle:
        print(f"Kontrollbilder: {os.path.join(OUT_DIR, '_kontrolle')}  (grün Haupt, orange Zusatz, blau Ausschnitt)")


def draw_control(img, signs, combos, path):
    vis = img.copy()
    h_img, w_img = img.shape[:2]
    for v, b in signs:
        color = (0, 165, 255) if v.startswith("complementary--") else (0, 200, 0)
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
    for _, _, box in combos:
        x0, y0, x1, y1 = padded(box, w_img, h_img)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 80, 0), 1)
    imwrite(path, vis)


# ---------------------------------------------------------------- Modus markieren

def propose(img):
    """Rote/blaue Schildflächen finden und um ein weißes Zusatzschild darunter verlängern."""
    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(cv2.inRange(hsv, (0, 90, 60), (10, 255, 255)),
                         cv2.inRange(hsv, (168, 90, 60), (180, 255, 255)))
    blue = cv2.inRange(hsv, (100, 120, 50), (128, 255, 255))
    mask = cv2.morphologyEx(cv2.bitwise_or(red, blue), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_side = 0.03 * min(w_img, h_img)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if min(w, h) < min_side or not 0.7 <= w / h <= 1.4:
            continue
        hull = cv2.contourArea(cv2.convexHull(cnt))
        if hull < 0.5 * w * h or cv2.contourArea(cnt) < 0.8 * hull:   # Schilder sind kompakt und konvex
            continue
        boxes.append(extend_down(hsv, (x, y, x + w, y + h)))
    boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    return boxes[:6]


def extend_down(hsv, box):
    """Verlängert die Box nach unten um eine helle, ungesättigte Fläche (Zusatzschild).

    Zeilen mit über 45 % Weiß zählen als hell; kurze Unterbrechungen (Rahmen, Schrift,
    Lücke am Pfosten) bis 0,15 Schildhöhen werden überbrückt.
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    xa, xb = x0 + int(0.15 * w), x1 - int(0.15 * w)
    limit = min(hsv.shape[0], y1 + int(1.2 * h))
    if xb <= xa or limit <= y1:
        return box
    strip = hsv[y1:limit, xa:xb]
    bright = ((strip[..., 1] < 60) & (strip[..., 2] > 140)).mean(axis=1) > 0.45
    max_gap = max(2, int(0.15 * h))
    start = next((i for i, v in enumerate(bright[: int(0.25 * h) + 1]) if v), None)
    if start is None:
        return box
    end, gap = start, 0
    for i in range(start, len(bright)):
        if bright[i]:
            end, gap = i + 1, 0
        else:
            gap += 1
            if gap > max_gap:
                break
    if end - start < 0.2 * h:
        return box
    return (x0, y0, x1, y1 + end)


def list_images(folder):
    for root, _, files in os.walk(folder):
        if os.path.basename(root).startswith("_"):
            continue
        for name in sorted(files):
            if name.lower().endswith(IMG_EXT):
                yield os.path.join(root, name)


def run_markieren(args):
    if not os.path.isdir(args.ordner):
        sys.exit(f"Ordner '{args.ordner}' nicht gefunden.")
    done = load_done()
    images = [p for p in list_images(args.ordner) if p not in done]
    print(f"{len(images)} Fotos zu bearbeiten ({len(done)} schon erledigt).")
    if not args.auto:
        print("Tasten: 1-9 Vorschlag an/aus | m Box selbst ziehen | Enter/Leertaste speichern + weiter"
              " | n weiter ohne Speichern | q beenden")
    sub = os.path.basename(os.path.normpath(args.ordner))
    f, writer = open_labels()
    n_crops = 0
    with f:
        for k, path in enumerate(images, 1):
            img = imread(path)
            if img is None:
                continue
            hint = os.path.basename(os.path.dirname(path))       # z. B. Wikimedia-Kategorie
            proposals = propose(img)
            if args.auto:
                chosen = [(b, "auto") for b in proposals]
            else:
                chosen = review(img, proposals, f"{k}/{len(images)}  {os.path.basename(path)}")
                if chosen is None:
                    break
            stem = os.path.splitext(os.path.basename(path))[0]
            for j, (box, methode) in enumerate(chosen):
                n_crops += save_crop(writer, img, path, box, os.path.join(sub, hint),
                                     f"{stem}_{j}.jpg", "", f"? ({hint})", methode)
            if not chosen:      # als erledigt vermerken, damit es nicht erneut erscheint
                writer.writerow(["", path, "", "", "", "", "", "", "kein_schild"])
            f.flush()
    if not args.auto:
        cv2.destroyAllWindows()
    print(f"{n_crops} Ausschnitte gespeichert in {os.path.join(OUT_DIR, sub)}. "
          "Haupt- und Zusatzzeichen-Label in crops/labels.csv noch eintragen bzw. prüfen.")


def review(img, proposals, title, max_w=1400, max_h=850):
    """Zeigt Vorschläge; gibt Liste (box, methode) zurück, None bei 'q'."""
    scale = min(1.0, max_w / img.shape[1], max_h / img.shape[0])
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()
    active = [True] * len(proposals)
    manual = []
    win = "Schilder markieren"
    while True:
        vis = small.copy()
        for i, b in enumerate(proposals):
            color = (0, 200, 0) if active[i] else (120, 120, 120)
            p0 = (int(b[0] * scale), int(b[1] * scale))
            cv2.rectangle(vis, p0, (int(b[2] * scale), int(b[3] * scale)), color, 2)
            cv2.putText(vis, str(i + 1), (p0[0], max(15, p0[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for b in manual:
            cv2.rectangle(vis, (int(b[0] * scale), int(b[1] * scale)),
                          (int(b[2] * scale), int(b[3] * scale)), (255, 80, 0), 2)
        cv2.putText(vis, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            return None
        if key == ord("n"):
            return []
        if key in (13, 32):
            return [(b, "auto") for b, a in zip(proposals, active) if a] + [(b, "manuell") for b in manual]
        if ord("1") <= key <= ord("9") and key - ord("1") < len(proposals):
            active[key - ord("1")] ^= True
        if key == ord("m"):
            # Mehrere Boxen nacheinander ziehen, jede mit Enter bestätigen, Esc beendet
            for x, y, w, h in cv2.selectROIs(win, vis, showCrosshair=False):
                manual.append((x / scale, y / scale, (x + w) / scale, (y + h) / scale))


# ---------------------------------------------------------------- Start

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schildkombinationen aus Gesamtfotos ausschneiden")
    sub = parser.add_subparsers(dest="modus", required=True)
    p_map = sub.add_parser("mapillary", help="automatisch mit Mapillary-Erkennungen")
    p_map.add_argument("--min-px", type=int, default=20, help="Mindesthöhe des Hauptschilds in Pixeln")
    p_map.add_argument("--ohne-zusatz", action=argparse.BooleanOptionalAction, default=True,
                       help="Hauptschilder ohne Zusatzzeichen ebenfalls speichern (Klasse 'none')")
    p_map.add_argument("--retten", action=argparse.BooleanOptionalAction, default=True,
                       help="Zusatzzeichen ohne erkanntes Hauptschild: Hauptschild darüber per Farbe suchen")
    p_map.add_argument("--geschaetzt", action="store_true",
                       help="beim Retten auch ohne Farbtreffer ein geschätztes Fenster speichern")
    p_map.add_argument("--kontrolle", type=int, default=30, help="Anzahl Kontrollbilder mit Boxen")
    p_mark = sub.add_parser("markieren", help="halbautomatisch für Fotos ohne Boxen")
    p_mark.add_argument("ordner")
    p_mark.add_argument("--auto", action="store_true", help="ohne Fenster alle Vorschläge speichern")
    args = parser.parse_args()
    run_mapillary(args) if args.modus == "mapillary" else run_markieren(args)
