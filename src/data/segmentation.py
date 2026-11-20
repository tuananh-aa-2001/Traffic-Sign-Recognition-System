"""
Synset-Segmentierung -> Klassenkarte für Architektur B.

Die Synset-Segmentierung ist keine reine Klassenkarte: das Hauptschild ist als saubere Zeichnung
enthalten, der Hintergrund ist hellcyan, andere Schilder sind kräftig türkis, der Mast dunkel weinrot.
Das Hauptschild kommt deshalb aus der eigenen Maske, der Rest über den Farbton (HSV).

Klassen: 0 Hintergrund, 1 Hauptschild, 2 andere Schilder (Zusatzschild), 3 Mast
"""

import cv2
import numpy as np

KLASSEN = ["Hintergrund", "Hauptschild", "andere Schilder", "Mast"]


def seg_zu_klassen(seg_rgb: np.ndarray, maske: np.ndarray) -> np.ndarray:
    """seg_rgb: Segmentierung (H, W, 3, RGB); maske: Hauptschild-Maske (H, W), >127 = Schild."""
    hsv = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2HSV)            # OpenCV: H in 0..179
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    out = np.zeros(seg_rgb.shape[:2], np.uint8)
    out[(h >= 85) & (h <= 100) & (s > 150) & (v > 40)] = 2   # türkis: andere Schilder
    out[((h >= 160) | (h <= 5)) & (s > 120) & (v > 40)] = 3  # weinrot: Mast (Schildrand liegt in der Maske)
    out[maske > 127] = 1
    # JPEG-Ränder glätten: kleine Inseln entfernen
    for k in (2, 3):
        m = cv2.morphologyEx((out == k).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        out[(out == k) & (m == 0)] = 0
    return out


def farbkarte(klassen: np.ndarray) -> np.ndarray:
    """Zur Ansicht: Klassenkarte als RGB-Bild."""
    farben = np.array([[40, 40, 40], [230, 60, 60], [0, 180, 200], [220, 0, 220]], np.uint8)
    return farben[klassen]


def zusatz_boxen(karte: np.ndarray, min_anteil: float = 0.002) -> list:
    """Boxen (x0, y0, x1, y1) der Zusatzschilder in einer Klassenkarte, von oben nach unten.

    Zusammenhängende Flächen der Klasse "andere Schilder", die unterhalb der Mitte des
    Hauptschilds liegen; sehr kleine Flächen (Rauschen) fallen weg.
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats((karte == 2).astype(np.uint8), connectivity=8)
    haupt = np.argwhere(karte == 1)
    mitte_y = haupt[:, 0].mean() if len(haupt) else 0
    boxen = []
    for i in range(1, n):
        x, y, w, h, flaeche = stats[i]
        if flaeche < min_anteil * karte.size or y + h / 2 < mitte_y:
            continue
        boxen.append((int(x), int(y), int(x + w), int(y + h)))
    return sorted(boxen, key=lambda b: b[1])
