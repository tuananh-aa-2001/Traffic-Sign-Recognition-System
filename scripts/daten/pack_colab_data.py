"""
Echte Daten für Colab verpacken (sie liegen wegen .gitignore nicht im Repo).

Packt in data/colab_daten.zip:
  crops/                            Ausschnitte, labels.csv, geprueft.csv (ohne Kontrollbilder)
  real_world_mapillary/labels.csv   Orte der Fotos (für den Split nach Region)
  configs/mapillary_to_synset.csv   Zuordnung Mapillary -> Synset

Die ZIP danach in Google Drive nach MyDrive/dlacv_gtsrb/ hochladen.

  python scripts/daten/pack_colab_data.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # Projektordner (src/)
import os
import zipfile

ZIEL = os.path.join("data", "colab_daten.zip")
DATEIEN = [os.path.join("real_world_mapillary", "labels.csv"), os.path.join("configs", "mapillary_to_synset.csv")]

os.makedirs("data", exist_ok=True)
with zipfile.ZipFile(ZIEL, "w", zipfile.ZIP_STORED) as z:     # JPEGs sind schon komprimiert
    n = 0
    for root, _, files in os.walk("crops"):
        if os.path.basename(root).startswith("_"):            # _kontrolle, _kontrolle_gerettet
            continue
        for name in files:
            z.write(os.path.join(root, name))
            n += 1
    for path in DATEIEN:
        if os.path.exists(path):
            z.write(path)
            n += 1
        else:
            print(f"Warnung: {path} fehlt")
print(f"{ZIEL}: {n} Dateien, {os.path.getsize(ZIEL) / 1e6:.1f} MB")
