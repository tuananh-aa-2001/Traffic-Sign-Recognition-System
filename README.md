# Kontexterkennung von Schildkombinationen

**Haupt- und Zusatzzeichen mit Multi-Task-Lernen aus synthetischen und echten Daten, geprüft an Straßenfotos aus NRW.**

Projekt im Modul *Deep Learning Applications in Computer Vision* (DLACV), FH Aachen, Fachbereich 5, Wintersemester 2026/27.

Ein Verkehrszeichen gilt oft nur zusammen mit seinem Zusatzzeichen: „Überholverbot“ plus „LKW, Bus, PKW mit Anhänger“ gilt nur für diese Fahrzeuge, „Gehweg“ plus „Radfahrer frei“ erlaubt auch das Radfahren. Bekannte Benchmarks wie GTSRB enthalten nur das Hauptschild. Dieses Projekt erkennt die ganze Kombination und leitet daraus ab, für welche Zielgruppe (LKW, PKW, Bus, Fahrrad, Fußgänger …) die Regel gilt.

**Forschungsfrage:** Wie viele echte Bilder braucht ein auf synthetischen Daten trainiertes Multi-Task-Modell, um Zusatzzeichen und Zielgruppen auf echten Fotos zu erkennen? Startpunkt der Lernkurve sind null echte Bilder, also das, was Synset allein leistet.

---

## Stand (20.11.2026)

| Baustein | Stand |
|---|---|
| GTSRB-Baseline (ResNet-18, 43 Klassen) | fertig: 99,36 % auf dem GTSRB-Testset, 17 ms je Schild auf der Laptop-CPU ([Details](docs/gtsrb_baseline.md)) |
| Synset-Stichprobe | fertig: Zusatzschild in 170 von 170 Bildern unten abgeschnitten (siehe [Daten](#daten)) |
| Echte Fotos NRW (Test) | 654 Mapillary-Fotos, 385 Kombinationen ausgeschnitten, Testsatz (86) von Hand geprüft |
| Echte Fotos Deutschland (Training) | gezielte Suche nach 15 Zusatzzeichen mit Zielgruppe in 66 Städten läuft |
| Architektur A | Code fertig, Probelauf auf CPU erfolgreich; Training auf Colab steht aus |
| E1 (Synset gegen GTSRB) | Code fertig, Probelauf erfolgreich; Lauf auf Colab steht aus |
| E4 Lernkurve, Grad-CAM-Beispiele, Demo | offen |
| Architektur B (U-Net, zweistufig) | optional (Ausblick); Code vorbereitet, Probelauf erfolgreich |

---

## Daten

| Quelle | Inhalt | Rolle | Lizenz |
|---|---|---|---|
| [Synset Signset Germany](https://huggingface.co/datasets/FraunhoferIOSB/Synset-Signset-Germany) (Cycles) | 84.400 gerenderte Bilder, 211 Hauptschild-Klassen (209 verschiedene Namen), 20 % mit Schild darunter | Hauptschild und „Zusatzschild vorhanden?“ | CC BY 4.0 |
| Mapillary, gezielte Suche in 66 deutschen Städten | bis 150 Fotos je Zusatzzeichen mit Zielgruppe | welches Zusatzzeichen | CC BY-SA 4.0 |
| Mapillary, 21 Regionen in NRW | 654 Fotos | **Test:** Aachen, Eifel, Köln; übrige Regionen Training/Validierung | CC BY-SA 4.0 |
| GTSRB | 51.800 echte Ausschnitte, 43 Klassen | Baseline, Vergleich synthetisch gegen echt beim Hauptschild | – |

**Warum echte Fotos fürs Zusatzzeichen?** Die Synset-Ausschnitte sind auf das Hauptschild zugeschnitten. In einer Stichprobe von 170 Bildern mit Zusatzschild war dieses jedes Mal am unteren Bildrand abgeschnitten; sichtbar ist im Median ein Streifen von 0,30 Hauptschildhöhen. Per Augenschein war das Zusatzzeichen nur in 9 von 42 Bildern erkennbar, bei Zusatzzeichen mit Zielgruppe in 1 von 16. Synset lehrt also, *ob* ein Zusatzschild da ist, kaum aber *welches*.

**Split nach Ort:** Ausschnitte aus Aachen, Eifel und Köln sind der feste Testsatz und kommen nie ins Training (auch nicht über die Deutschland-Suche). Das Modell wird an Bonn, Düsseldorf, Münster, Berlin, München und Hamburg ausgewählt, nie am Test. Dasselbe Schild steht oft auf mehreren Fotos, ein zufälliger Split würde den Test schönen.

**Zuordnung Mapillary → Synset:** [`configs/mapillary_to_synset.csv`](configs/) ordnet 85 Mapillary-Hauptschilder und 31 Zusatzzeichen den Synset-Klassen zu, mit Sicherheitsstufe und Begründung je Zeile (lokal, wegen `*.csv` in `.gitignore`).

Fotos, Ausschnitte und vorbereitete Synset-Daten liegen nicht im Repo (Größe, Lizenz CC BY-SA); die Skripte unten erzeugen sie neu.

---

## Architektur A: ein Netz, drei Köpfe

```
Bild 128×128 ──► ResNet-18 (ImageNet) ──┬─► Hauptschild      209 Klassen, Softmax        ◄ Synset (+ echte, wenn zuordenbar)
                                        ├─► Zusatz vorhanden? 1 Logit, Sigmoid            ◄ Synset + echte
                                        └─► Zusatzzeichen     26 Klassen, Multi-Label     ◄ echte Ausschnitte
                                                                      │
                                                                      ▼
                                                  Regel + Zielgruppe (Tabelle nach StVO)
```

Maskierter Verlust: Jeder Kopf lernt nur aus Bildern, die sein Label haben.

$$L = \lambda_1 L_\text{haupt} + \lambda_2\, m_\text{da} L_\text{da} + \lambda_3\, m_\text{zusatz} L_\text{zusatz}, \quad m \in \{0, 1\}$$

- Eingabe 128×128 statt 48×48 wie bei GTSRB, damit Text und Sinnbild des Zusatzschilds lesbar bleiben.
- Zusatz-Kopf als Multi-Label (gestapelte Zusatzschilder), Focal Loss gegen seltene Klassen.
- Je Batch ca. 30 % echte Ausschnitte, damit sie neben 84.400 Synset-Bildern nicht untergehen.
- Augmentierung ohne Spiegeln (Pfeile und Richtungen würden falsch).
- Klassenliste des Zusatz-Kopfs: [`src/data/zusatz_klassen.py`](src/data/zusatz_klassen.py), vorläufig 26 Klassen, wird nach der Deutschland-Suche anhand der Häufigkeiten festgelegt.

**Architektur B** (optional, Ausblick): U-Net auf den Synset-Klassenkarten trennt Hauptschild, andere Schilder und Mast; aus der Karte entstehen Boxen, jedes Zusatzschild wird einzeln klassifiziert (`train_zweistufig.py`, Code und Probelauf fertig).

---

## Ablauf

### 0. Umgebung

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Für Mapillary einen Client-Token anlegen (https://www.mapillary.com/dashboard/developers) und in `mapillary_token.txt` speichern (steht in `.gitignore`).

### 1. Echte Fotos sammeln

```powershell
python scrape_mapillary.py --quelle mapillary                      # 21 Regionen in NRW (Test + Training)
python scrape_mapillary.py --quelle zielgruppen                    # 15 Zielgruppen-Zusatzzeichen in 66 Städten
python scrape_mapillary.py --quelle zielgruppen --testregionen     # dieselben Zeichen nur in Aachen, Eifel, Köln
```

### 2. Schildkombinationen ausschneiden

```powershell
python crop_signs.py mapillary
```

Nutzt die Schild-Umrisse von Mapillary: Hauptschild plus die Zusatzzeichen direkt darunter, 10 % Rand. Zusatzzeichen ohne erkanntes Hauptschild werden per Farbsuche gerettet (`crops/mapillary_gerettet/`).

### 3. Labels von Hand prüfen

```powershell
python review_crops.py --split test        # zuerst der Testsatz
python review_crops.py                     # dann die Trainingsausschnitte
python review_crops.py --statistik         # Zählung je Klasse und Split
```

Fenster mit Ausschnitt, Umgebung und Vorschlag; Enter bestätigt, Tasten korrigieren. Ergebnisse in `crops/geprueft.csv`.

### 4. Synset vorbereiten

```powershell
python prepare_synset.py --split validation
python prepare_synset.py --split train      # liest 7,4 GB, besser auf Colab
```

Liest die Parquet-Datei Zeilengruppe für Zeilengruppe (auch direkt von Hugging Face) und speichert die Bilder als 128×128-Array in `data/synset/`.

### 5. Architektur A trainieren

```powershell
python train_multitask.py --smoke           # Probelauf auf der CPU (ca. 20 s)
```

Das eigentliche Training läuft auf Google Colab: [`notebooks/train_multitask_colab.ipynb`](notebooks/train_multitask_colab.ipynb). Vorher `python pack_colab_data.py` ausführen und `colab_daten.zip` nach `MyDrive/dlacv_gtsrb/` hochladen. Ergebnis: `results/multitask/ergebnis.json` (Kennzahlen je Kopf, F1 je Zusatzzeichen).

### 6. Experiment E1

```powershell
python experiment_e1.py --smoke             # Probelauf auf der CPU
python experiment_e1.py                     # auf Colab: Abschnitt 9 im Notebook
```

Trainiert die GTSRB-Architektur (ResNet-18, 48×48) auf den 43 Synset-Klassen, die den GTSRB-Klassen entsprechen (Label 0–42, Namen geprüft), und vergleicht mit dem GTSRB-Modell in einer 2×2-Tabelle (trainiert auf Synset/GTSRB × getestet auf Synset/GTSRB). Ergebnis: `results/e1/ergebnis.json`.

---

## Evaluation

Umfang am 20.11.2026 gekürzt: Kern sind E1 und E4, E4 beantwortet die Forschungsfrage.

| | Vergleich | Metrik | Status |
|---|---|---|---|
| E1 | Synset gegen GTSRB beim Hauptschild (Klassen 0–42) | Accuracy, Makro-F1 | Kern |
| E2 | A auf der Synset-Validierung | Accuracy je Kopf | Kern (fällt beim Training mit ab) |
| E4 | Lernkurve: 0 / 25 / 50 / 100 / 150 echte Bilder je Zusatzzeichen | Zusatz exakt, Zielgruppe korrekt (Test) | Kern |
| E7 | Laufzeit von A als ONNX auf der Laptop-CPU | ms je Schild | ein Messwert |
| E3 | A gegen B auf dem echten Testsatz | wie E4 | optional |
| E5, E6 | Bedingungen; Kombinations-Prior, Focal Loss gegen Kreuzentropie | – | gestrichen |

---|---|---|
| E1 | Synset gegen GTSRB beim Hauptschild (Klassen 0–42) | Accuracy, Makro-F1 |
| E2 | A auf der Synset-Validierung | Accuracy je Kopf, exakte Trefferquote der Kombination |
| E3 | A gegen B auf dem echten Testsatz | wie E2, dazu Abfall gegenüber Synset |
| E4 | Lernkurve: 0 / 25 / 50 / 100 / 150 echte Bilder je Zusatzzeichen | exakte Trefferquote auf dem Test |
| E5 | Bedingungen (Tageszeit, Kontrast, Unschärfe) | Accuracy je Bedingung |
| E6 | Kombinations-Prior; Kreuzentropie gegen Focal Loss | Makro-F1 der Zusatzzeichen |
| E7 | Laufzeit A gegen B als ONNX auf der Laptop-CPU | ms je Schild |

---

## Projektstruktur

```
├── scrape_mapillary.py        Fotos von Mapillary laden (NRW-Regionen, Zielgruppen in Deutschland)
├── crop_signs.py              Schildkombinationen aus Gesamtfotos ausschneiden
├── review_crops.py            Prüfwerkzeug für die Ausschnitte
├── prepare_synset.py          Synset Signset Germany vorbereiten
├── train_multitask.py         Architektur A trainieren und auswerten
├── experiment_e1.py           E1: Synset gegen GTSRB beim Hauptschild
├── pack_colab_data.py         echte Daten für Colab verpacken
├── train_gtsrb.py             GTSRB-Baseline: Training
├── evaluate_gtsrb.py          GTSRB-Baseline: Auswertung
├── export_onnx_gtsrb.py       GTSRB-Baseline: ONNX-Export
├── demo_video.py              Live-Demo der GTSRB-Baseline (Video/Webcam)
├── configs/
│   ├── multitask.yaml         Konfiguration Architektur A
│   ├── synset_klassen.json    Synset-Klassennamen (von prepare_synset.py)
│   └── gtsrb.yaml             Konfiguration GTSRB-Baseline
├── src/
│   ├── data/
│   │   ├── multitask_data.py  Datensätze Synset und echte Ausschnitte
│   │   ├── crop_records.py    Split und Labels der echten Ausschnitte
│   │   ├── zusatz_klassen.py  Klassenliste des Zusatz-Kopfs
│   │   └── …                  GTSRB-Datenlader, CLAHE, Augmentierung
│   ├── models/
│   │   ├── multitask.py       MultiTaskResNet und maskierter Verlust
│   │   └── resnet_gtsrb.py    GTSRB-Baseline
│   ├── training/              Trainer und Metriken der Baseline
│   └── utils/
│       └── zielgruppen.py     Zielgruppen-Tabelle nach StVO, Regel als Text
├── notebooks/
│   ├── train_multitask_colab.ipynb
│   └── train_gtsrb_colab.ipynb
├── praesentationen/           Präsentationen (.pptx)
└── docs/                      Dokumentation der GTSRB-Baseline (Anleitung, Ergebnisse, Code-Überblick)
```

Nur lokal (in `.gitignore`): `real_world_mapillary/`, `crops/`, `data/`, `models/`, `results/`, `kurs/`, `mapillary_token.txt`.

---

## Quellen

1. A. Sielemann, L. Loercher, M.-L. Schumacher, S. Wolf, M. Roschani, J. Ziehn und J. Beyerer, „Synset Signset Germany: A synthetic dataset for German traffic sign recognition,“ in *Proc. IEEE ITSC*, 2024, S. 3383–3390, doi: 10.1109/ITSC58415.2024.10920175.
2. J. Stallkamp, M. Schlipsing, J. Salmen und C. Igel, „Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition,“ *Neural Networks*, Bd. 32, S. 323–332, 2012, doi: 10.1016/j.neunet.2012.02.016.
3. O. Ronneberger, P. Fischer und T. Brox, „U-Net: Convolutional networks for biomedical image segmentation,“ in *Proc. MICCAI*, LNCS, Springer, 2015, S. 234–241, doi: 10.1007/978-3-319-24574-4_28.
4. K. He, X. Zhang, S. Ren und J. Sun, „Deep residual learning for image recognition,“ in *Proc. IEEE CVPR*, 2016, S. 770–778, doi: 10.1109/CVPR.2016.90.
5. T.-Y. Lin, P. Goyal, R. Girshick, K. He und P. Dollár, „Focal loss for dense object detection,“ in *Proc. IEEE ICCV*, 2017, S. 2999–3007, doi: 10.1109/ICCV.2017.324.
6. R. R. Selvaraju et al., „Grad-CAM: Visual explanations from deep networks via gradient-based localization,“ in *Proc. IEEE ICCV*, 2017, S. 618–626, doi: 10.1109/ICCV.2017.74.
7. Mapillary, Straßenfotos verschiedener Nutzer (Urheber je Bild in `real_world_mapillary/labels.csv`), Lizenz CC BY-SA 4.0, https://www.mapillary.com. Zugriff: 18.11.2026.
