"""
Klassen des Zusatzzeichen-Kopfs, gemeinsam genutzt von review_crops.py und dem Training.

Jede Klasse: (Anzeigename, Mapillary-Werte). Der Synset-Name kommt aus
configs/mapillary_to_synset.csv. Die Reihenfolge bestimmt die Tasten im Prüfwerkzeug
und den Index im Modell; neue Klassen deshalb vor "andere" einfügen und danach neu trainieren.
"""

# Vorläufige Klassenliste: nach dem Lauf über ganz Deutschland anhand der Häufigkeiten überarbeiten.
# Der Synset-Name kommt aus configs/mapillary_to_synset.csv (leer = kein Gegenstück in Synset).
KLASSEN = [
    ("Rad frei", ["complementary--except-bicycles--g1"]),
    ("LKW, Bus, Anhänger", ["complementary--trucks-buses-trailers--g1", "complementary--trucks-and-trailers--g1"]),
    ("LKW", ["complementary--trucks--g1"]),
    ("Bus", ["complementary--buses--g1"]),
    ("Bus frei", ["complementary--except-buses--g1"]),
    ("Krafträder", ["complementary--motorcycles--g2"]),
    # Mapillary "vehicles"/"except-vehicles" zeigen auf deutschen Fotos ein PKW-Sinnbild (geprüft an 8 + 8 Fotos)
    ("PKW", ["complementary--vehicles--g1", "complementary--vehicles--g2"]),
    ("PKW frei", ["complementary--except-vehicles--g1"]),
    ("Schwerbehinderte", ["complementary--disabled-persons--g1"]),
    ("Straßenbahn frei", ["complementary--except-trams--g1"]),
    ("Straßenbahn", ["complementary--trams--g1"]),
    ("Landwirtschaft frei", ["complementary--except-tractors--g1"]),
    # Mapillary "dangerous-or-pollutant-good" ist auf deutschen Fotos das Hinweiszeichen 354
    # "Wasserschutzgebiet", kein Zusatzzeichen; die Klasse bleibt für echtes 1052-30, ohne Mapillary-Wert.
    ("Gefahrgut", []),
    ("Umweltzone frei", ["complementary--except-polluting-level-green--g1"]),
    ("Text (Anlieger frei u. a.)", ["complementary--texts-two-lines--g1", "complementary--texts-three-lines--g1"]),
    ("Länge einer Strecke", ["complementary--distance--g1"]),
    ("Entfernung", ["complementary--distance--g2"]),
    ("Pfeil rechts", ["complementary--turn-right--g2"]),
    ("Pfeil links", ["complementary--turn-left--g2"]),
    ("Pfeile beidseitig", ["complementary--two-way-traffic--g2", "complementary--two-way-traffic--g3",
                           "complementary--both-directions--g2"]),
    ("Abknickende Vorfahrt", ["complementary--priority-route-at-intersection--g1",
                              "complementary--priority-route-at-intersection--g2",
                              "complementary--priority-route-at-intersection--g3",
                              "complementary--priority-route-at-intersection--g4"]),
    ("Fußgänger Pfeil", ["complementary--pedestrians-left--g1", "complementary--pedestrians-right--g1"]),
    ("Bei Nässe", ["complementary--when-wet--g1"]),
    ("Zeitangabe", ["complementary--time-restrictions--g3", "complementary--working-days--g1"]),
    ("Abschleppen", ["complementary--tow-away-zone--g1"]),
    ("andere", []),
    ("keins", []),
]

# Für das Modell: "keins" ist kein Ausgang, sondern "alle Ausgänge 0"
MODELL_KLASSEN = [name for name, _ in KLASSEN if name != "keins"]
