"""
Von Haupt- und Zusatzzeichen zur Regel: Für wen gilt das Schild?

Keine Netz-Ausgabe, sondern eine Tabelle nach der StVO. Jede Klasse des Zusatz-Kopfs
(src/data/zusatz_klassen.py) hat eine Wirkung:

  nur_fuer     das Hauptschild gilt nur für diese Gruppen            (1049-13 LKW, Bus, PKW mit Anhänger)
  ausgenommen  das Hauptschild gilt nicht für diese Gruppen          (1022-10 Radverkehr frei)
  sinnbild     Fahrzeug-Sinnbild: unter Verbot/Gebot "nur für", unter einem Gefahrzeichen ein Hinweis
               auf die Gefahr (Gefahrenstelle + Straßenbahn = Straßenbahn kreuzt)
  bedingung    gilt nur unter einer Bedingung                         (1053-35 bei Nässe, Zeitangaben)
  hinweis      ändert nicht, für wen es gilt (Entfernung, Pfeile, Abschleppen)
  unklar       ohne Texterkennung nicht bestimmbar                    (Textschilder, "andere")

Quellen der Zeichennummern: Wikipedia "Bildtafel der Verkehrszeichen in der Bundesrepublik
Deutschland seit 2017", Abschnitt Zusatzzeichen (abgerufen 20.11.2026). "sicherheit" sagt, wie sicher
die Mapillary-Klasse dem genannten Zeichen entspricht; LKW, PKW und PKW frei sind zusätzlich an je
8 Mapillary-Fotos geprüft.

Beispiel:
  >>> regel("Ueberholverbot", ["LKW, Bus, Anhänger"])["text"]
  'Überholverbot · gilt nur für: LKW über 3,5 t, Bus, PKW mit Anhänger · nicht betroffen: PKW, Krafträder, Fahrrad'
"""

import re

PKW, PKW_ANH, LKW, BUS, KRAD = "PKW", "PKW mit Anhänger", "LKW über 3,5 t", "Bus", "Krafträder"
FAHRRAD, FUSS, TRAM, LANDW = "Fahrrad", "Fußgänger", "Straßenbahn", "Landwirtschaftlicher Verkehr"

# Gruppen, für die ein Verkehrszeichen ohne Zusatz in der Regel gilt ("nicht betroffen" = Rest davon)
BASIS = [PKW, PKW_ANH, LKW, BUS, KRAD, FAHRRAD]

# Klasse -> (Wirkung, Gruppen oder Bedingungstext, StVO-Zusatzzeichen, Sicherheit der Zuordnung)
ZUSATZ_REGELN = {
    "Rad frei":                   ("ausgenommen", [FAHRRAD], "1022-10", "sicher"),
    "LKW, Bus, Anhänger":         ("nur_fuer", [LKW, BUS, PKW_ANH], "1049-13", "sicher"),
    "LKW":                        ("sinnbild", [LKW], "1010-51", "sicher"),
    "Bus":                        ("sinnbild", [BUS], "1010-57", "wahrscheinlich"),
    "Bus frei":                   ("ausgenommen", [BUS], "1024-14", "sicher"),
    "Krafträder":                 ("sinnbild", [KRAD], "1010-62", "wahrscheinlich"),
    "PKW":                        ("sinnbild", [PKW], "1010-58", "sicher"),
    "PKW frei":                   ("ausgenommen", [PKW], "1024-10", "sicher"),
    "Schwerbehinderte":           ("nur_fuer", ["Schwerbehinderte mit Parkausweis"], "1044-10", "sicher"),
    "Straßenbahn frei":           ("ausgenommen", [TRAM], "1024-16", "sicher"),
    "Straßenbahn":                ("sinnbild", [TRAM], "1010-56", "wahrscheinlich"),
    "Landwirtschaft frei":        ("ausgenommen", [LANDW], "1026-36", "sicher"),
    "Gefahrgut":                  ("nur_fuer", ["Fahrzeuge mit gefährlichen Gütern"], "1052-30", "sicher"),
    "Umweltzone frei":            ("ausgenommen", ["Kfz mit grüner Plakette"], "1031-52", "wahrscheinlich"),
    "Text (Anlieger frei u. a.)": ("unklar", "Text lesen (z. B. Anlieger frei)", "", "unsicher"),
    "Länge einer Strecke":        ("hinweis", "auf der angegebenen Strecke", "", "wahrscheinlich"),
    "Entfernung":                 ("hinweis", "in der angegebenen Entfernung", "1004-30", "wahrscheinlich"),
    "Pfeil rechts":               ("hinweis", "Richtung rechts", "", "sicher"),
    "Pfeil links":                ("hinweis", "Richtung links", "", "sicher"),
    "Pfeile beidseitig":          ("hinweis", "in beide Richtungen", "", "sicher"),
    "Abknickende Vorfahrt":       ("hinweis", "Verlauf der Vorfahrtstraße", "", "sicher"),
    "Fußgänger Pfeil":            ("hinweis", "Fußgänger: Gehweg gegenüber benutzen", "1000-12/1000-22", "sicher"),
    "Bei Nässe":                  ("bedingung", "nur bei Nässe", "1053-35", "sicher"),
    "Zeitangabe":                 ("bedingung", "nur zu den angegebenen Zeiten", "1040-30/1042-3x", "wahrscheinlich"),
    # kein amtliches Zusatzzeichen (fehlt in der Bildtafel), sondern ein Hinweisschild
    "Abschleppen":                ("hinweis", "verbotswidrig abgestellte Fahrzeuge werden abgeschleppt", "nicht amtlich", "sicher"),
    "andere":                     ("unklar", "unbekanntes Zusatzzeichen", "", "unsicher"),
}

# Gefahrzeichen in Synset (Präfixe der Klassennamen); unter ihnen ist ein Fahrzeug-Sinnbild ein Hinweis
_GEFAHR_PRAEFIXE = (
    "Amphibienwanderung", "Andreaskreuz", "Arbeitsstelle", "Bahnuebergang", "BeweglicheBruecke", "Doppelkurve",
    "EinseitigVerengteFahrbahn", "Flugbetrieb", "FussgaengerLinks", "FussgaengerRechts", "Fussgaengerueberweg",
    "Gefaelle", "Gefahrenstelle", "Gegenverkehr", "Kinder", "KreuzungOderEinmuendung", "Kurve", "Lichtzeichenanlage",
    "RadverkehrLinks", "RadverkehrRechts", "ReiterLinks", "ReiterRechts", "SchleuderOderRutschgefahr",
    "SchneeOderEisglaette", "Seitenwind", "SplittSchotter", "Stau", "Steigung", "Steinschlag", "Ufer",
    "UnebeneFahrbahn", "UnzureichendesLichtraumprofil", "VerengteFahrbahn", "Viehtrieb", "Wildwechsel",
)


def ist_gefahrzeichen(hauptschild: str) -> bool:
    """Synset-Name (z. B. 'Gefahrenstelle') oder Mapillary-Wert (z. B. 'warning--other-danger--g1')."""
    return hauptschild.startswith("warning--") or hauptschild.startswith(_GEFAHR_PRAEFIXE)


def lesbar(name: str) -> str:
    """Synset-Klassenname lesbar machen: 'VerbotFuerKraftfahrzeugeUeber...' -> 'Verbot Für Kraftfahrzeuge Über ...'."""
    if "--" in name:                                     # Mapillary-Wert unverändert lassen
        return name
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[a-z])(?=[0-9])", " ", name)
    s = s.replace("Ae", "Ä").replace("Oe", "Ö").replace("Ue", "Ü")
    s = s.replace("ae", "ä").replace("oe", "ö").replace("ue", "ü")
    s = re.sub(r"(F|f)uss", r"\1uß", s)
    s = s.replace("strasse", "straße").replace("Strasse", "Straße").replace("Gross", "Groß")
    return s


def regel(hauptschild: str, zusatz_klassen: list, gefahrzeichen: bool = None) -> dict:
    """Regel aus Hauptschild und erkannten Zusatzzeichen-Klassen.

    Rückgabe: gilt_fuer (Liste oder ['alle']), ausgenommen, nicht_betroffen, bedingungen,
    hinweise, unklar, text.
    """
    if gefahrzeichen is None:
        gefahrzeichen = ist_gefahrzeichen(hauptschild)
    gilt, aus, bed, hin, unklar = None, [], [], [], []
    for k in zusatz_klassen:
        if k in ("keins", ""):
            continue
        wirkung, wert, _, _ = ZUSATZ_REGELN.get(k, ZUSATZ_REGELN["andere"])
        if wirkung == "sinnbild":
            wirkung = "hinweis" if gefahrzeichen else "nur_fuer"
            if wirkung == "hinweis":
                wert = "Gefahr durch " + ", ".join(wert)
        if wirkung == "nur_fuer":
            gilt = list(wert) if gilt is None else [g for g in gilt if g in wert] or list(wert)
        elif wirkung == "ausgenommen":
            aus += [g for g in wert if g not in aus]
        elif wirkung == "bedingung":
            bed.append(wert)
        elif wirkung == "hinweis":
            hin.append(wert)
        else:
            unklar.append(wert)
    if gilt is not None:
        gilt = [g for g in gilt if g not in aus]
        nicht = [g for g in BASIS if g not in gilt]
        if not any(g in BASIS for g in gilt):           # z. B. nur Schwerbehinderte
            nicht = ["alle anderen"]
    else:
        gilt, nicht = ["alle"], []

    teile = [lesbar(hauptschild)]
    if gilt != ["alle"]:
        teile.append("gilt nur für: " + ", ".join(gilt))
    if aus:
        teile.append("ausgenommen: " + ", ".join(aus))
    if nicht:
        teile.append("nicht betroffen: " + ", ".join(nicht))
    teile += bed + hin
    if unklar:
        teile.append("unklar: " + ", ".join(unklar))
    return {"gilt_fuer": gilt, "ausgenommen": aus, "nicht_betroffen": nicht, "bedingungen": bed,
            "hinweise": hin, "unklar": unklar, "text": " · ".join(teile)}


def zielgruppe_gleich(a: dict, b: dict) -> bool:
    """Gleiche Zielgruppen-Wirkung (für die Auswertung): gilt_fuer und ausgenommen stimmen überein."""
    return sorted(a["gilt_fuer"]) == sorted(b["gilt_fuer"]) and sorted(a["ausgenommen"]) == sorted(b["ausgenommen"])
