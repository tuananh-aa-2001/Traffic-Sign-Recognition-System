"""Tests der Zielgruppen-Tabelle. Ausführen: python -m pytest tests  (oder: python tests/test_zielgruppen.py)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.zusatz_klassen import KLASSEN
from src.utils.zielgruppen import ZUSATZ_REGELN, lesbar, regel, zielgruppe_gleich


def test_jede_klasse_hat_eine_regel():
    fehlend = [name for name, _ in KLASSEN if name != "keins" and name not in ZUSATZ_REGELN]
    assert not fehlend, fehlend


def test_ueberholverbot_nur_lkw_bus_anhaenger():           # Folie 3
    r = regel("Ueberholverbot", ["LKW, Bus, Anhänger"])
    assert r["gilt_fuer"] == ["LKW über 3,5 t", "Bus", "PKW mit Anhänger"]
    assert r["nicht_betroffen"] == ["PKW", "Krafträder", "Fahrrad"]
    assert r["text"].startswith("Überholverbot · gilt nur für:")


def test_gehweg_rad_frei():
    r = regel("Gehweg", ["Rad frei"])
    assert r["gilt_fuer"] == ["alle"] and r["ausgenommen"] == ["Fahrrad"]
    assert r["text"] == "Gehweg · ausgenommen: Fahrrad"


def test_sinnbild_unter_gefahrzeichen_ist_hinweis():
    r = regel("Gefahrenstelle", ["Straßenbahn"])
    assert r["gilt_fuer"] == ["alle"] and r["hinweise"] == ["Gefahr durch Straßenbahn"]
    assert regel("warning--other-danger--g1", ["Straßenbahn"])["hinweise"] == ["Gefahr durch Straßenbahn"]


def test_sinnbild_unter_verbot_ist_nur_fuer():
    r = regel("Parken", ["Bus"])
    assert r["gilt_fuer"] == ["Bus"]


def test_mehrere_zusatzzeichen():
    r = regel("Geschwindigkeit70", ["Krafträder", "Länge einer Strecke"])
    assert r["gilt_fuer"] == ["Krafträder"] and r["hinweise"] == ["auf der angegebenen Strecke"]
    r = regel("AbsolutesHalteverbot", ["Abschleppen", "Zeitangabe"])
    assert r["gilt_fuer"] == ["alle"] and r["bedingungen"] == ["nur zu den angegebenen Zeiten"]


def test_ohne_zusatz_gilt_fuer_alle():
    assert regel("Stop", ["keins"])["text"] == "Stop"


def test_nur_sondergruppe():
    r = regel("Parken", ["Schwerbehinderte"])
    assert r["nicht_betroffen"] == ["alle anderen"]


def test_zielgruppe_gleich():
    assert zielgruppe_gleich(regel("A", ["Rad frei", "Pfeil links"]), regel("B", ["Rad frei"]))
    assert not zielgruppe_gleich(regel("A", ["Rad frei"]), regel("A", ["Bus frei"]))


def test_lesbar():
    assert lesbar("VorfahrtGewaehren") == "Vorfahrt Gewähren"
    assert lesbar("BeginnFussgaengerzone") == "Beginn Fußgängerzone"
    assert lesbar("Vorfahrtsstrasse") == "Vorfahrtsstraße"
    assert lesbar("Geschwindigkeit30") == "Geschwindigkeit 30"


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
