"""Kennzahlen des Zusatz-Kopfs an Hand-Beispielen mit bekanntem Ergebnis."""

import numpy as np
import pytest

from src.training.zusatz_metriken import zusatz_kennzahlen

K = ["LKW", "Bus", "Rad frei"]


def run(pred, true, gefahr=None):
    return zusatz_kennzahlen(np.array(pred), np.array(true), gefahr or [False] * len(true), K)


def test_alles_richtig():
    r = run([[1, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 0]])
    assert r["zusatz_exakt"] == 1.0
    assert r["zielgruppe_korrekt"] == 1.0
    assert r["zusatz_micro_f1"] == 1.0


def test_keins_richtig_erkannt_zaehlt_als_treffer():
    # leere Zeile = 'keins'; leere Vorhersage dazu ist exakt richtig
    r = run([[0, 0, 0]], [[0, 0, 0]])
    assert r["zusatz_exakt"] == 1.0
    assert r["n_zielgruppe_relevant"] == 0          # 'keins' ändert die Zielgruppe nicht


def test_falsche_klasse():
    r = run([[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0]])
    assert r["zusatz_exakt"] == 0.5
    assert r["zielgruppe_korrekt"] == 0.5
    assert r["zusatz_f1_je_klasse"]["LKW"]["anzahl"] == 2


def test_zu_viele_klassen_nie_exakt():
    # wie Stufe 0 in E4: ein untrainierter Kopf meldet überall mehrere Klassen
    r = run([[1, 1, 1]] * 3, [[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert r["zusatz_exakt"] == 0.0
    assert r["zusatz_micro_f1"] == pytest.approx(2 * 2 / (2 * 2 + 7 + 0))


def test_leer():
    assert run(np.zeros((0, 3)), np.zeros((0, 3)))["zusatz_exakt"] is None
