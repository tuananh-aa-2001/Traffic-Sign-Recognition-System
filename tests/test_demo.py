import os

import numpy as np
import pytest
import torch

from src.demo.vorhersage import Demonstrator
from src.models.multitask import MultiTaskResNet


@pytest.fixture
def modell(tmp_path):
    """Zufällig initialisiertes Mini-Checkpoint im Format von train_multitask.py."""
    haupt, zusatz = ["Parken", "UeberholverbotFuerKraftfahrzeugeAllerArt", "Gehweg"], ["Rad frei", "LKW"]
    m = MultiTaskResNet(len(haupt), len(zusatz), pretrained=False)
    pfad = os.path.join(tmp_path, "modell.pth")
    torch.save({"model": m.state_dict(), "config": {"data": {"image_size": 64, "clahe": True}}, "epoche": 1,
                "haupt_klassen": haupt, "zusatz_klassen": zusatz}, pfad)
    return Demonstrator(pfad)


def test_vorhersage_liefert_regel(modell):
    bild = np.random.default_rng(0).integers(0, 255, (90, 50, 3), dtype=np.uint8)   # hochkant wie ein Mast
    e = modell.vorhersage(bild)
    assert len(e["haupt"]) == 3 and abs(sum(p for _, p in e["haupt"])) <= 1.0001
    assert 0 <= e["zusatz_vorhanden"] <= 1
    assert all(p >= 0.5 for _, p in e["zusatz"])
    assert {"gilt_fuer", "ausgenommen", "nicht_betroffen"} <= e["regel"].keys()


def test_grad_cam_hat_eingabegroesse(modell):
    bild = np.random.default_rng(1).integers(0, 255, (90, 50, 3), dtype=np.uint8)
    cam = modell.grad_cam(bild)
    assert cam.shape == (64, 64, 3) and cam.dtype == np.uint8
