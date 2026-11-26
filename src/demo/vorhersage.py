"""
Vorhersage für den Demonstrator: Ausschnitt eines Schildmasts -> Hauptschild, Zusatzzeichen, Regel, Grad-CAM.

Unabhängig von der Oberfläche (scripts/demo/app.py), damit es testbar bleibt.
Der Ausschnitt wird wie im Training vorverarbeitet: CLAHE, Letterbox auf ein Quadrat, Normalisieren.
"""

import cv2
import numpy as np
import torch

from src.data.multitask_data import _CLAHE, build_transform, letterbox
from src.models.multitask import MultiTaskResNet
from src.utils.zielgruppen import ist_gefahrzeichen, lesbar, regel

SCHWELLE = 0.5      # wie in der Auswertung: Zusatzzeichen ab 50 %


class Demonstrator:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        cfg = state["config"]
        self.haupt_klassen = state["haupt_klassen"]
        self.zusatz_klassen = state["zusatz_klassen"]
        self.size = cfg["data"]["image_size"]
        self.clahe = cfg["data"].get("clahe", True)
        self.epoche = state.get("epoche")
        self.device = device
        self.model = MultiTaskResNet(len(self.haupt_klassen), len(self.zusatz_klassen), pretrained=False)
        self.model.load_state_dict(state["model"])
        self.model.eval().to(device)
        self.transform = build_transform(False, self.size)

    def _eingabe(self, rgb: np.ndarray):
        if self.clahe:
            rgb = _CLAHE(rgb)
        quadrat = letterbox(rgb, self.size)
        return self.transform(quadrat).unsqueeze(0).to(self.device), quadrat

    @torch.no_grad()
    def vorhersage(self, rgb: np.ndarray) -> dict:
        """rgb: Ausschnitt (H x W x 3, uint8, RGB)."""
        x, _ = self._eingabe(rgb)
        out = self.model(x)
        p_haupt = torch.softmax(out["haupt"][0], 0)
        top = torch.topk(p_haupt, 3)
        p_da = float(torch.sigmoid(out["da"][0]))
        p_zusatz = torch.sigmoid(out["zusatz"][0]).cpu().numpy()
        zusatz = [(self.zusatz_klassen[i], float(p_zusatz[i])) for i in np.argsort(-p_zusatz)
                  if p_zusatz[i] >= SCHWELLE]
        haupt = self.haupt_klassen[int(top.indices[0])]
        r = regel(haupt, [k for k, _ in zusatz], ist_gefahrzeichen(haupt))
        return {"haupt": [(lesbar(self.haupt_klassen[int(i)]), float(p)) for p, i in zip(top.values, top.indices)],
                "zusatz_vorhanden": p_da,
                "zusatz": zusatz,
                "zusatz_knapp": [(self.zusatz_klassen[i], float(p_zusatz[i])) for i in np.argsort(-p_zusatz)
                                 if 0.2 <= p_zusatz[i] < SCHWELLE][:3],   # nur die drei stärksten Kandidaten
                "regel": r}

    def grad_cam(self, rgb: np.ndarray, kopf: str = "zusatz") -> np.ndarray:
        """Heatmap über dem quadratischen Eingabebild für die stärkste Klasse des Kopfs ('haupt' oder 'zusatz')."""
        x, quadrat = self._eingabe(rgb)
        merkmale = self.model.features(x)
        merkmale.retain_grad()
        f = torch.flatten(self.model.pool(merkmale), 1)
        logits = self.model.head_haupt(f) if kopf == "haupt" else self.model.head_zusatz(f)
        self.model.zero_grad()
        logits[0, logits[0].argmax()].backward()
        gewichte = merkmale.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((gewichte * merkmale).sum(1))[0].detach().cpu().numpy()
        cam = cv2.resize(cam / (cam.max() + 1e-8), (quadrat.shape[1], quadrat.shape[0]))
        farbe = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        return np.uint8(0.55 * quadrat + 0.45 * farbe)
