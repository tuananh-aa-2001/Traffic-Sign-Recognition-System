"""
Architektur A: ein gemeinsames ResNet-18 mit drei Köpfen.

  haupt   Hauptschild (Synset-Klassen, Softmax)
  da      ist ein Zusatzschild da? (ein Logit, Sigmoid)
  zusatz  welche Zusatzzeichen (Multi-Label, je ein Logit; gestapelte Schilder möglich)

Anders als ResNetGTSRB (48x48, 3x3-Eingang ohne Pooling) nutzt das Modell den Standard-Eingang
von ResNet-18. Die Eingabe ist größer (128x128), damit Text und Sinnbild des Zusatzschilds
erkennbar bleiben, und ImageNet-Gewichte lassen sich unverändert übernehmen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class MultiTaskResNet(nn.Module):
    def __init__(self, n_haupt: int, n_zusatz: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                      base.layer1, base.layer2, base.layer3, base.layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head_haupt = nn.Linear(512, n_haupt)
        self.head_da = nn.Linear(512, 1)
        self.head_zusatz = nn.Linear(512, n_zusatz)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Merkmalskarte vor dem Pooling (für Grad-CAM)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> dict:
        f = self.dropout(torch.flatten(self.pool(self.backbone(x)), 1))
        return {"haupt": self.head_haupt(f), "da": self.head_da(f).squeeze(1), "zusatz": self.head_zusatz(f)}


class MaskedMultiTaskLoss(nn.Module):
    """L = l1 * L_haupt + l2 * m_da * L_da + l3 * m_zusatz * L_zusatz

    Jeder Kopf mittelt nur über die Bilder, für die sein Label bekannt ist (Maske = 1).
    Synset liefert Hauptschild und "da", echte Ausschnitte liefern "da" und "zusatz"
    (und das Hauptschild, wenn es sich einer Synset-Klasse zuordnen lässt).
    Für den Zusatz-Kopf wahlweise Focal Loss gegen das starke Ungleichgewicht der Klassen.
    """

    def __init__(self, gewichte=(1.0, 1.0, 1.0), label_smoothing: float = 0.1, focal_gamma: float = 0.0):
        super().__init__()
        self.l_haupt, self.l_da, self.l_zusatz = gewichte
        self.label_smoothing = label_smoothing
        self.gamma = focal_gamma

    def _bce(self, logits, target):
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if self.gamma > 0:
            p = torch.sigmoid(logits)
            p_t = p * target + (1 - p) * (1 - target)
            loss = loss * (1 - p_t) ** self.gamma
        return loss

    def forward(self, out: dict, tgt: dict):
        zero = out["haupt"].sum() * 0.0
        teile = {}
        m = tgt["m_haupt"] > 0
        teile["haupt"] = (F.cross_entropy(out["haupt"][m], tgt["haupt"][m], label_smoothing=self.label_smoothing)
                          if m.any() else zero)
        m = tgt["m_da"] > 0
        teile["da"] = self._bce(out["da"][m], tgt["da"][m]).mean() if m.any() else zero
        m = tgt["m_zusatz"] > 0
        teile["zusatz"] = self._bce(out["zusatz"][m], tgt["zusatz"][m]).mean() if m.any() else zero
        gesamt = self.l_haupt * teile["haupt"] + self.l_da * teile["da"] + self.l_zusatz * teile["zusatz"]
        return gesamt, {k: float(v.detach()) for k, v in teile.items()}
