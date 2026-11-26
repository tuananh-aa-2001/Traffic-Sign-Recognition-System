"""
Kennzahlen des Zusatz-Kopfs (multi-label) und der daraus abgeleiteten Zielgruppe.

pred, true: 0/1-Matrizen (Bilder x Klassen); eine Zeile ohne 1 bedeutet 'keins'.
gefahr: je Bild, ob das Hauptschild ein Gefahrzeichen ist (ändert die Wirkung mancher Zusatzzeichen).
"""

import numpy as np

from src.utils.zielgruppen import regel, zielgruppe_gleich


def zusatz_kennzahlen(pred, true, gefahr, klassen, trainierbar=None):
    """trainierbar: Klassen mit echten Trainingsbildern (nur Synset liefert kein Zusatz-Label);
    None = Kennzahl zusatz_macro_f1_trainierbar entfällt."""
    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    n = len(true)
    if n == 0:
        return {"zusatz_exakt": None, "n_zusatz": 0, "zielgruppe_korrekt": None,
                "zielgruppe_korrekt_relevant": None, "n_zielgruppe_relevant": 0}
    namen = lambda row: [klassen[j] for j in np.flatnonzero(row)] or ["keins"]
    c_zg = n_rel = c_rel = 0
    for p_row, t_row, g in zip(pred, true, gefahr):
        r_true = regel("", namen(t_row), bool(g))
        ok = zielgruppe_gleich(regel("", namen(p_row), bool(g)), r_true)
        c_zg += ok
        if r_true["gilt_fuer"] != ["alle"] or r_true["ausgenommen"]:   # Zusatz ändert die Zielgruppe
            n_rel += 1
            c_rel += ok
    tp, fp, fn = (pred * true).sum(0), (pred * (1 - true)).sum(0), ((1 - pred) * true).sum(0)
    f1 = np.where(tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    support = tp + fn
    return {"zusatz_exakt": float((pred == true).all(axis=1).mean()), "n_zusatz": n,
            # gleiche Wirkung auf Zielgruppen (gilt nur für / ausgenommen) wie das echte Label
            "zielgruppe_korrekt": c_zg / n,
            "zielgruppe_korrekt_relevant": c_rel / n_rel if n_rel else None,
            "n_zielgruppe_relevant": n_rel,
            "zusatz_micro_f1": float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)),
            "zusatz_macro_f1": float(np.nanmean(f1[support > 0])) if (support > 0).any() else None,
            # Makro-F1 nur über Klassen, die das Modell aus echten Trainingsbildern lernen konnte
            "zusatz_macro_f1_trainierbar": (float(np.nanmean(f1[maske])) if (maske := (support > 0) & np.array(
                [k in trainierbar for k in klassen])).any() else None) if trainierbar is not None else None,
            "zusatz_f1_je_klasse": {klassen[i]: {"f1": round(float(f1[i]), 3), "anzahl": int(support[i])}
                                    for i in range(len(klassen)) if support[i] > 0}}
