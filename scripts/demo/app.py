"""
Demonstrator für Architektur A: Foto eines Schildmasts -> Hauptschild, Zusatzzeichen und für wen die Regel gilt.

Läuft lokal auf der CPU im Browser (Gradio). Das Modell ist auf Ausschnitte trainiert: Im Bild den
Schildmast mit dem Zuschneide-Werkzeug markieren, dann „Erkennen“.

  python scripts/demo/app.py --modell models/multitask/modell.pth
  python scripts/demo/app.py --modell ... --beispiele 12      # Beispiele aus dem echten Testsatz
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # Projektordner (src/)
import argparse
import os

import gradio as gr
import numpy as np

from src.demo.vorhersage import Demonstrator


def prozent(p):
    return f"{100 * p:.0f} %"


def als_text(e: dict) -> str:
    r = e["regel"]
    z = [f"- Hauptschild: **{e['haupt'][0][0]}** ({prozent(e['haupt'][0][1])})",
         "  - Alternativen: " + ", ".join(f"{n} ({prozent(p)})" for n, p in e["haupt"][1:]),
         f"- Zusatzschild vorhanden: {'ja' if e['zusatz_vorhanden'] >= 0.5 else 'nein'} "
         f"({prozent(e['zusatz_vorhanden'])})",
         "- Zusatzzeichen: " + (", ".join(f"**{k}** ({prozent(p)})" for k, p in e["zusatz"]) or "keins")]
    if e["zusatz_knapp"]:
        z.append("  - knapp unter der Schwelle: " + ", ".join(f"{k} ({prozent(p)})" for k, p in e["zusatz_knapp"]))
    z += ["", "### Regel (StVO-Tabelle)",
          f"- **gilt für:** {', '.join(r['gilt_fuer'])}",
          f"- **ausgenommen:** {', '.join(r['ausgenommen']) or '–'}",
          f"- **nicht betroffen:** {', '.join(r['nicht_betroffen']) or '–'}"]
    z += [f"- Bedingung: {b}" for b in r["bedingungen"]] + [f"- Hinweis: {h}" for h in r["hinweise"]]
    if r["unklar"]:
        z.append(f"- ⚠ unklar: {', '.join(r['unklar'])}")
    if e["haupt"][0][1] < 0.5 or e["zusatz_knapp"]:
        z.append("\n⚠ Das Modell ist sich unsicher, Ergebnis mit Vorsicht lesen.")
    return "\n".join(z)


def beispiel_pfade(n: int) -> list:
    """Echte Testausschnitte (Aachen, Eifel, Köln) mit Zusatzzeichen, für die Beispielleiste."""
    if n <= 0:
        return []
    from src.data.crop_records import build_records
    recs = [r for r in build_records() if r["split"] == "test" and r["klassen"] != ["keins"]]
    schritt = max(1, len(recs) // n)
    return [r["crop"] for r in recs[::schritt][:n] if os.path.exists(r["crop"])]


def baue_app(demo: Demonstrator, beispiele: list) -> gr.Blocks:
    def erkennen(bild, gradcam):
        if bild is None:
            return "Bitte ein Bild laden.", None
        rgb = bild["composite"] if isinstance(bild, dict) else bild
        if rgb is None:
            return "Bitte ein Bild laden.", None
        rgb = np.asarray(rgb)[..., :3].astype(np.uint8)
        e = demo.vorhersage(rgb)
        return als_text(e), demo.grad_cam(rgb, "zusatz" if e["zusatz"] else "haupt") if gradcam else None

    with gr.Blocks(title="Schildkombinationen") as app:
        gr.Markdown("## Kontexterkennung von Schildkombinationen\n"
                    "Foto laden, den Schildmast mit dem Zuschneide-Werkzeug markieren, dann **Erkennen**. "
                    f"Modell A (ResNet-18, Epoche {demo.epoche}).")
        with gr.Row():
            with gr.Column():
                bild = gr.ImageEditor(type="numpy", label="Foto / Ausschnitt", transforms=("crop",),
                                      brush=False, eraser=False, height=420)
                gradcam = gr.Checkbox(value=True, label="Grad-CAM anzeigen (worauf das Netz schaut)")
                knopf = gr.Button("Erkennen", variant="primary")
                if beispiele:
                    gr.Examples(beispiele, inputs=bild, label="Beispiele aus dem echten Testsatz")
            with gr.Column():
                ausgabe = gr.Markdown()
                heatmap = gr.Image(label="Grad-CAM", height=300)
        knopf.click(erkennen, [bild, gradcam], [ausgabe, heatmap])
    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Demonstrator Architektur A")
    p.add_argument("--modell", default=os.path.join("models", "multitask", "modell.pth"))
    p.add_argument("--beispiele", type=int, default=12)
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args()
    baue_app(Demonstrator(a.modell), beispiel_pfade(a.beispiele)).launch(server_port=a.port)
