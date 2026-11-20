"""
Experiment E4: Lernkurve. Wie viele echte Bilder je Zusatzzeichen braucht das auf Synset trainierte
Multi-Task-Modell (Architektur A)?

Für jede Stufe (0 / 25 / 50 / 100 / 150 / alle echten Trainingsausschnitte je Zusatzzeichen) wird A
neu trainiert (train_multitask.py --echte-je-klasse N) und auf dem echten Test (Aachen, Eifel, Köln)
ausgewertet. Fertige Stufen werden übersprungen (ergebnis.json vorhanden), ein abgebrochener Lauf
auf Colab kann also einfach neu gestartet werden.

  python experiment_e4.py --config configs/colab_multitask.yaml --ziel results/e4
  python experiment_e4.py --smoke            # kurzer Probelauf auf der CPU
"""

import argparse
import json
import os
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STUFEN = ["0", "25", "50", "100", "150", "alle"]
METRIKEN = [("zusatz_exakt", "Zusatzzeichen exakt"),
            ("zielgruppe_korrekt_relevant", "Zielgruppe korrekt"),
            ("zusatz_macro_f1", "Zusatz Makro-F1"),
            ("haupt_acc", "Hauptschild")]


def lauf(stufe, args):
    ordner = os.path.join(args.ziel, f"n_{stufe}")
    ergebnis = os.path.join(ordner, "ergebnis.json")
    if os.path.exists(ergebnis) and not args.neu:
        print(f"Stufe {stufe}: schon vorhanden, übersprungen")
    else:
        cmd = [sys.executable, "train_multitask.py", "--config", args.config, "--ausgabe", ordner,
               "--epochen", str(args.epochen)]
        if stufe != "alle":
            cmd += ["--echte-je-klasse", stufe]
        if args.smoke:
            cmd.append("--smoke")
        print(f"\n=== Stufe {stufe}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
    with open(ergebnis, encoding="utf-8") as f:
        return json.load(f)


def zeichnen(tabelle, pfad):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = list(range(len(tabelle)))
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, name in METRIKEN:
        y = [z["echt_test"].get(key) for z in tabelle]
        ax.plot(x, [v if v is not None else float("nan") for v in y], marker="o", label=name)
    ax.set_xticks(x, [f"{z['stufe']}\n({z['n_echt_train']})" for z in tabelle])
    ax.set_xlabel("echte Trainingsbilder je Zusatzzeichen (Ausschnitte gesamt)")
    ax.set_ylabel("echter Test (Aachen, Eifel, Köln)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(pfad, dpi=150)


def main(args):
    os.makedirs(args.ziel, exist_ok=True)
    tabelle = []
    for stufe in args.stufen:
        e = lauf(stufe, args)
        tabelle.append({"stufe": stufe, "n_echt_train": e.get("n_echt_train"), "echt_test": e["echt_test"]})
    with open(os.path.join(args.ziel, "lernkurve.json"), "w", encoding="utf-8") as f:
        json.dump(tabelle, f, ensure_ascii=False, indent=1)
    zeichnen(tabelle, os.path.join(args.ziel, "lernkurve.png"))
    print("\nE4 Lernkurve (echter Test)")
    print(f"{'Stufe':>6} {'n':>5} " + " ".join(f"{name:>20}" for _, name in METRIKEN))
    for z in tabelle:
        werte = [z["echt_test"].get(k) for k, _ in METRIKEN]
        print(f"{z['stufe']:>6} {z['n_echt_train'] or 0:>5} "
              + " ".join(f"{v:>20.3f}" if v is not None else f"{'-':>20}" for v in werte))
    print(f"Ergebnis: {os.path.join(args.ziel, 'lernkurve.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Experiment E4: Lernkurve über die Zahl echter Bilder")
    p.add_argument("--config", default=os.path.join("configs", "multitask.yaml"))
    p.add_argument("--ziel", default=os.path.join("results", "e4"))
    p.add_argument("--stufen", nargs="+", default=STUFEN)
    p.add_argument("--epochen", type=int, default=8, help="je Stufe (A selbst: 15)")
    p.add_argument("--neu", action="store_true", help="vorhandene Stufen neu trainieren")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.ziel = a.ziel + "_smoke"
        a.stufen = ["0", "25"]
    main(a)
