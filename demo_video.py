"""
Live-Demo: Verkehrszeichenerkennung in Video oder Webcam (CPU, ONNX Runtime)

Pipeline pro Frame:
  1. Kandidaten finden: HSV-Farbmaske (rot/blau) + Konturen  (klassischer Baseline-Detektor)
  2. Klassifizieren: Ausschnitt -> preprocess_for_inference() -> ONNX-Modell
  3. Glätten: einfaches IoU-Tracking + Mehrheitsentscheid über die letzten N Frames

Beispiele:
  python demo_video.py --source mein_video.mp4
  python demo_video.py --source 0                       # Webcam
  python demo_video.py --source mein_video.mp4 --save results/demo.mp4 --no-window
"""

import argparse
import time
from collections import Counter, deque

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from src.data.preprocessing import preprocess_for_inference
from src.utils.class_names import get_class_name

# cv2.putText kann keine Umlaute darstellen
_UMLAUTE = str.maketrans({'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue', 'ß': 'ss'})


def label_text(class_id: int) -> str:
    return get_class_name(class_id, 'de').translate(_UMLAUTE)


def candidate_boxes(frame: np.ndarray, min_area: int):
    """HSV-Baseline: rote und blaue, ungefähr quadratische Flächen als Schildkandidaten."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)),
        cv2.inRange(hsv, (170, 70, 50), (180, 255, 255)),
    )
    blue = cv2.inRange(hsv, (100, 100, 50), (130, 255, 255))
    mask = cv2.morphologyEx(cv2.bitwise_or(red, blue), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = frame.shape[:2]
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if not 0.75 <= w / float(h) <= 1.3:
            continue
        # GTSRB-Bilder enthalten ca. 10 % Rand um das Schild -> gleiches Padding
        pad = int(0.1 * max(w, h))
        boxes.append((max(0, x - pad), max(0, y - pad), min(w_img, x + w + pad), min(h_img, y + h + pad)))
    return boxes


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


class Track:
    """Ein Schild über mehrere Frames: sammelt Vorhersagen für den Mehrheitsentscheid."""

    def __init__(self, box, history: int):
        self.box = box
        self.votes = deque(maxlen=history)
        self.missed = 0

    def decision(self, min_votes: int):
        if not self.votes:
            return None
        class_id, count = Counter(c for c, _ in self.votes).most_common(1)[0]
        if count < min_votes:
            return None
        conf = np.mean([p for c, p in self.votes if c == class_id])
        return class_id, conf


def main():
    parser = argparse.ArgumentParser(description="Live-Verkehrszeichenerkennung mit ONNX")
    parser.add_argument("--source", default="0", help="Videodatei oder Kamera-Index (z. B. 0)")
    parser.add_argument("--model", default="models/onnx/gtsrb_model.onnx")
    parser.add_argument("--config", default="configs/config.yaml")
    # Mit label_smoothing=0.1 liegt die Konfidenz eines sicheren Treffers bei ca. 0.9,
    # eine Schwelle von 0.85 würde viele richtige Treffer verwerfen.
    parser.add_argument("--conf", type=float, default=0.5, help="Mindest-Softmax-Wahrscheinlichkeit")
    parser.add_argument("--min-area", type=int, default=600, help="Mindestfläche einer Kontur in Pixeln")
    parser.add_argument("--smooth", type=int, default=5, help="Frames im Mehrheitsentscheid (1 = aus)")
    parser.add_argument("--width", type=int, default=960, help="Frames auf diese Breite skalieren")
    parser.add_argument("--save", default=None, help="Ergebnisvideo speichern (z. B. results/demo.mp4)")
    parser.add_argument("--no-window", action="store_true", help="Kein Fenster (z. B. nur --save)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    size = config["data"]["image_size"]
    apply_clahe = config["augmentation"]["apply_clahe"]

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Videoquelle '{args.source}' konnte nicht geöffnet werden.")

    writer = None
    tracks: list[Track] = []
    min_votes = max(1, args.smooth // 2 + 1)
    fps_avg = 0.0
    print("Starte Erkennung. 'q' beendet.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()

        scale = args.width / frame.shape[1]
        frame = cv2.resize(frame, (args.width, int(frame.shape[0] * scale)))

        # 1 + 2: Kandidaten finden und im Batch klassifizieren
        boxes = candidate_boxes(frame, args.min_area)
        predictions = []
        if boxes:
            batch = np.concatenate([
                preprocess_for_inference(frame[y1:y2, x1:x2], (size, size), apply_clahe)
                for x1, y1, x2, y2 in boxes
            ])
            probs = softmax(session.run(None, {input_name: batch})[0])
            predictions = [(int(p.argmax()), float(p.max())) for p in probs]

        # 3: Detektionen den bestehenden Tracks zuordnen (IoU), Stimmen sammeln
        for track in tracks:
            track.missed += 1
        for box, (class_id, conf) in zip(boxes, predictions):
            best = max(tracks, key=lambda t: iou(t.box, box), default=None)
            if best is None or iou(best.box, box) < 0.3:
                best = Track(box, args.smooth)
                tracks.append(best)
            best.box, best.missed = box, 0
            if conf >= args.conf:
                best.votes.append((class_id, conf))
        tracks = [t for t in tracks if t.missed <= args.smooth]

        # Zeichnen
        for track in tracks:
            if track.missed:
                continue
            decision = track.decision(min_votes if args.smooth > 1 else 1)
            if decision is None:
                continue
            class_id, conf = decision
            x1, y1, x2, y2 = track.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_text(class_id)} ({conf * 100:.0f}%)", (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        fps_avg = fps if fps_avg == 0 else 0.9 * fps_avg + 0.1 * fps
        cv2.putText(frame, f"{fps_avg:.0f} FPS | {len(boxes)} Kandidaten", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if args.save:
            if writer is None:
                fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_src,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        if not args.no_window:
            cv2.imshow("GTSRB Live-Erkennung", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Video gespeichert: {args.save}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
