"""People and animals under the drone, and the drop lock built on them (JAMRA camera.html / detect.js).

The detector is Ultralytics' YOLOv8n trained on COCO, exported to ONNX (output 1×84×A). JAMRA keeps 11 of the 80 classes:
person and the ten animals COCO knows. The lock then says whether an extinguisher ball may be dropped.
"""
from __future__ import annotations

from importlib.resources import files

import numpy as np
from PIL import Image

# COCO ids of the living things a ball must not hit
LIFE = {0: "person", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe"}
CONF = 0.35          # JAMRA's person/animal threshold
BLIND_LUMA = 46      # mean brightness (0–255) under which "nobody seen" is not trusted
HOLD_S = 2.5         # keep the lock on this long after the last sighting (video)
IOU = 0.45
MAX_BOXES = 12


def letterbox(img: Image.Image, size: int):
    """RGB image -> (1×3×size×size float32, scale, pad_x, pad_y), the picture centred on black like the browser code."""
    img = img.convert("RGB")
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("RGB", (size, size))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, r, px, py


def nms(boxes: list[dict], iou: float = IOU, keep_max: int = MAX_BOXES) -> list[dict]:
    """Greedy non-maximum suppression within each class, best score first."""
    keep: list[dict] = []
    for c in sorted(boxes, key=lambda b: -b["score"]):
        def overlap(k):
            x1, y1 = max(k["x"], c["x"]), max(k["y"], c["y"])
            x2, y2 = min(k["x"] + k["w"], c["x"] + c["w"]), min(k["y"] + k["h"], c["y"] + c["h"])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            return inter / (k["w"] * k["h"] + c["w"] * c["h"] - inter + 1e-9)
        if not any(k["cls"] == c["cls"] and overlap(k) > iou for k in keep):
            keep.append(c)
        if len(keep) >= keep_max:
            break
    return keep


def decode(out: np.ndarray, r: float, px: int, py: int, classes, conf: float, width: int, height: int) -> list[dict]:
    """Model output 1×(4+k)×A -> boxes in photo pixels for the class ids in `classes` (dict id -> name)."""
    o = out[0]
    ids = np.array(sorted(classes))
    scores = o[4 + ids]                              # k×A
    best = scores.argmax(0)
    sc = scores[best, np.arange(o.shape[1])]
    boxes = []
    for a in np.nonzero(sc >= conf)[0]:
        cx, cy, w, h = (float(v) for v in o[:4, a])   # plain floats: results go to JSON
        x, y = (cx - w / 2 - px) / r, (cy - h / 2 - py) / r
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(width), x + w / r), min(float(height), y + h / r)
        if x1 <= x0 or y1 <= y0:
            continue
        cid = int(ids[best[a]])
        boxes.append({"cls": classes[cid], "id": cid, "score": round(float(sc[a]), 3),
                      "x": round(x0, 1), "y": round(y0, 1), "w": round(x1 - x0, 1), "h": round(y1 - y0, 1)})
    return nms(boxes)


def luma(img: Image.Image) -> float:
    """Mean brightness 0–255 (Rec. 601), the camera page's darkness check."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    return float((a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114).mean())


def decide(boxes: list[dict], brightness: float) -> dict:
    """The drop rule. Blocked if any person or animal is seen; blocked too when the picture is too dark, because a
    detector that sees nothing in the dark looks exactly like an empty scene. Allowed only when it can see and nobody is there."""
    people = sum(b["cls"] == "person" for b in boxes)
    animals = len(boxes) - people
    if boxes:
        return {"allowed": False, "reason": "life", "people": people, "animals": animals}
    if brightness < BLIND_LUMA:
        return {"allowed": False, "reason": "too_dark", "people": 0, "animals": 0}
    return {"allowed": True, "reason": "clear", "people": 0, "animals": 0}


class Detector:
    def __init__(self, size: int = 640, model: str | None = None):
        import onnxruntime as ort
        if size not in (640, 416) and model is None:
            raise ValueError("bundled models are 640 and 416")
        path = model or str(files("jamra_life_lock") / "models" / f"person_{size}.onnx")
        self.size = size
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, image, conf: float = CONF) -> dict:
        img = image if isinstance(image, Image.Image) else Image.open(image)
        x, r, px, py = letterbox(img, self.size)
        out = self.session.run(None, {self.input: x})[0]
        boxes = decode(out, r, px, py, LIFE, conf, img.width, img.height)
        return {"boxes": boxes, **decide(boxes, luma(img))}


class Lock:
    """For a video stream: the lock stays on HOLD_S seconds after the last person or animal, so one missed frame
    cannot open it."""

    def __init__(self, hold_s: float = HOLD_S):
        self.hold_s, self.seen_at = hold_s, None

    def update(self, result: dict, t: float) -> dict:
        if result["reason"] == "life":
            self.seen_at = t
        if result["allowed"] and self.seen_at is not None and t - self.seen_at < self.hold_s:
            return {**result, "allowed": False, "reason": "hold"}
        return result


_default: Detector | None = None


def check(image, conf: float = CONF) -> dict:
    """One photo -> {boxes, allowed, reason, people, animals}. Loads the 640 model once."""
    global _default
    if _default is None:
        _default = Detector()
    return _default(image, conf)


def main(argv=None):
    """Print the lock decision for each photo as JSON."""
    import argparse
    import json
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("photos", nargs="+")
    p.add_argument("--conf", type=float, default=CONF)
    p.add_argument("--size", type=int, default=640, choices=(640, 416))
    a = p.parse_args(argv)
    d = Detector(a.size)
    for f in a.photos:
        print(json.dumps({"photo": f, **d(f, a.conf)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
