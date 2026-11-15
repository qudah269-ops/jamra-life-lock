import numpy as np
from PIL import Image

import jamra_life_lock as jl
from jamra_life_lock.detect import decode, letterbox

PERSON = {"cls": "person", "id": 0, "score": .8, "x": 0, "y": 0, "w": 10, "h": 20}
DOG = {"cls": "dog", "id": 16, "score": .6, "x": 30, "y": 0, "w": 10, "h": 10}


def test_lock_rule():
    assert jl.decide([], 120)["allowed"]
    assert jl.decide([PERSON, DOG], 120) == {"allowed": False, "reason": "life", "people": 1, "animals": 1}
    assert jl.decide([], jl.BLIND_LUMA - 1)["reason"] == "too_dark"     # unseen is not empty


def test_dark_and_empty_photos():
    d = jl.Detector(416)
    assert d(Image.new("RGB", (640, 480), (5, 5, 5)))["reason"] == "too_dark"
    r = d(Image.new("RGB", (640, 480), (120, 140, 110)))
    assert r["allowed"] and r["boxes"] == []


def test_lock_holds_after_last_sighting():
    lock = jl.Lock(hold_s=2.5)
    seen = jl.decide([PERSON], 120)
    clear = jl.decide([], 120)
    assert not lock.update(seen, 10.0)["allowed"]
    assert lock.update(clear, 11.0)["reason"] == "hold"
    assert lock.update(clear, 13.0)["allowed"]


def test_only_life_classes_are_kept():
    out = np.zeros((1, 84, 2), dtype=np.float32)
    out[0, :4, 0] = [320, 320, 60, 120]; out[0, 4 + 0, 0] = 0.9       # person
    out[0, :4, 1] = [100, 100, 60, 60]; out[0, 4 + 2, 1] = 0.95       # car: not a living thing
    b = decode(out, 1.0, 0, 0, jl.LIFE, jl.CONF, 640, 640)
    assert [x["cls"] for x in b] == ["person"]


def test_models_output_80_coco_classes():
    for size in (640, 416):
        d = jl.Detector(size)
        x, *_ = letterbox(Image.new("RGB", (200, 200)), size)
        assert d.session.run(None, {d.input: x})[0].shape[1] == 84


def test_result_is_json_ready():
    import json
    out = np.zeros((1, 84, 1), dtype=np.float32)
    out[0, :4, 0] = [320, 320, 60, 120]; out[0, 4, 0] = 0.9
    json.dumps(decode(out, 1.0, 0, 0, jl.LIFE, jl.CONF, 640, 640))
