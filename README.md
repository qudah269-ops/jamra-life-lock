# jamra-life-lock — no extinguisher ball over a life

Team TFC · NASA Space Apps 2026 · part of [JAMRA](https://jamra.tfs.ink).

[العربية ↓](#العربية)

## What it does
JAMRA's drone drops extinguisher balls that burst and spread powder. Before any drop, this checks the camera frame and
answers one question: **may the drone drop here?**

- **Blocked** if it sees a **person** or any of the **ten animals** COCO knows (bird, cat, dog, horse, sheep, cow,
  elephant, bear, zebra, giraffe).
- **Blocked** if the picture is **too dark** to tell (mean brightness under 46 of 255): a detector that sees nothing in
  the dark looks exactly like an empty scene, so "nobody seen" is not trusted there.
- **Held** for 2.5 s after the last sighting on video (`Lock`), so one missed frame cannot open it.
- **Allowed** only when it can see and nobody is there.

In JAMRA it runs on the drone camera and the drone simulation, and it marks "lives in the photo" on citizen reports
for civil defence.

## The model, honestly
The detector is **Ultralytics' YOLOv8n trained on COCO**, exported to ONNX at 640 and 416 px (in
`src/jamra_life_lock/models/`). We did not retrain it. Our part is choosing the 11 living classes out of COCO's 80, the
thresholds (0.35, IoU 0.45), the darkness rule, the hold time, and wiring it into the drop decision.

Check on a street photo with people (`uv run jamra-life-lock photo.jpg`): 5 person boxes at 0.39 to 0.87, drop blocked.

## Install
```bash
uv sync
uv run pytest -q
```

## Minimal example
```python
import jamra_life_lock as jl

r = jl.check("frame.jpg")
print(r["allowed"], r["reason"], r["people"], r["animals"])   # reason: clear, life or too_dark

lock = jl.Lock()                                              # video: keep the lock on after a sighting
for t, frame in frames:                                       # your camera loop, t in seconds
    if lock.update(jl.check(frame), t)["allowed"]:
        drop()
```

## Honest limits
- **Seen from above is hard:** COCO photos are mostly taken from eye level. From a drone at 28 m a person is small and
  seen from the top, and recall there has **not been measured**. That is why darkness and the hold time lean toward
  blocking.
- Only the ten COCO animals: goats, foxes, deer, wild boar and others are not in COCO and may be missed.
- People under trees or in smoke can be hidden from the camera.

<div dir="rtl">

## العربية

# jamra-life-lock — ما في كرة إطفاء فوق روح

### شو بيعمل
درون جمرة بيرمي كرات إطفاء بتنفتح وبتنشر مسحوق. قبل أي إسقاط، هاد بيفحص صورة الكاميرا وبيجاوب: **مسموح نرمي هون؟**
- **ممنوع** إذا شاف **إنسان** أو أي حيوان من **العشرة** اللي بيعرفهم COCO.
- **ممنوع** إذا الصورة **عتمة كتير** (معدّل الإضاءة أقل من 46 من 255)، لأنه «ما شفت حدا» بالعتمة مش معناه «ما في حدا».
- **بيضل ممنوع** 2.5 ثانية بعد آخر مرة شاف حدا بالفيديو، فلقطة وحدة فاتها ما بتفتح القفل.
- **مسموح** بس لما يقدر يشوف وما في حدا.

### الموديل بصراحة
الكاشف هو **YOLOv8n من Ultralytics متدرّب على COCO**، وما أعدنا تدريبه. شغلنا: اختيار 11 صنف حي من 80،
والعتبات، وقاعدة العتمة، ومدة الحجز، وربطه بقرار الإسقاط.

### حدود بصراحة
- **الشوف من فوق صعب:** صور COCO أغلبها من مستوى العين. من درون على ارتفاع 28 م الإنسان صغير ومشوف من فوق، والاسترجاع هناك **ما انقاس**.
- بس الحيوانات العشرة تبع COCO: الماعز والثعالب والغزلان والخنازير البرية مش فيه.
- الناس تحت الشجر أو بالدخان ممكن ما يبيّنوا.

</div>

## License
AGPL-3.0, see [LICENSE](LICENSE): the weights are Ultralytics YOLOv8n, which is AGPL-3.0, so this repository is too.
