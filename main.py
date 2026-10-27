
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

app = FastAPI(title="Onion Quality Assessment")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(
    "/home/death/onion-ml/runs/onion_grading/weights/best.pt"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))


@app.get("/")
def home():
    return {"message": "Onion Quality API is running"}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image must be smaller than 10 MB.")

    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "Invalid image file.")

    # Run the trained model on the uploaded image.
    results = model.predict(image, conf=0.25, verbose=False)
    result = results[0]

    detections = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        detections.append({
            "class_id": class_id,
            "class_name": model.names[class_id],
            "confidence": round(float(box.conf.item()), 4),
            "box": [
                round(float(v), 2)
                for v in box.xyxy[0].tolist()
            ],
        })

    return {
        "filename": file.filename,
        "image_width": image.width,
        "image_height": image.height,
        "total_detections": len(detections),
        "detections": detections,
        "message": "AI prediction completed",
    }
