import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

sys.path.insert(0, '/app/ultralytics/yolov5')
from models.experimental import attempt_load  # noqa: E402
from models.common import AutoShape  # noqa: E402

model = None
gen_time = None

MODEL_WEIGHTS = os.environ.get("YOLO_MODEL", "yolov5m.pt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, gen_time
    print(f"Loading '{MODEL_WEIGHTS}' from local files...")
    model = AutoShape(attempt_load(MODEL_WEIGHTS, device='cpu'))
    gen_time = time.monotonic()
    print("Model loaded successfully.")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def get_status():
    return {"success": True}


@app.post("/detect")
async def handle_image_upload(image: UploadFile = File(...)):
    try:
        start_time = time.monotonic()

        contents = await image.read()
        img_pil = Image.open(__import__('io').BytesIO(contents))
        frame = np.array(img_pil)

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        results = model(frame)
        end_time = time.monotonic()
        processing_time_s = round((end_time - start_time), 2)
        total_time_s = round((end_time - gen_time), 2)
        warm = total_time_s > 30

        return {
            "success": True,
            "warm": warm,
            "processing_time_s": processing_time_s,
            "total_time_s": total_time_s,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/terminate")
@app.post("/terminate")
def terminate():
    os._exit(0)
