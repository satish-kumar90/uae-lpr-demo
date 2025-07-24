from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2

from ultralytics import YOLO
from extractor import YOLOPlateNumberExtractor

import os

MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
PLATE_MODEL_PATH = os.path.join(MODEL_DIR, "lp_uae_detection.pt")
OCR_MODEL_PATH = os.path.join(MODEL_DIR, "lp_ocr_uae.pt")


# Load models once
plate_model = YOLO(PLATE_MODEL_PATH)
ocr_extractor = YOLOPlateNumberExtractor(OCR_MODEL_PATH, conf=0.25, device="cpu")

app = FastAPI()

from db_utils import insert_result

@app.post("/api/extract")
async def extract_plate(file: UploadFile = File(...)):
    np_arr = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return JSONResponse(content={"error": "Unable to decode image."}, status_code=400)

    result = YOLOPlateNumberExtractor.process_plate_image_from_array(img, plate_model, ocr_extractor)
    if result is None:
        return JSONResponse(content={"error": "Fatal error."}, status_code=422)

    img, final_number, area, number, others, angle, bboxes, obb_xywhr = result

    # Add region(s) to final_number string, just like before
    regions = [x for x in (others or []) if x and not x.startswith("AreaCode:")]
    final_number_str = f"{area} {number}"
    if regions:
        final_number_str = f"{final_number_str} {' '.join(regions)}"

    # Save to DB
    insert_result(
        image_name=file.filename,
        final_number=final_number_str.strip(),
        area=area,
        number=number,
        others=others,
        angle=angle
    )

    return {
        "final_number": final_number_str.strip(),
        "area": area,
        "number": number,
        "others": others,
        "angle": angle
    }



#uvicorn app:app --reload
