from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
#import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image

model = YOLO('models/trained.pt')

class BoundingBox(BaseModel):
    x_min: int = Field(..., description="Левая координата", ge=0)
    y_min: int = Field(..., description="Верхняя координата", ge=0)
    x_max: int = Field(..., description="Правая координата", ge=0)
    y_max: int = Field(..., description="Нижняя координата", ge=0)


class Detection(BaseModel):
    bbox: BoundingBox = Field(..., description="Результат детекции")


class DetectionResponse(BaseModel):
    detections: List[Detection] = Field(..., description="Список найденных логотипов")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Дополнительная информация")


app = FastAPI(title="Распознавание логотипа Т-Банка")


@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png", "image/bmp", "image/webp"]:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    try:

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)

        #rgb to bgr
        if image_np.shape[-1] == 3:
            image_np = image_np[:, :, ::-1]

        detections = []


        if model is not None:
            results = model(image_np)

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # bounding box
                        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()

                        # filter
                        if confidence > 0.5:
                            detections.append(Detection(bbox=BoundingBox(
                                x_min=int(x_min),
                                y_min=int(y_min),
                                x_max=int(x_max),
                                y_max=int(y_max)
                            )))
        else:
            print("ошибка при загрузке модели")

        return DetectionResponse(detections=detections)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "OK", "message": "Service is running"}