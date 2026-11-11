from fastapi import FastAPI
import io
from collections import Counter
from datetime import datetime
from typing import Literal
import cv2
import numpy as np
from fastapi import File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI(title="Termômetro de humor", )

@app.get("/")
def root():
    return {"message": "API no ar!"}

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

classifier = pipeline("image-classification", model="trpakov/vit-face-expression")

MAX_FACES = 5

TRANSLATION = {
    "angry": "raiva",
    "disgust": "nojo",
    "fear": "medo",
    "happy": "feliz",
    "neutral": "neutro",
    "sad": "triste",
    "surprise": "surpreso"
}

Emotion = Literal["raiva", "nojo", "medo", "feliz", "neutro", "triste", "surpreso"]

Polarity = Literal["positivo", "negativo", "neutro"]

POLARITY: dict[str,str] = {
    "raiva": "negativo",
    "nojo": "negativo",
    "medo": "negativo",
    "feliz": "positivo",
    "neutro": "neutro",
    "triste": "negativo",
    "surpreso": "positivo"
}

class Face(BaseModel):
    emotion: Emotion 
    trust: float
    polarity: Polarity
    box: tuple[int, int, int, int]

class Analysis(BaseModel):
    found_faces: bool
    total_faces: int
    faces: list[Face]

class InitialAssessment(BaseModel):
    emotion: Emotion
    trust: float = Field(ge=0, le=1)

class Assessment(InitialAssessment):
    id: int
    polarity: Polarity
    date_time: datetime

assessments: list[Assessment] = []

def save(data: InitialAssessment) -> Assessment:
    new = Assessment(id=len(assessments) + 1,
        polarity=POLARITY[data.emotion],
        date_time=datetime.now(),
        **data.model_dump(),
    )
    assessments.append(new)
    return new

@app.post("/analyze", response_model=Analysis)
def analyze(image: UploadFile = File(...), register: bool = False):
    try:
        photo = Image.open(io.BytesIO(image.file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    gray = cv2.cvtColor(np.array(photo), cv2.COLOR_RGB2GRAY)
    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    boxes = sorted(boxes, key=lambda c: c[2] * c[3], reverse=True) [:MAX_FACES]

    if len(boxes) == 0:
        return Analysis(found_faces=False, total_faces=0, faces=[])

    boxes = [tuple(int(v) for v in c) for c in boxes]
    excerpts = [photo.crop((x, y, x + w, y + h)) for x, y, w, h in boxes]
    forecast = classifier(excerpts)

    faces = []

    for box, forecast in zip(boxes, forecast):
        best = forecast[0]
        emotion = TRANSLATION[best["label"].lower()]
        trust = float(best["score"])
        faces.append(Face(emotion = emotion, trust=trust, polarity=POLARITY[emotion], box=box))

        if register:
            save(InitialAssessment(emotion=emotion, trust=trust))
    return Analysis(found_faces=True, total_faces=len(boxes), faces=faces)

@app.post("/assessments", response_model=Assessment, status_code=201)
def create_assessment(data: InitialAssessment):
    return save(data)

@app.get("/assessments", response_model=list[Assessment])
def list(polarity: Polarity | None = None):
    if polarity:
        return [a for a in assessments if a.polarity == polarity]
    return assessments

@app.get("/assessments/{assessment_id}", response_model=Assessment)
def get(assessment_id: int):
    for a in assessments:
        if a.id == assessment_id:
            return a
    raise HTTPException(status_code=404, detail="Assessment not found")

@app.get("/resume")
def resume():
    return {"total": len(assessments), 
    "polarity": dict(Counter(a.polarity for a in assessments)),
    "emotion": dict(Counter(a.emotion for a in assessments))
    }

@app.delete("/assessments", status_code=204)
def clean():
    assessments.clear()
    return