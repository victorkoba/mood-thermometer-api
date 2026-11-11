# Client - Class Thermometer: captures image and shows API analysis
# Run: python -m streamlit run app.py  (with the API running)
import base64
import hashlib
import io
from pathlib import Path

import cv2
import pandas as pd
import requests
import streamlit as st
from PIL import Image

API = "http://127.0.0.1:8000"
EXAMPLES_DIR = Path(__file__).parent / "img"
INTERVAL = 2  # seconds between analyses in live mode

# Room cameras: replace with real IPs
CAMERAS = {
    "Professor": "http://192.168.0.10:8001/frame",
    "Bancada 1": "http://192.168.0.11:8001/frame",
    "Celular 1": "http://192.168.0.21:8080/shot.jpg",
}

TRANSLATION = {
    "angry": "raiva",
    "disgust": "nojo",
    "fear": "medo",
    "happy": "feliz",
    "neutral": "neutro",
    "sad": "triste",
    "surprise": "surpreso",
}

# Emojis by Unicode code: \U + 8 hex digits
EMOJIS = {
    "feliz": "\U0001F604",
    "surpresa": "\U0001F632",
    "surpreso": "\U0001F632",
    "neutro": "\U0001F610",
    "triste": "\U0001F622",
    "raiva": "\U0001F620",
    "medo": "\U0001F628",
    "nojo": "\U0001F922",
}

# Image style: face border and emoji "jump" animation
CSS = """
<style>
.frame { position: relative; display: inline-block; }
.frame img { max-width: 100%; border-radius: 8px; }
.face { position: absolute; border: 3px solid #F2C46D; border-radius: 6px; }
.emoji { position: absolute; font-size: 64px; transform: translate(-50%, -50%);
         animation: jump 0.6s ease-out; }
@keyframes jump {
  0%   { transform: translate(-50%, -50%) scale(0); }
  60%  { transform: translate(-50%, -110%) scale(1.3); }
  100% { transform: translate(-50%, -50%) scale(1); }
}
</style>
"""


# ---------- API communication ----------
# Converts to RGB and resizes the image
def prepare(image):
    image = image.convert("RGB")
    image.thumbnail((640, 640))
    return image


# Converts image to JPEG bytes
def to_jpeg(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


# Sends the image to the API and returns the JSON response
def analyze(image, register=True):
    r = requests.post(
        f"{API}/analyze",
        params={"register": str(register).lower()},
        files={"image": ("frame.jpg", to_jpeg(image), "image/jpeg")},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


# Image on the left, API JSON on the right
def display(image, response, caption):
    left, right = st.columns([3, 2])
    with left:
        show(image, response)
        st.caption(caption)
    with right:
        st.markdown("**Resposta da API**")
        st.json(response)


# ---------- Display ----------
# Draws border and emoji over each face
def show(image, response):
    photo = base64.b64encode(to_jpeg(image)).decode()
    width, height = image.size
    pct = lambda value, total: f"{100 * value / total:.1f}%"  # position in % of image
    overlays = ""
    for face in response["faces"]:
        x, y, w, h = face["box"]
        overlays += (
            f'<div class="face" style="left:{pct(x, width)}; top:{pct(y, height)}; '
            f'width:{pct(w, width)}; height:{pct(h, height)}"></div>'
            f'<div class="emoji" style="left:{pct(x + w, width)}; top:{pct(y, height)}">'
            f'{EMOJIS[face["emotion"]]}</div>'
        )
    st.html(CSS + f'<div class="frame"><img src="data:image/jpeg;base64,{photo}">{overlays}</div>')


# E.g.: "2 rostos: feliz 92%, neutro 70%"
def describe(response):
    if not response["found_faces"]:
        return "Nenhum rosto encontrado. Tente com mais luz e de frente para a câmera."
    parts = [f"{f['emotion']} {f['trust']:.0%}" for f in response["faces"]]
    n = response["total_faces"]
    return f"{n} rosto{'s' if n > 1 else ''}: " + ", ".join(parts)


# ---------- Live mode ----------
# Reads the latest frame from the webcam
def webcam_frame():
    if "webcam" not in st.session_state:
        st.session_state.webcam = cv2.VideoCapture(0)
    cam = st.session_state.webcam
    for _ in range(4):  # discards old frames left in the buffer
        cam.grab()
    ok, frame = cam.retrieve()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ok else None


# Requests a frame from a room camera
def room_frame(url):
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


# Closes the webcam to release the device
def release_webcam():
    cam = st.session_state.pop("webcam", None)
    if cam is not None:
        cam.release()


@st.fragment(run_every=INTERVAL)  # runs automatically every 2 seconds
def live_mode(camera):
    try:
        if camera == "Webcam deste computador":
            image = webcam_frame()
        else:
            image = room_frame(CAMERAS[camera])
    except requests.RequestException as error:
        st.error(f"Não foi possível acessar a câmera '{camera}': {error}")
        return
    if image is None:
        st.error("Não consegui ler a webcam. Ela está conectada e livre?")
        return

    image = prepare(image)
    try:
        response = analyze(image)  # API analyzes and registers the assessment
    except requests.RequestException as error:
        st.error(f"Falha ao falar com a API: {error}")
        return
    caption = describe(response)
    if response["found_faces"]:
        caption += " · registrado na API"
    display(image, response, caption)


# ---------- Photo mode ----------
# Returns (image, expected emotion or None)
def get_photo():
    source = st.radio(
        "Fonte da imagem",
        ["Câmera do computador", "Galeria de exemplos", "Upload"],
        horizontal=True,
    )

    if source == "Câmera do computador":
        photo = st.camera_input("Tire uma foto")
        return (Image.open(photo), None) if photo else (None, None)

    if source == "Galeria de exemplos":
        files = sorted(
            p for p in EXAMPLES_DIR.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not files:
            st.info("A pasta img/ está vazia. Rode antes: python baixar_exemplos.py")
            return None, None
        file = st.selectbox("Exemplo", files, format_func=lambda p: p.name)
        raw_expected = file.stem.split("_")[0]  # e.g.: happy_01.jpg -> happy -> feliz
        expected = TRANSLATION.get(raw_expected, raw_expected)
        return Image.open(file), (expected if expected in EMOJIS else None)

    uploaded = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])
    return (Image.open(uploaded), None) if uploaded else (None, None)


# ---------- Summary (auto-updates every 5 seconds) ----------
@st.fragment(run_every=5)
def class_summary():
    st.subheader("Resumo da turma")
    try:
        summary = requests.get(f"{API}/resume", timeout=5).json()
    except requests.ConnectionError:
        st.error("API fora do ar. Rode antes: uvicorn api:app --reload")
        return
    polarity = summary["polarity"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", summary["total"])
    c2.metric("Positivo", polarity.get("positivo", 0))
    c3.metric("Neutro", polarity.get("neutro", 0))
    c4.metric("Negativo", polarity.get("negativo", 0))
    if summary["emotion"]:
        st.bar_chart(pd.Series(summary["emotion"], name="avaliações"))


# ---------- Page ----------
st.title("\U0001F60A Termômetro da Aula")
mode = st.radio("Modo", ["Ao vivo", "Foto"], horizontal=True)

if mode == "Ao vivo":
    camera = st.selectbox("Câmera", ["Webcam deste computador", *CAMERAS])
    if camera != "Webcam deste computador":
        release_webcam()
    if st.toggle(f"Analisar a cada {INTERVAL} segundos"):
        live_mode(camera)
    else:
        release_webcam()
else:
    release_webcam()
    image, expected = get_photo()
    if image is not None:
        image = prepare(image)
        # Registers each photo only once
        key = hashlib.md5(to_jpeg(image)).hexdigest()
        is_new = key != st.session_state.get("last_photo")
        try:
            response = analyze(image, register=is_new)
        except requests.RequestException as error:
            st.error(f"Falha ao falar com a API: {error}")
            st.stop()
        st.session_state.last_photo = key

        caption = describe(response)
        if response["found_faces"]:
            caption += " · registrado na API" if is_new else " · foto já registrada"
            if expected:  # in gallery, compares with the largest face in the photo
                match_icon = "\u2705" if expected == response["faces"][0]["emotion"] else "\u274C"
                caption += f" · {match_icon} esperado: {expected}"
        display(image, response, caption)

st.divider()
class_summary()
