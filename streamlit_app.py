import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import re
import difflib
from db import save_upload
import os
import base64

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Rotten or Not 🍎", layout="wide")

# ---------------- VIDEO BACKGROUND ----------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

video_base64 = get_base64("fruit.mp4")

st.markdown(
    f"""
    <style>
    .stApp {{
        background: transparent;
    }}

    video {{
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        z-index: -1;
        object-fit: cover;
        filter: brightness(1.05);
    }}

    .content {{
        position: relative;
        z-index: 1;
    }}

    :root{{--glass-bg: rgba(255,255,255,0.04); --glass-border: rgba(255,255,255,0.06);}}

    .glass {{
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        box-shadow: 0 8px 32px rgba(2,6,23,0.6);
        backdrop-filter: blur(8px) saturate(140%);
        border-radius: 14px;
        padding: 18px;
    }}

    .big-title{{font-size:28px;font-weight:700;color:white}}
    .subtitle{{color:#9aa6b2}}

    </style>

    <video autoplay muted loop>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="content">', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best1.pt")

model = load_model()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.success("✅ Model loaded")
    auto = st.checkbox("Auto recipe select", True)
    conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.05)

# ---------------- TITLE ----------------
st.markdown("""
<div class='glass'>
<div class='big-title'>🍓 Fruit Freshness Detector</div>
<div class='subtitle'>Detect whether a fruit is fresh or rotten using YOLO</div>
</div>
""", unsafe_allow_html=True)

# ---------------- RECIPES ----------------
RECIPES = {
    "apple": {"title": "Apple Crumble", "content": "Bake apples with flour, butter & sugar."},
    "banana": {"title": "Banana Smoothie", "content": "Blend banana with milk & honey."},
    "mango": {"title": "Mango Salsa", "content": "Mix mango, onion & lime."},
    "orange": {"title": "Orange Granita", "content": "Freeze sweet orange juice."},
    "strawberry": {"title": "Strawberry Salad", "content": "Strawberry with spinach."},
    "cucumber": {"title": "Cucumber Raita", "content": "Mix cucumber with yogurt."}
}

# ---------------- FUNCTIONS ----------------
def extract_fruit_name(label):
    s = re.sub(r"\b(fresh|rotten)\b", "", label.lower()).strip()
    for k in RECIPES:
        if k in s:
            return k
    return s.split()[-1]

def auto_map_fruit(detected_info):
    items = sorted(detected_info, key=lambda x: x["conf"], reverse=True)
    for it in items:
        if it["conf"] < conf_thresh:
            continue
        name = extract_fruit_name(it["label"])
        if name in RECIPES:
            return name
    return None

# ---------------- UPLOADER ----------------
st.markdown("<div class='glass'><h3>📤 Upload Fruit Image</h3></div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    raw_bytes = uploaded_file.read()
    file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (640, 640))

    st.image(frame_rgb, caption="Uploaded Image", use_container_width=True)

    results = model.predict(frame_resized, conf=0.5, verbose=False)[0]

    detected_info = []

    if results.boxes:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = results.names[cls_id]

            detected_info.append({"label": label, "conf": conf})

            color = (0,255,0) if "fresh" in label.lower() else (255,0,0)

            cv2.rectangle(frame_resized,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame_resized,f"{label} {conf:.2f}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

        st.image(frame_resized, caption="Detection Result", use_container_width=True)

        # ---------------- RECIPE ----------------
        fruit = auto_map_fruit(detected_info) if auto else None

        if not fruit:
            fruit = st.selectbox("Select fruit", RECIPES.keys())

        if fruit in RECIPES:
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.subheader(RECIPES[fruit]["title"])
            st.write(RECIPES[fruit]["content"])
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- SAVE ----------------
        try:
            cloud_cfg = {
                "cloud_name": os.getenv("CLOUDINARY_CLOUD_NAME"),
                "api_key": os.getenv("CLOUDINARY_API_KEY"),
                "api_secret": os.getenv("CLOUDINARY_API_SECRET"),
            }

            save_res = save_upload(raw_bytes, uploaded_file.name, fruit, detected_info, cloudinary_config=cloud_cfg)

            st.caption(f"Saved to DB: {save_res.get('_id')}")
        except Exception as e:
            st.warning(f"DB save failed: {e}")

    else:
        st.warning("⚠️ No fruit detected.")

st.markdown("</div>", unsafe_allow_html=True)