import streamlit as st
import tempfile
import cv2
from PIL import Image
import time
from ultralytics import YOLO

# Load all models once and map them
@st.cache_resource
def load_models():
    return {
        "Nano (Eco Mode)": YOLO("models/Nano/weights/best.pt"),
        "Small": YOLO("models/Small/weights/best.pt"),
        "Medium": YOLO("models/Medium/weights/best.pt")
    }

model_map = load_models()

# Streamlit page configuration
st.set_page_config(page_title="📼 Video Detection", layout="wide")
st.title("📼 Video Detection Dashboard")

# Sidebar selections
model_name = st.sidebar.selectbox("Select YOLOv8 Model", list(model_map.keys()))
model = model_map[model_name]

# Upload video
uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)

    stframe = st.empty()
    stop_btn = st.button("Stop Detection")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_btn:
            break

        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_container_width=True)

    cap.release()
    st.success("Video Detection Finished")
