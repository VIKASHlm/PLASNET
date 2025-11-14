import streamlit as st
import json
import tempfile
import cv2
from PIL import Image
import sys, os

# Ensure project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Imports
from plastiphalt_ai.classifiers.plastic_classifier import PlasticClassifier
from plastiphalt_ai.sorting.sorter import sort_plastics
from plastiphalt_ai.designer.optim_plastiphalt import suggest_mix

# Try importing CNN (optional if TensorFlow installed)
try:
    from tensorflow.keras.models import load_model
    import numpy as np
    USE_CNN = True
except ImportError:
    USE_CNN = False

st.title("♻️ PlasNet")

# -------------------- Inputs --------------------
video_file = st.file_uploader("Upload Conveyor Belt Video", type=["mp4", "avi", "mov"])

# Model choice dropdown
model_choice = st.selectbox("Choose Model", ["Random Forest", "CNN"])

if model_choice == "Random Forest":
    default_model = "D:\plastiphalt_ai_project\models\plastic_rf.joblib"
else:
    default_model = "D:\plastiphalt_ai_project\models\plastic_cnn.h5"

model_path = st.text_input("Classifier Model Path", value=default_model)

frame_skip = st.number_input("Process every Nth frame", min_value=1, value=30)
severity = st.selectbox("Severity", ["low", "medium", "high"])
traffic = st.number_input("Average Daily Traffic (vehicles/day)", value=25000)
temperature = st.number_input("Typical Ambient Temperature (°C)", value=35.0)

# Instead of JSON, ask user dynamically
plastics_available = {}
st.subheader("Available Plastics (kg)")
for plastic in ["HDPE", "PP", "LDPE", "PET", "PVC", "PS", "OTHER"]:
    plastics_available[plastic] = st.number_input(plastic, value=0)

bitumen = st.number_input("Batch Bitumen (kg)", value=1000.0)
agg_ratio = st.number_input("Aggregate-to-Bitumen Ratio", value=10.0)

# Container to display frame images
frame_display = st.empty()

if st.button("Run Pipeline"):

    if video_file is None:
        st.error("Please upload a video!")
    else:
        # Save video temporarily
        tmp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(tmp_video_path, "wb") as f:
            f.write(video_file.read())

        detections = []
        class_counts = {}

        cap = cv2.VideoCapture(tmp_video_path)
        frame_count = 0

        st.info(f"Processing video frames using {model_choice}...")

        # -------------------- Load model --------------------
        if model_choice == "Random Forest":
            clf = PlasticClassifier(model_path=model_path).load()
        else:
            if not USE_CNN:
                st.error("TensorFlow not installed. CNN not available.")
                st.stop()
            cnn_model = load_model(model_path)
            labels = ["HDPE", "PP", "LDPE", "PET", "PVC", "PS", "OTHER"]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip == 0:
                tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
                cv2.imwrite(tmp_path, frame)

                if model_choice == "Random Forest":
                    proba, classes = clf.predict_proba(tmp_path)
                    pred = clf.predict(tmp_path)
                    confidence = max(proba) if proba else 1.0
                else:
                    # CNN prediction
                    img = cv2.resize(frame, (128, 128))
                    img = img.astype("float32") / 255.0
                    img = np.expand_dims(img, axis=0)
                    preds = cnn_model.predict(img)
                    pred_idx = np.argmax(preds)
                    pred = labels[pred_idx]
                    confidence = float(np.max(preds))

                detection = {
                    "id": f"frame_{frame_count}",
                    "label": pred,
                    "confidence": confidence,
                }
                detections.append(detection)
                class_counts[pred] = class_counts.get(pred, 0) + 1

                # ----------------- Overlay label on frame -----------------
                label_text = f"{pred} ({confidence:.2f})"
                display_frame = frame.copy()
                cv2.putText(
                    display_frame,
                    label_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Convert BGR to RGB for PIL/Streamlit
                display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(display_frame_rgb)

                frame_display.image(img_pil, caption=f"Frame {frame_count}", use_column_width=True)

        cap.release()

        # -------------------- Aggregate --------------------
        total_detected = sum(class_counts.values())
        dynamic_allocation = {}

        if total_detected > 0:
            for label, count in class_counts.items():
                if plastics_available[label] > 0:
                    dynamic_allocation[label] = plastics_available[label] * (count / total_detected)
        else:
            dynamic_allocation = plastics_available

        sorted_list = sort_plastics(detections, severity=severity)

        plan = suggest_mix(
            severity,
            traffic,
            temperature,
            dynamic_allocation,
            batch_bitumen_kg=bitumen,
            aggregate_to_bitumen_ratio=agg_ratio,
        )

        # -------------------- Results --------------------
        st.subheader("Summary Counts")
        st.json(class_counts)

        st.subheader("Detections")
        st.json(detections)

        st.subheader("Sorted Plastics")
        st.json(sorted_list)

        st.subheader("Mix Design & Expected Durability")
        st.json(plan)
