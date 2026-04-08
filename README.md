# 🛣️ PLASNET — AI-Driven Sustainable Road Repair System

> An end-to-end computer vision pipeline that detects road potholes and classifies plastic waste for use as recycled filler material — enabling smarter, greener, and more durable road repair.

---

## 🔍 Problem Statement

India loses an estimated **₹87,000 crore annually** due to poor road conditions. At the same time, plastic waste pollution is a growing crisis with millions of tonnes going unprocessed each year. PLASNET bridges both problems: using AI to detect damaged road sections and classify plastic waste suitable for use as a binding additive in road repair — improving road durability while reducing plastic waste.

---

## ✨ Key Results

| Metric | Value |
|--------|-------|
| Plastic classification accuracy | **97.8%** |
| Pothole detection precision | **95.8%** |
| Road durability improvement | **40–50%** |
| Maintenance frequency reduction | **30%** |

---

## 🏗️ System Architecture

```
Input (Image / Video Feed)
        │
        ▼
┌───────────────────┐     ┌──────────────────────────┐
│  Pothole Detector │     │  Plastic Waste Classifier │
│  (YOLOv8)         │     │  (CNN + Random Forest)    │
└────────┬──────────┘     └────────────┬─────────────┘
         │                             │
         ▼                             ▼
  Bounding Box +               Plastic Type Label
  Severity Score               (PET / HDPE / LDPE …)
         │                             │
         └──────────┬──────────────────┘
                    ▼
         Rule-Based Post-Processor
         (Repair recommendation engine)
                    │
                    ▼
         Structured Report Output
         (location, severity, material suitability)
```

---

## 🚀 Features

- **Real-time pothole detection** using YOLOv8 on video streams or static images
- **Plastic waste classification** into recyclable categories using CNN + Random Forest ensemble
- **Severity scoring** of detected potholes to prioritize repair urgency
- **Material suitability mapping** — identifies which plastic types are fit for road-filler use
- **Rule-based post-processing** for decision-making beyond raw model output
- **Structured JSON output** per detection for downstream integration

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8 (Ultralytics) |
| Image Classification | CNN (TensorFlow / Keras) |
| Classical ML | Random Forest (Scikit-learn) |
| Image Processing | OpenCV |
| Deep Learning Framework | PyTorch, TensorFlow |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
PLASNET/
├── data/
│   ├── raw/                  # Raw images and video inputs
│   ├── processed/            # Preprocessed datasets
│   └── annotations/          # YOLO-format labels
│
├── models/
│   ├── pothole_detector/     # YOLOv8 weights and config
│   └── plastic_classifier/   # CNN model + Random Forest pickle
│
├── src/
│   ├── detect.py             # Pothole detection pipeline
│   ├── classify.py           # Plastic waste classification
│   ├── postprocess.py        # Rule-based repair recommender
│   └── utils.py              # Helper functions
│
├── notebooks/
│   ├── EDA.ipynb             # Exploratory data analysis
│   ├── model_training.ipynb  # Training experiments
│   └── evaluation.ipynb      # Metrics and visualizations
│
├── outputs/                  # Detection results, reports
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended for inference speed)
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/VIKASHlm/PLASNET.git
cd PLASNET

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🔧 Usage

### Pothole Detection

```bash
# Run on a single image
python src/detect.py --source data/raw/road_sample.jpg --output outputs/

# Run on a video file
python src/detect.py --source data/raw/road_video.mp4 --output outputs/

# Run on webcam (live)
python src/detect.py --source 0
```

### Plastic Waste Classification

```bash
# Classify a single image
python src/classify.py --image data/raw/plastic_sample.jpg

# Batch classify a folder
python src/classify.py --folder data/raw/plastics/ --output outputs/report.json
```

### Full Pipeline (Detection + Classification + Report)

```bash
python src/postprocess.py \
  --road data/raw/road_video.mp4 \
  --plastic data/raw/plastic_samples/ \
  --output outputs/repair_report.json
```

---

## 📊 Model Performance

### Pothole Detection (YOLOv8)

| Class | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| Pothole | 95.8% | 93.4% | 94.6% |

### Plastic Classification (CNN + RF Ensemble)

| Plastic Type | Accuracy |
|--------------|----------|
| PET | 98.1% |
| HDPE | 97.5% |
| LDPE | 97.2% |
| Mixed/Other | 96.8% |
| **Overall** | **97.8%** |

---

## 🌍 Real-World Impact

- Roads repaired with plastic-bitumen mix show **40–50% higher durability** vs conventional asphalt
- AI-driven prioritization reduces **manual inspection time significantly**
- Reduces plastic waste that would otherwise end up in landfills or water bodies
- Scalable to municipal road maintenance workflows via JSON output integration

---

## 🗺️ Roadmap

- [ ] Deploy as a REST API (FastAPI) for municipal integration
- [ ] Add GPS-tagged pothole mapping for city-level dashboards
- [ ] Mobile app for field workers to submit images and receive instant repair recommendations
- [ ] Expand plastic classification to 10+ categories
- [ ] Edge deployment on Raspberry Pi / Jetson Nano for on-site use

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 👤 Author

**Dasa Vikash R**
M.Tech Software Engineering, VIT Chennai

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/dasa-vikash-r-355602235)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/VIKASHlm)
[![Email](https://img.shields.io/badge/Email-Contact-red?style=flat-square&logo=gmail)](mailto:rdasavikash2004@gmail.com)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> *"The best road repair is the one that doesn't need to happen again."*
