
from dataclasses import dataclass, field
from typing import List
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path

PLASTIC_LABELS = ["PET", "HDPE", "LDPE", "PP", "PS", "PVC", "OTHER"]

def _image_features(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv],[0],None,[32],[0,180]).flatten()
    s_hist = cv2.calcHist([hsv],[1],None,[32],[0,256]).flatten()
    v_hist = cv2.calcHist([hsv],[2],None,[32],[0,256]).flatten()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = np.var(lap)
    sobelx = cv2.Sobel(gray, cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F,0,1,ksize=3)
    sobel_mean = np.mean(np.abs(sobelx))+np.mean(np.abs(sobely))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.mean()/255.0
    feats = np.concatenate([h_hist, s_hist, v_hist, [lap_var, sobel_mean, edge_density]])
    return feats.astype(np.float32)

@dataclass
class PlasticClassifier:
    model_path: str = "models/plastic_rf.joblib"
    pipeline: Pipeline = field(init=False, default=None)

    def __post_init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
        ])

    def fit(self, image_paths: List[str], labels: List[str], test_size: float=0.2, random_state: int=42):
        X = np.vstack([_image_features(p) for p in image_paths])
        y = np.array(labels)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        self.pipeline.fit(Xtr, ytr)
        preds = self.pipeline.predict(Xte)
        acc = accuracy_score(yte, preds)
        report = classification_report(yte, preds, output_dict=False)
        return {"accuracy": acc, "report": report}

    def predict(self, image_path: str) -> str:
        feats = _image_features(image_path).reshape(1, -1)
        return str(self.pipeline.predict(feats)[0])

    def predict_proba(self, image_path: str):
        feats = _image_features(image_path).reshape(1, -1)
        if hasattr(self.pipeline[-1], "predict_proba"):
            return self.pipeline[-1].predict_proba(self.pipeline[0].transform(feats))[0].tolist(), self.pipeline[-1].classes_.tolist()
        return None, None

    def save(self):
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)

    def load(self):
        self.pipeline = joblib.load(self.model_path)
        return self
