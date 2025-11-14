
# Plastiphalt AI

Implements:
- AI-based plastic classification
- AI-based sorting
- AI/ML suggested optimized Plastiphalt designer
- AI-based crack finding

## Quickstart

```
cd plastiphalt_ai

# Classification (train)
python app.py classify --train /path/to/datasets --model models/plastic_rf.joblib

# Classification (single image)
python app.py classify --image /path/to/img.jpg --model models/plastic_rf.joblib

# Sorting
python app.py sort --detections tests/sample_detections.json --severity high

# Designer
python app.py design --severity high --traffic 25000 --temperature 35   --available '{"HDPE":300,"PP":250,"LDPE":120,"PET":60,"PVC":0}'   --bitumen 1000 --agg_ratio 10

# Crack detection
python app.py crack --image /path/to/road.jpg --out /tmp/crack_mask.png
```

Dependencies: numpy, scikit-learn, opencv-python, joblib, imageio.
