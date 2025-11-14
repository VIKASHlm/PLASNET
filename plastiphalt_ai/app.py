import argparse, json, os
from pathlib import Path
import cv2

from plastiphalt_ai.classifiers.plastic_classifier import PlasticClassifier, PLASTIC_LABELS
from plastiphalt_ai.sorting.sorter import sort_plastics
from plastiphalt_ai.designer.optim_plastiphalt import suggest_mix
from plastiphalt_ai.vision.crack_detector import detect_cracks


# -------------------- CLASSIFY --------------------
def cmd_classify(args):
    clf = PlasticClassifier(model_path=args.model)
    if args.train:
        labels = []
        paths = []
        for label in os.listdir(args.train):
            d = os.path.join(args.train, label)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    paths.append(os.path.join(d, fn))
                    labels.append(label.upper() if label.upper() in PLASTIC_LABELS else "OTHER")
        metrics = clf.fit(paths, labels)
        clf.save()
        print(json.dumps(metrics, indent=2))
    else:
        clf.load()
        proba, classes = clf.predict_proba(args.image)
        pred = clf.predict(args.image)
        print(json.dumps({"prediction": pred, "proba": proba, "classes": classes}, indent=2))


# -------------------- SORT --------------------
def cmd_sort(args):
    detections = json.loads(Path(args.detections).read_text())
    sorted_list = sort_plastics(detections, severity=args.severity)
    print(json.dumps(sorted_list, indent=2))


# -------------------- DESIGN --------------------
def cmd_design(args):
    available = json.loads(args.available)
    plan = suggest_mix(
        args.severity,
        args.traffic,
        args.temperature,
        available,
        batch_bitumen_kg=args.bitumen,
        aggregate_to_bitumen_ratio=args.agg_ratio,
    )
    print(json.dumps(plan, indent=2))


# -------------------- CRACK DETECTION --------------------
def cmd_crack(args):
    out = detect_cracks(args.image, out_mask_path=args.out)
    print(json.dumps(out, indent=2))


# -------------------- PIPELINE (Image) --------------------
def cmd_pipeline(args):
    clf = PlasticClassifier(model_path=args.model).load()
    proba, classes = clf.predict_proba(args.image)
    pred = clf.predict(args.image)

    detection = {
        "id": "test1",
        "label": pred,
        "confidence": max(proba) if proba else 1.0,
    }

    sorted_list = sort_plastics([detection], severity=args.severity)

    plan = suggest_mix(
        args.severity,
        args.traffic,
        args.temperature,
        json.loads(args.available),
        batch_bitumen_kg=args.bitumen,
        aggregate_to_bitumen_ratio=args.agg_ratio,
    )

    print(
        json.dumps(
            {"classification": detection, "sorted": sorted_list, "design": plan}, indent=2
        )
    )


# -------------------- VIDEO PIPELINE (Aggregate) --------------------
def cmd_video_pipeline(args):
    clf = PlasticClassifier(model_path=args.model).load()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return

    frame_count = 0
    detections = []
    class_counts = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % args.frame_skip == 0:
            tmp_path = "temp_frame.jpg"
            cv2.imwrite(tmp_path, frame)

            proba, classes = clf.predict_proba(tmp_path)
            pred = clf.predict(tmp_path)

            detection = {
                "id": f"frame_{frame_count}",
                "label": pred,
                "confidence": max(proba) if proba else 1.0,
            }
            detections.append(detection)

            class_counts[pred] = class_counts.get(pred, 0) + 1

            print(f"[Frame {frame_count}] Detected: {pred} ({detection['confidence']:.2f})")

    cap.release()

    # Aggregate detections into stock usage
    available_kg = json.loads(args.available)
    total_frames = sum(class_counts.values())

    if total_frames > 0:
        for label, count in class_counts.items():
            if label in available_kg:
                available_kg[label] = available_kg[label] * (count / total_frames)

    sorted_list = sort_plastics(detections, severity=args.severity)

    plan = suggest_mix(
        args.severity,
        args.traffic,
        args.temperature,
        available_kg,
        batch_bitumen_kg=args.bitumen,
        aggregate_to_bitumen_ratio=args.agg_ratio,
    )

    print(
        json.dumps(
            {
                "summary_counts": class_counts,
                "detections": detections,
                "sorted": sorted_list,
                "design": plan,
            },
            indent=2,
        )
    )


# -------------------- ARG PARSER --------------------
def build_parser():
    p = argparse.ArgumentParser(description="Plastiphalt AI Toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    # classify
    p1 = sub.add_parser("classify", help="Train or run plastic classifier")
    p1.add_argument("--model", default="models/plastic_rf.joblib")
    g = p1.add_mutually_exclusive_group(required=True)
    g.add_argument("--train", help="Folder with subfolders per label for training")
    g.add_argument("--image", help="Single image to classify")
    p1.set_defaults(func=cmd_classify)

    # sort
    p2 = sub.add_parser("sort", help="Sort detections by suitability")
    p2.add_argument("--detections", required=True, help="JSON file of detections")
    p2.add_argument("--severity", default="medium", choices=["low", "medium", "high"])
    p2.set_defaults(func=cmd_sort)

    # design
    p3 = sub.add_parser("design", help="Suggest optimized plastiphalt mix")
    p3.add_argument("--severity", required=True, choices=["low", "medium", "high"])
    p3.add_argument("--traffic", type=int, required=True, help="Average Daily Traffic (vehicles/day)")
    p3.add_argument("--temperature", type=float, required=True, help="Typical ambient temp (°C)")
    p3.add_argument(
        "--available",
        required=True,
        help='JSON dict of available plastics in kg, e.g. {"HDPE":200,"PP":150,"LDPE":100}',
    )
    p3.add_argument("--bitumen", type=float, default=1000.0)
    p3.add_argument("--agg_ratio", type=float, default=10.0)
    p3.set_defaults(func=cmd_design)

    # crack
    p4 = sub.add_parser("crack", help="Detect cracks in a road image")
    p4.add_argument("--image", required=True)
    p4.add_argument("--out", help="Optional path to save crack mask")
    p4.set_defaults(func=cmd_crack)

    # pipeline (single image)
    p5 = sub.add_parser("pipeline", help="End-to-end: classify -> sort -> design")
    p5.add_argument("--image", required=True)
    p5.add_argument("--model", default="models/plastic_rf.joblib")
    p5.add_argument("--severity", required=True, choices=["low", "medium", "high"])
    p5.add_argument("--traffic", type=int, required=True)
    p5.add_argument("--temperature", type=float, required=True)
    p5.add_argument("--available", required=True)
    p5.add_argument("--bitumen", type=float, default=1000.0)
    p5.add_argument("--agg_ratio", type=float, default=10.0)
    p5.set_defaults(func=cmd_pipeline)

    # video pipeline
    p6 = sub.add_parser("video_pipeline", help="Run pipeline on video (aggregate all detections -> final mix)")
    p6.add_argument("--video", required=True, help="Path to input video file")
    p6.add_argument("--model", default="models/plastic_rf.joblib")
    p6.add_argument("--frame_skip", type=int, default=10, help="Process every Nth frame")
    p6.add_argument("--severity", required=True, choices=["low", "medium", "high"])
    p6.add_argument("--traffic", type=int, required=True)
    p6.add_argument("--temperature", type=float, required=True)
    p6.add_argument("--available", required=True, help="JSON dict of stock plastics in kg")
    p6.add_argument("--bitumen", type=float, default=1000.0)
    p6.add_argument("--agg_ratio", type=float, default=10.0)
    p6.set_defaults(func=cmd_video_pipeline)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
