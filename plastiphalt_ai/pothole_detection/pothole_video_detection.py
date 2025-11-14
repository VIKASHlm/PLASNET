import cv2
import json
import time
import os
from inference_sdk import InferenceHTTPClient

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="UJpHlPdOWnAAVJ46BfwN"
)

MODEL_ID = "pothole-vhmow/2"

def process_video(video_path, log_file="output.json", frame_interval=10):
    """
    Detects potholes in video frames using Roboflow API, 
    draws bounding boxes, saves frames with detections as images, 
    and logs all coordinates to JSON.
    """
    # Create output folder for detected pothole frames
    output_folder = "_potholes"
    os.makedirs(output_folder, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video file.")
        return

    pothole_log = []
    frame_count = 0
    saved_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every Nth frame
        if frame_count % frame_interval != 0:
            continue

        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)

        try:
            result = CLIENT.infer(temp_path, model_id=MODEL_ID)
            detections = result.get("predictions", [])
        except Exception as e:
            print(f"⚠️ Frame {frame_count} - API error: {e}")
            continue

        if detections:
            print(f"✅ Frame {frame_count}: {len(detections)} potholes detected")

            # Draw bounding boxes
            for det in detections:
                x, y, w, h = int(det["x"]), int(det["y"]), int(det["width"]), int(det["height"])
                conf = det["confidence"]
                label = f"Pothole {conf:.2f}"

                # Draw rectangle and label
                cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x - w//2, y - h//2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Save coordinates
                pothole_log.append({
                    "frame": frame_count,
                    "x_center": x,
                    "y_center": y,
                    "width": w,
                    "height": h,
                    "confidence": round(conf, 2)
                })

            # Save the frame with detected potholes as an image
            saved_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(saved_filename, frame)
            saved_count += 1

        # Display frame in a live window
        cv2.imshow("Pothole Detection", frame)

        # Press 'q' to stop early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Stopped by user.")
            break

        # Small delay to reduce lag
        time.sleep(0.05)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Save results to JSON file
    with open(log_file, "w") as f:
        json.dump(pothole_log, f, indent=4)

    duration = round(time.time() - start_time, 2)
    print(f"\n✅ Processed {frame_count} frames in {duration}s")
    print(f"🗂️ Coordinates saved to: {os.path.abspath(log_file)}")
    print(f"🖼️ {saved_count} frames with potholes saved to: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    process_video(r"D:\plastiphalt_ai_project\plastiphalt_ai\pothole_detection\input_video_3.mp4")
