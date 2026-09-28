from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import time
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_PATH = INPUT_DIR / "vehicle_scene.jpg"
OUTPUT_PATH = OUTPUT_DIR / "vehicle_detection.jpg"
CSV_PATH = RESULTS_DIR / "vehicle_detection_results.csv"

model = YOLO("yolo11n.pt")

start = time.time()
results = model(str(IMAGE_PATH), conf=0.25)
elapsed = time.time() - start

annotated = results[0].plot()
cv2.imwrite(str(OUTPUT_PATH), annotated)

detections = []

for result in results:
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            detections.append({
                "Object": model.names[class_id],
                "Confidence": confidence
            })

df = pd.DataFrame(detections)
df.to_csv(CSV_PATH, index=False)

print("YOLO Autonomous Vehicle Detection")
print("---------------------------------")
print("Model: YOLO11n")
print(f"Inference time: {elapsed:.4f} seconds")
print(f"Approximate FPS: {1 / elapsed:.2f}")

print("\nDetections:")
print(df.to_string(index=False))

print(f"\nAnnotated image saved to: {OUTPUT_PATH}")
print(f"Detection CSV saved to: {CSV_PATH}")