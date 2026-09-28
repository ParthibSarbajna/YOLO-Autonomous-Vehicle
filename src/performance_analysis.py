from ultralytics import YOLO
import time
import statistics

IMAGE_PATH = "../input/vehicle_scene.jpg"
model = YOLO("yolo11n.pt")

NUM_RUNS = 20
times = []

print(f"Running {NUM_RUNS} inference tests...")

# Warm-up
model(IMAGE_PATH, verbose=False)

for i in range(NUM_RUNS):
    start = time.time()
    model(IMAGE_PATH, verbose=False)
    elapsed = time.time() - start
    times.append(elapsed)

average_time = statistics.mean(times)
minimum_time = min(times)
maximum_time = max(times)
std_time = statistics.stdev(times)
fps = 1 / average_time

print("\nYOLO11n Performance")
print("-------------------")
print(f"Average inference time : {average_time:.4f} sec")
print(f"Minimum inference time : {minimum_time:.4f} sec")
print(f"Maximum inference time : {maximum_time:.4f} sec")
print(f"Standard deviation     : {std_time:.4f} sec")
print(f"Approximate FPS        : {fps:.2f}")