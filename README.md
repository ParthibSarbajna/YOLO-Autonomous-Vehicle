# YOLO Autonomous Vehicle Object Detection

## Analysis of Real-Time Object Detection Using YOLO

This project implements YOLO11n object detection for an
autonomous-vehicle perception scenario.

The system processes an urban vehicle scene and detects objects
such as buses and pedestrians. It also measures inference time
and approximate frames per second (FPS).

## Objectives

- Implement YOLO-based object detection.
- Detect road-scene objects relevant to autonomous vehicles.
- Draw bounding boxes and confidence scores.
- Store detection results in CSV format.
- Measure inference speed.
- Analyze suitability for real-time perception.

## Technology Stack

- Python
- Ultralytics YOLO
- YOLO11n
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Google Colab
- NVIDIA Tesla T4

## System Pipeline

Input Image -> Preprocessing -> YOLO11n ->
Object Classes + Confidence -> Annotated Image ->
Performance Analysis

## Experimental Results

The vehicle scene produced detections including a bus and
multiple pedestrians.

The highest-confidence bus detection had approximately
0.94 confidence.

## Performance Results

| Metric | Result |
|---|---:|
| Model | YOLO11n |
| GPU | NVIDIA Tesla T4 |
| Single inference time | 0.0626 s |
| Single-run FPS | 15.97 |
| Benchmark runs | 20 |
| Average inference time | 0.0202 s |
| Minimum inference time | 0.0184 s |
| Maximum inference time | 0.0249 s |
| Standard deviation | 0.0016 s |
| Average benchmark FPS | 49.53 |

## Important Limitation

The measured FPS depends on the Google Colab runtime and
NVIDIA Tesla T4 GPU. It is not a guaranteed production
autonomous-vehicle frame rate.

A complete AV benchmark would require a larger labeled dataset
and metrics such as precision, recall and mAP.

## Project Structure

```text
YOLO-Autonomous-Vehicle/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- model_comparison.md
|-- COLAB_INSTRUCTIONS.md
|-- YOLO_Autonomous_Vehicle_Project.ipynb
|
|-- src/
|   |-- detect_image.py
|   |-- performance_analysis.py
|
|-- input/
|   |-- vehicle_scene.jpg
|
|-- output/
|   |-- vehicle_detection.jpg
|
|-- results/
|   |-- vehicle_detection_results.csv
|   |-- experiment_summary.csv
|   |-- experiment_summary.txt
|   |-- graphs/
|
|-- report/
|   |-- Project_Report.pdf
```

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python src/detect_image.py
```

The YOLO model weights are downloaded automatically.

## Notebook

The complete experimental workflow is provided in
YOLO_Autonomous_Vehicle_Project.ipynb.

## Conclusion

The project demonstrates YOLO11n object detection in an urban
vehicle scene and evaluates inference speed. The experiment
successfully detected buses and pedestrians while providing
measurable inference performance.