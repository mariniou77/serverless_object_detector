from ultralytics import YOLO
import cv2
import json
import os

def detect_objects(frame_paths, target_objects, output_dir):
    model = YOLO("yolov8n.pt")  # Load pretrained model
    detections = []

    for frame_path in frame_paths:
        results = model.predict(frame_path, classes=[2, 16])  # COCO classes: car=2, dog=16
        timestamp = os.path.basename(frame_path).split("_")[1].replace(".jpg", "")
        for box in results[0].boxes:
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]
            if cls_name in target_objects:
                detections.append({
                    "timestamp": timestamp,
                    "object": cls_name,
                    "confidence": float(box.conf)
                })
    # Save output
    output_path = os.path.join(output_dir, "detections.json")
    with open(output_path, "w") as f:
        json.dump({"detections": detections}, f)
    return output_path

if __name__ == "__main__":
    # Read input from the REDUNDANCY REMOVER'S output
    input_json_path = "output/unique_frames.json"
    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    output_dir = "output/detections"
    os.makedirs(output_dir, exist_ok=True)

    # Detect objects and save output
    detect_objects(input_data["frames"], input_data["target_objects"], output_dir)
    print(f"Detection complete. Output: {output_dir}/detections.json")