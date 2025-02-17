from ultralytics import YOLO
import cv2
import json
import os

def detect_objects(frame_paths, output_dir, target_objects=[]):
    model = YOLO("yolov8n.pt")  # Load pretrained model
    detections = []

    for frame_path in frame_paths:
        # Detect all objects without filtering
        results = model.predict(frame_path)
        timestamp = os.path.basename(frame_path).split("_")[1].replace(".jpg", "")

        for box in results[0].boxes:
            obj_name = model.names[int(box.cls)]
            # Check if the detected object matches any of the target objects
            if obj_name in target_objects:
                detections.append({
                    "timestamp": timestamp,
                    "object": obj_name,
                    "confidence": float(box.conf)
                })

    # Save output
    output_path = os.path.join(output_dir, "detections.json")
    with open(output_path, "w") as f:
        json.dump({"detections": detections}, f, indent=4)

    return output_path

if __name__ == "__main__":
    # Read input from the unique_frames.json
    input_json_path = "output/unique_frames.json"
    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    frame_paths = input_data["frames"]
    target_objects = input_data.get("target_objects", [])
    output_dir = "output/detections"
    os.makedirs(output_dir, exist_ok=True)

    detect_objects(frame_paths, output_dir, target_objects)
