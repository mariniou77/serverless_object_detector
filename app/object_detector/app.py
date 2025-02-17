from ultralytics import YOLO
import cv2
import json
import os

def detect_objects(frame_paths, output_dir):
    model = YOLO("yolov8n.pt")  # Load pretrained model
    detections = []

    for frame_path in frame_paths:
        results = model.predict(frame_path)  # Detect all objects without filtering
        timestamp = os.path.basename(frame_path).split("_")[1].replace(".jpg", "")
        
        for box in results[0].boxes:
            detections.append({
                "timestamp": timestamp,
                "object": model.names[int(box.cls)],  # Class name
                "confidence": float(box.conf),  # Confidence score
            })
    
    # Save output
    output_path = os.path.join(output_dir, "detections.json")
    with open(output_path, "w") as f:
        json.dump({"detections": detections}, f, indent=4)
    
    return output_path

if __name__ == "__main__":
    # Read input from the frame extraction process
    input_json_path = "output/extracted_frames.json"
    with open(input_json_path, "r") as f:
        input_data = json.load(f)
    
    frame_paths = input_data["frames"]
    output_dir = "output/detections"
    os.makedirs(output_dir, exist_ok=True)
    
    detect_objects(frame_paths, output_dir)
