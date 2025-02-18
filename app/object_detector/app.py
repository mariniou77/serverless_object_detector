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

def handler(event, context):
    """
    AWS Lambda handler for the object detector step.
    Expects an event dict like:
    {
      'frames': [...list of frame paths...],
      'target_objects': [...objects to filter...]
    }
    """
    # 1. Extract info from event
    frame_paths = event.get("frames", [])
    target_objects = event.get("target_objects", [])
    output_dir = "output/detections"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Run the detection
    detection_output_path = detect_objects(frame_paths, output_dir, target_objects)

    # 3. Package the final JSON (read back detections.json or re-create it)
    with open(detection_output_path, "r") as f:
        detection_data = json.load(f)

    # 4. Return result in Lambda style
    return {
        "statusCode": 200,
        "body": json.dumps(detection_data)
    }

# Optional: local testing without Lambda
if __name__ == "__main__":
    # Read input from the local JSON if desired
    input_json_path = "output/unique_frames.json"
    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    frame_paths = input_data["frames"]
    target_objects = input_data.get("target_objects", [])
    event_mock = {
        "frames": frame_paths,
        "target_objects": target_objects
    }

    response = handler(event_mock, None)
    print("Local run response:", response)
