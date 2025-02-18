import os
import cv2
import json
import shutil
import numpy as np
from ultralytics import YOLO

# Toggle local mode for S3 simulation
LOCAL_MODE = os.environ.get("LOCAL_MODE", "1") == "1"

def download_file(bucket, key, local_path):
    """
    In local mode, simulate an S3 download by copying from local_s3_bucket/<bucket>/<key>.
    If 'bucket' is empty, it just goes under local_s3_bucket/<key>.
    """
    if LOCAL_MODE:
        source_path = os.path.join("local_s3_bucket", bucket, key)
        print(f"Simulating download: {source_path} -> {local_path}")
        shutil.copy(source_path, local_path)
    else:
        import boto3
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket, key, local_path)

def upload_file(local_path, bucket, key):
    """
    In local mode, simulate an S3 upload by copying to local_s3_bucket/<bucket>/<key>.
    If 'bucket' is empty, it just goes under local_s3_bucket/<key>.
    """
    if LOCAL_MODE:
        dest_dir = os.path.join("local_s3_bucket", bucket, os.path.dirname(key))
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join("local_s3_bucket", bucket, key)
        print(f"Simulating upload: {local_path} -> {dest_path}")
        shutil.copy(local_path, dest_path)
    else:
        import boto3
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_path, bucket, key)

def detect_objects(frame_paths, model_path, target_objects=[]):
    """
    Loads the YOLO model from model_path, runs detection on each frame in frame_paths,
    and returns a list of detections that match 'target_objects'.
    """
    model = YOLO(model_path)
    detections = []

    for local_frame_path in frame_paths:
        # Run YOLO prediction on the local image
        results = model.predict(local_frame_path)
        # The base filename can serve as an ID for the frame
        frame_filename = os.path.basename(local_frame_path)

        # YOLOv8's 'results[0].boxes' contains bounding boxes for that frame
        for box in results[0].boxes:
            class_id = int(box.cls)
            obj_name = model.names[class_id]  # e.g. 'person', 'car', etc.
            confidence = float(box.conf)

            # Only record if it's in the target_objects list (if provided)
            if not target_objects or obj_name in target_objects:
                detections.append({
                    "frame": frame_filename,
                    "object": obj_name,
                    "confidence": confidence
                })
    return detections

def lambda_handler(event, context):
    """
    Expected event:
    {
      "bucket": "",  # or some folder name if you prefer
      "metadata_key": "unique_frames.json",
      "model_key": "yolov8n.pt"  (optional if you want to store your model in local_s3_bucket)
    }

    The 'unique_frames.json' should look like:
    {
      "frames": ["unique_frames/frame_1970-01-01_00-00-00.jpg", ...],
      "target_objects": ["person", "car", ...]
    }
    """
    bucket = event.get("bucket", "")
    metadata_key = event.get("metadata_key", "unique_frames.json")
    model_key = event.get("model_key", "yolov8n.pt")  # default name

    # 1. Download unique_frames.json to /tmp
    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    local_metadata_path = os.path.join(tmp_dir, "unique_frames.json")
    download_file(bucket, metadata_key, local_metadata_path)

    # 2. Parse the JSON to get frame keys + target objects
    with open(local_metadata_path, "r") as f:
        metadata = json.load(f)

    frame_keys = metadata["frames"]          # e.g. ["unique_frames/frame_1970-01-01_00-00-00.jpg", ...]
    target_objects = metadata["target_objects"]

    # 3. Download each unique frame to /tmp for detection
    local_frames = []
    for frame_key in frame_keys:
        frame_filename = os.path.basename(frame_key)
        local_frame_path = os.path.join(tmp_dir, frame_filename)
        download_file(bucket, frame_key, local_frame_path)
        local_frames.append(local_frame_path)

    # 4. Download the YOLO model if needed (or just reference a local path)
    #    If you prefer to keep your model in local_s3_bucket, do this:
    local_model_path = os.path.join(tmp_dir, os.path.basename(model_key))
    download_file(bucket, model_key, local_model_path)

    # 5. Run object detection
    detections = detect_objects(local_frames, local_model_path, target_objects)

    # 6. Create detections.json
    detections_data = {
        "detections": detections,
        "target_objects": target_objects
    }
    local_detections_json = os.path.join(tmp_dir, "detections.json")
    with open(local_detections_json, "w") as f:
        json.dump(detections_data, f, indent=2)

    # 7. Upload detections.json back to local S3
    detections_key = "detections.json"
    upload_file(local_detections_json, bucket, detections_key)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Object detection complete",
            "detections_key": detections_key
        })
    }

# For local testing only
if __name__ == "__main__":
    event = {
        "bucket": "",  # If your files are directly under local_s3_bucket/
        "metadata_key": "unique_frames.json",
        "model_key": "yolov8n.pt",  # Or "models/yolov8n.pt" if you keep it in subfolder
    }
    result = lambda_handler(event, None)
    print(result)
