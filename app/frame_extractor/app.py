import cv2
import os
import json
from datetime import datetime

def extract_frames(video_path, output_dir, interval_sec=1):
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)
    count = 0
    frames = []

    while vidcap.isOpened():
        success, frame = vidcap.read()
        if not success:
            break
        if count % frame_interval == 0:
            # Use UTC-based timestamp for consistent naming
            timestamp = str(datetime.utcfromtimestamp(count / fps)).replace(":", "-")
            frame_path = os.path.join(output_dir, f"frame_{timestamp}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
        count += 1
    vidcap.release()
    return frames

def handler(event, context):
    """
    AWS Lambda handler for frame extraction.
    Expects an event dict with:
      {
        'video_path': str,
        'interval_sec': int,
        'target_objects': list of str
      }
    """
    # 1. Read inputs
    video_path = event.get("video_path", "input_videos/test_video.mp4")
    interval_sec = event.get("interval_sec", 1)
    target_objects = event.get("target_objects", ["person", "car"])

    # 2. Ensure output directories exist
    output_dir = "output/extracted_frames"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Extract frames
    frames = extract_frames(video_path, output_dir, interval_sec)

    # 4. Format result
    #    - Typically you'd pass frames & target_objects to the next step
    #    - We'll also write them to JSON for local debugging
    result = {
        "frames": [os.path.abspath(f) for f in frames],
        "target_objects": target_objects,
    }
    output_json = "output/extracted_frames.json"
    with open(output_json, "w") as f:
        json.dump(result, f)

    # 5. Return result (HTTP-like response)
    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }

# Optional: for local debugging without Lambda
if __name__ == "__main__":
    mock_event = {
        "video_path": "input_videos/test_video.mp4",
        "interval_sec": 1,
        "target_objects": ["person", "car", "surfboard", "tower"]
    }
    response = handler(mock_event, None)
    print("Local run response:", response)
