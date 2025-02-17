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
            timestamp = str(datetime.utcfromtimestamp(count/fps)).replace(":", "-")
            frame_path = os.path.join(output_dir, f"frame_{timestamp}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
        count += 1
    vidcap.release()
    return frames

if __name__ == "__main__":
    video_path = "input_videos/test_video.mp4"  # Replace with your video
    output_dir = "output/extracted_frames"
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames and target_objects
    frames = extract_frames(video_path, output_dir)
    output_data = {
        "frames": [os.path.abspath(frame) for frame in frames],  # Use absolute paths
        "target_objects": ["person", "car", "surfboard", "tower"]  # Or read from input JSON
    }

    # Save output to JSON
    output_json_path = "output/extracted_frames.json"
    with open(output_json_path, "w") as f:
        json.dump(output_data, f)

    print(f"Frame extraction complete. Output: {output_json_path}")