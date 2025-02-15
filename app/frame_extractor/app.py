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
    # Example: Extract frames from input_video.mp4 every 1 second
    video_path = "input_videos/sample.mp4"
    output_dir = "output/extracted_frames"
    os.makedirs(output_dir, exist_ok=True)
    frames = extract_frames(video_path, output_dir)
    print(json.dumps({"frames": frames, "target_objects": ["bus", "dog"]}))