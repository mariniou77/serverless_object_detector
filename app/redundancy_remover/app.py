import cv2
import numpy as np
import os
import json

def compare_frames(frame1, frame2, threshold=0.05):
    # Calculate pixel difference between two frames
    diff = cv2.absdiff(frame1, frame2)
    non_zero = np.count_nonzero(diff)
    total_pixels = diff.size
    return (non_zero / total_pixels) > threshold  # Keep if >5% pixels change

def remove_redundant_frames(frame_paths, output_dir):
    prev_frame = None
    unique_frames = []
    for path in frame_paths:
        frame = cv2.imread(path)
        if prev_frame is None or compare_frames(prev_frame, frame):
            unique_path = os.path.join(output_dir, os.path.basename(path))
            cv2.imwrite(unique_path, frame)
            unique_frames.append(unique_path)
            prev_frame = frame
    return unique_frames

if __name__ == "__main__":
    # Read input from the FRAME EXTRACTOR'S output
    input_json_path = "output/extracted_frames.json"
    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    output_dir = "output/unique_frames"
    os.makedirs(output_dir, exist_ok=True)

    # Process frames and save output
    unique_frames = remove_redundant_frames(input_data["frames"], output_dir)
    output_data = {
        "frames": [os.path.abspath(frame) for frame in unique_frames],
        "target_objects": input_data["target_objects"]
    }

    output_json_path = "output/unique_frames.json"
    with open(output_json_path, "w") as f:
        json.dump(output_data, f)

    print(f"Redundancy removal complete. Output: {output_json_path}")