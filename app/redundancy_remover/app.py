import cv2
import numpy as np
import os
import json

def compare_frames(frame1, frame2, threshold=0.05):
    """
    Calculate pixel difference between two frames.
    Return True if difference is above the threshold.
    """
    diff = cv2.absdiff(frame1, frame2)
    non_zero = np.count_nonzero(diff)
    total_pixels = diff.size
    return (non_zero / total_pixels) > threshold

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

def handler(event, context):
    """
    AWS Lambda handler for the redundancy remover step.
    Expects 'frames' and 'target_objects' in the event payload:
    {
      'frames': [...paths...],
      'target_objects': [...list of objects...]
    }
    """
    # 1. Read input data from 'event'
    frames_input = event.get("frames", [])
    target_objects = event.get("target_objects", [])

    # 2. Prepare output directory
    output_dir = "output/unique_frames"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Remove redundant frames
    unique_frames = remove_redundant_frames(frames_input, output_dir)

    # 4. Format output data (preserve 'target_objects')
    output_data = {
        "frames": [os.path.abspath(frame) for frame in unique_frames],
        "target_objects": target_objects
    }

    # 5. (Optional) Write to JSON file for local debugging
    output_json_path = "output/unique_frames.json"
    with open(output_json_path, "w") as f:
        json.dump(output_data, f)

    # 6. Return result in Lambda format
    return {
        "statusCode": 200,
        "body": json.dumps(output_data)
    }

# Optional local test
if __name__ == "__main__":
    # This simulates what your 'event' might look like,
    # if the previous step (frame_extractor) returned 'frames' & 'target_objects'
    mock_event = {
        "frames": [
            "path/to/frame1.jpg",
            "path/to/frame2.jpg"
        ],
        "target_objects": ["person", "car"]
    }
    response = handler(mock_event, None)
    print("Local run response:", response)
