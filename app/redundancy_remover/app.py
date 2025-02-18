import os
import cv2
import json
import shutil
import numpy as np

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

def compare_frames(frame1, frame2, threshold=0.05):
    """
    Return True if the difference between frame1 and frame2 is above the threshold.
    This means they are "different enough" to keep.
    """
    diff = cv2.absdiff(frame1, frame2)
    non_zero = np.count_nonzero(diff)
    total_pixels = diff.size
    fraction_diff = non_zero / total_pixels
    return fraction_diff > threshold

def remove_redundant_frames(local_frame_paths, output_dir, threshold=0.05):
    """
    Given a list of frame paths (on the local filesystem), compare consecutive frames
    and keep only those that differ by more than the threshold.
    """
    unique_local_paths = []
    prev_frame = None

    for path in local_frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        if prev_frame is None or compare_frames(prev_frame, frame, threshold=threshold):
            unique_local_paths.append(path)
            prev_frame = frame

    # Copy unique frames into output_dir
    final_paths = []
    for src_path in unique_local_paths:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(output_dir, filename)
        shutil.copy(src_path, dst_path)
        final_paths.append(dst_path)

    return final_paths

def lambda_handler(event, context):
    """
    Expected event:
    {
      "bucket": "",                    # empty string, matching your local structure
      "metadata_key": "extracted_frames.json"
    }

    The 'extracted_frames.json' should be in local_s3_bucket/extracted_frames.json and look like:
    {
      "frames": ["extracted_frames/frame_1970-01-01_00-00-00.jpg", ...],
      "target_objects": ["person", "car", ...]
    }
    """
    bucket = event["bucket"]
    metadata_key = event["metadata_key"]  # e.g. "extracted_frames.json"

    # 1. Download extracted_frames.json to /tmp
    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    local_metadata_path = os.path.join(tmp_dir, "extracted_frames.json")

    download_file(bucket, metadata_key, local_metadata_path)

    # 2. Parse the JSON to get frame keys + target objects
    with open(local_metadata_path, "r") as f:
        metadata = json.load(f)

    frame_keys = metadata["frames"]  # e.g. ["extracted_frames/frame_1970-01-01_00-00-00.jpg", ...]
    target_objects = metadata["target_objects"]

    # 3. Download each frame to /tmp for local comparison
    local_frames = []
    for frame_key in frame_keys:
        frame_filename = os.path.basename(frame_key)
        local_frame_path = os.path.join(tmp_dir, frame_filename)
        download_file(bucket, frame_key, local_frame_path)
        local_frames.append(local_frame_path)

    # 4. Remove redundant frames
    unique_frames_dir = os.path.join(tmp_dir, "unique_frames")
    os.makedirs(unique_frames_dir, exist_ok=True)
    unique_local_paths = remove_redundant_frames(local_frames, unique_frames_dir)

    # 5. Upload unique frames back to "S3" (local folder)
    unique_s3_prefix = "unique_frames/"
    uploaded_unique_frames = []
    for path in unique_local_paths:
        filename = os.path.basename(path)
        unique_frame_key = unique_s3_prefix + filename
        upload_file(path, bucket, unique_frame_key)
        uploaded_unique_frames.append(unique_frame_key)

    # 6. Create unique_frames.json with updated frames + the same target_objects
    unique_metadata = {
        "frames": uploaded_unique_frames,
        "target_objects": target_objects
    }
    local_unique_json = os.path.join(tmp_dir, "unique_frames.json")
    with open(local_unique_json, "w") as f:
        json.dump(unique_metadata, f)

    # 7. Upload unique_frames.json
    unique_json_key = "unique_frames.json"
    upload_file(local_unique_json, bucket, unique_json_key)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Redundancy removal complete",
            "unique_frames_key": unique_json_key
        })
    }

# For local testing only
if __name__ == "__main__":
    event = {
        "bucket": "",                     # Empty because frames + JSON are in local_s3_bucket/...
        "metadata_key": "extracted_frames.json"
    }
    # Make sure local_s3_bucket/extracted_frames.json exists
    # (the output from your frame_extractor).
    result = lambda_handler(event, None)
    print(result)
