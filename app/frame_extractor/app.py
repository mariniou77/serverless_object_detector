import os
import cv2
import json
import shutil  # For local file copying
from datetime import datetime

# Set up a flag to indicate local mode (simulate S3)
LOCAL_MODE = os.environ.get("LOCAL_MODE", "1") == "1"

# --- Helper functions to simulate S3 operations locally ---
def download_file(bucket, key, local_path):
    """
    In local mode, simulate S3 download by copying from a local directory.
    """
    if LOCAL_MODE:
        source_path = os.path.join("local_s3_bucket", key)
        print(f"Simulating download: {source_path} -> {local_path}")
        shutil.copy(source_path, local_path)
    else:
        # When using AWS, you would use boto3 here.
        import boto3
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket, key, local_path)

def upload_file(local_path, bucket, key):
    """
    In local mode, simulate S3 upload by copying to a local directory.
    """
    if LOCAL_MODE:
        dest_dir = os.path.join("local_s3_bucket", os.path.dirname(key))
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join("local_s3_bucket", key)
        print(f"Simulating upload: {local_path} -> {dest_path}")
        shutil.copy(local_path, dest_path)
    else:
        # When using AWS, you would use boto3 here.
        import boto3
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_path, bucket, key)

# --- The Frame Extraction Function ---
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
            # Create a unique filename using a timestamp
            timestamp = str(datetime.utcfromtimestamp(count / fps)).replace(":", "-")
            frame_filename = f"frame_{timestamp}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frames.append(frame_filename)  # Save just the file name for later reference
        count += 1
    vidcap.release()
    return frames

# --- Lambda Handler for Frame Extraction ---
def lambda_handler(event, context):
    """
    Expected event format:
    {
      "bucket": "your-bucket-name",         # For local testing, this is arbitrary.
      "video_key": "videos/test_video.mp4",   # Path relative to local_s3_bucket folder.
      "target_objects": ["person", "car"]      # List of target objects.
    }
    """
    bucket = event['bucket']
    video_key = event['video_key']
    target_objects = event.get('target_objects', [])

    # Use /tmp as our working directory (mimicking Lambda)
    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    video_filename = os.path.basename(video_key)
    video_local_path = os.path.join(tmp_dir, video_filename)

    # Download the video (simulated locally)
    download_file(bucket, video_key, video_local_path)

    # Create a directory for extracted frames in /tmp
    frames_dir = os.path.join(tmp_dir, "extracted_frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Extract frames from the downloaded video
    frame_files = extract_frames(video_local_path, frames_dir)

    # Simulate uploading extracted frames to S3 (local folder)
    frames_s3_prefix = "extracted_frames/"
    uploaded_frames = []
    for frame_filename in frame_files:
        frame_local_path = os.path.join(frames_dir, frame_filename)
        frame_s3_key = frames_s3_prefix + frame_filename
        upload_file(frame_local_path, bucket, frame_s3_key)
        uploaded_frames.append(frame_s3_key)

    # Create metadata JSON
    metadata = {
        "frames": uploaded_frames,
        "target_objects": target_objects
    }
    metadata_local_path = os.path.join(tmp_dir, "extracted_frames.json")
    with open(metadata_local_path, "w") as f:
        json.dump(metadata, f)

    # Simulate uploading metadata JSON to S3
    metadata_s3_key = "extracted_frames.json"
    upload_file(metadata_local_path, bucket, metadata_s3_key)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Frame extraction completed",
            "metadata_key": metadata_s3_key
        })
    }

# --- For local testing only ---
if __name__ == "__main__":
    # Example event payload for local testing
    event = {
        "bucket": "my-local-bucket",              # This folder (local_s3_bucket/my-local-bucket) should exist.
        "video_key": "videos/test_video.mp4",       # Ensure local_s3_bucket/videos/test_video.mp4 exists.
        "target_objects": ["person", "car"]
    }
    # Create a local folder structure for the simulated S3 bucket if needed.
    os.makedirs(os.path.join("local_s3_bucket", "videos"), exist_ok=True)
    # Run the Lambda handler function
    result = lambda_handler(event, None)
    print(result)
