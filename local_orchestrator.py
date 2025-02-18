import requests
import json

def invoke_lambda(url, payload):
    """
    Helper to invoke a local Lambda container.
    url: e.g. 'http://localhost:9001/2015-03-31/functions/function/invocations'
    payload: dict to send as JSON
    returns: dict parsed from response JSON
    """
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    return response.json()  # the container returns { "statusCode": ..., "body": ... }

if __name__ == "__main__":
    # 1. Call the frame_extractor
    extract_url = "http://localhost:9001/2015-03-31/functions/function/invocations"
    extract_event = {
        "video_path": "input_videos/test_video.mp4",
        "interval_sec": 1,
        # We can include target_objects here, or let the code define them.
        "target_objects": ["person", "car", "surfboard", "tower"]
    }
    extract_response = invoke_lambda(extract_url, extract_event)
    print("Frame Extractor raw response:", extract_response)

    # The local Lambda returns something like:
    # {
    #   "statusCode": 200,
    #   "body": "{\"frames\": [...], \"target_objects\": [...]}",
    # }
    if "body" not in extract_response:
        raise ValueError("Frame Extractor: unexpected response format")

    # Extract body and parse JSON
    extract_data = json.loads(extract_response["body"])
    frames_for_next = extract_data["frames"]
    target_objects = extract_data["target_objects"]  # Pass these along

    # 2. Call the redundancy_remover
    remove_url = "http://localhost:9002/2015-03-31/functions/function/invocations"
    remove_event = {
        "frames": frames_for_next,
        "target_objects": target_objects
    }
    remove_response = invoke_lambda(remove_url, remove_event)
    print("Redundancy Remover raw response:", remove_response)

    if "body" not in remove_response:
        raise ValueError("Redundancy Remover: unexpected response format")

    remove_data = json.loads(remove_response["body"])
    frames_for_final = remove_data["frames"]
    # Keep same target_objects or read from remove_data if it got changed
    target_objects = remove_data["target_objects"]

    # 3. Call the object_detector
    detect_url = "http://localhost:9003/2015-03-31/functions/function/invocations"
    detect_event = {
        "frames": frames_for_final,
        "target_objects": target_objects
    }
    detect_response = invoke_lambda(detect_url, detect_event)
    print("Object Detector raw response:", detect_response)

    if "body" not in detect_response:
        raise ValueError("Object Detector: unexpected response format")

    detect_data = json.loads(detect_response["body"])
    print("\nFinal detection data:\n", json.dumps(detect_data, indent=2))
