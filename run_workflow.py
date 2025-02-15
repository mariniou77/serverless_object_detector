# File: run_workflow.py
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import time
import subprocess

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if "extracted_frames" in event.src_path:
            subprocess.run(["python", "app/redundancy_remover/app.py"])

observer = Observer()
observer.schedule(Handler(), path="output/extracted_frames", recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()