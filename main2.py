from flask import Flask, request, jsonify
import cv2
import yt_dlp
import threading
import os
from ultralytics import YOLO
import time
import numpy as np
import torch  # Add this import

# Initialize Flask app
app = Flask(__name__)

# Global configuration
FRAME_DIR = "frames"
os.makedirs(FRAME_DIR, exist_ok=True)  # Ensure frame storage directory exists

# Configure device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load YOLOv8n model
# model = YOLO('yolov8n.pt')
# model = YOLO('yolo11n.pt')
model = YOLO('yolo11x.pt')
model.to(device)  # Move model to MPS device

# Global color mapping for object classes
CLASS_COLORS = {}

def get_color_for_class(class_id):
    """Generate and store a consistent color for each class."""
    if class_id not in CLASS_COLORS:
        np.random.seed(class_id)
        CLASS_COLORS[class_id] = tuple(map(int, np.random.randint(0, 255, 3)))
    return CLASS_COLORS[class_id]

def draw_detections(frame, results, conf_threshold=0.01):
    """Draw bounding boxes and labels for all detected objects above confidence threshold."""
    for box in results.boxes:
        # Get coordinates and confidence - move computations to CPU for numpy
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        
        if conf < conf_threshold:
            continue
            
        # Get class name and color
        class_name = results.names[cls]
        color = get_color_for_class(cls)
        
        # Draw bounding box with class-specific color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare and draw label
        label = f"{class_name}: {conf:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        cv2.rectangle(
            frame,
            (x1, y1 - label_height - 5),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    return frame

# Function to fetch video stream
class YouTubeStream:
    def __init__(self, url, frame_rate=2):
        self.url = url
        self.stream = None
        self.capture = None
        self.running = False
        self.frame_rate = frame_rate
        self.frames = []
        self.conf_threshold = 0.01  # Add confidence threshold

    def initialize_stream(self):
        try:
            # Use yt-dlp to extract the direct video stream URL
            ydl_opts = {
                # "quiet": True,
                # "format": "best[ext=mp4]",
                # "format": "229",  # 240p 30fps video only
                "format": "232",    # 720p video only
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
                self.stream = info["url"]
            
            # Open the stream with OpenCV
            self.capture = cv2.VideoCapture(self.stream)
            if not self.capture.isOpened():
                raise Exception("Failed to open video capture.")
        except Exception as e:
            print(f"Failed to initialize stream: {e}")
            raise

    def start_stream(self):
        self.running = True
        frame_count = 0
        process_interval = 1.0 / self.frame_rate

        while self.running and self.capture.isOpened():
            start_time = time.time()
            
            ret, frame = self.capture.read()
            if not ret:
                break

            frame_id = (frame_count % 1) + 1
            
            # Detect all objects
            # results = model(frame.to(device))[0]
            results = model(frame)[0]
            detections = results.boxes
            detection_count = len([box for box in detections if float(box.conf[0]) > self.conf_threshold])

            if detection_count > 0:
                frame = draw_detections(frame, results, self.conf_threshold)
                print(f"Frame {frame_id}: {detection_count} objects detected")

            # Save and display frame handling
            frame_path = os.path.join(FRAME_DIR, f"frame_{frame_id}.jpg")
            cv2.imwrite(frame_path, frame)
            if frame_path not in self.frames:
                self.frames.append(frame_path)

            if len(self.frames) > 3:
                oldest_frame = self.frames.pop(0)
                os.remove(oldest_frame)

            # Frame rate control
            processing_time = time.time() - start_time
            sleep_time = max(0, process_interval - processing_time)
            time.sleep(sleep_time)
            
            frame_count += 1

    def stop_stream(self):
        self.running = False
        if self.capture:
            self.capture.release()

# Controller for YouTube stream handling
streams = {}

@app.route("/start", methods=["POST"])
def start_stream():
    """
    Start a livestream from a YouTube URL. 
    Payload format: {"url": "<YouTube livestream URL>"}
    """
    data = request.json
    url = data.get("url")
    if not url:
        return jsonify({"error": "URL is required"}), 400

    if url in streams:
        return jsonify({"error": "Stream already running"}), 400

    yt_stream = YouTubeStream(url, frame_rate=2)
    try:
        yt_stream.initialize_stream()
    except Exception as e:
        return jsonify({"error": f"Failed to initialize stream: {e}"}), 500

    thread = threading.Thread(target=yt_stream.start_stream)
    thread.start()
    streams[url] = yt_stream

    return jsonify({"message": "Stream started successfully"}), 200

@app.route("/stop", methods=["POST"])
def stop_stream():
    """
    Stop a livestream. 
    Payload format: {"url": "<YouTube livestream URL>"}
    """
    data = request.json
    url = data.get("url")
    if not url:
        return jsonify({"error": "URL is required"}), 400

    yt_stream = streams.pop(url, None)
    if not yt_stream:
        return jsonify({"error": "Stream not found"}), 404

    yt_stream.stop_stream()
    return jsonify({"message": "Stream stopped successfully"}), 200

@app.route("/frames", methods=["GET"])
def list_frames():
    """
    List all frames saved from the livestream.
    """
    frames = [f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg")]
    return jsonify({"frames": frames}), 200

if __name__ == "__main__":
    app.run(debug=True)
