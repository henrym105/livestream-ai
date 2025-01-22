from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles  # Add back static files
import base64
import asyncio
import cv2
import yt_dlp
import threading
import os
from ultralytics import YOLO
import time
import numpy as np
import torch  # Add this import
import json

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global configuration
FRAME_DIR = "frames"
os.makedirs(FRAME_DIR, exist_ok=True)  # Ensure frame storage directory exists

# Configure device
device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
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

# Add WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"New client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast_frame(self, frame):
        if not self.active_connections:
            return
            
        try:
            # Compress frame for faster transmission
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = base64.b64encode(buffer).decode('utf-8')
            
            # Broadcast to all connected clients
            disconnect_list = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(frame_bytes)
                except Exception as e:
                    print(f"Error sending frame: {e}")
                    disconnect_list.append(connection)
            
            # Clean up disconnected clients
            for connection in disconnect_list:
                self.disconnect(connection)
                
        except Exception as e:
            print(f"Error broadcasting frame: {e}")

manager = ConnectionManager()

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
        self.latest_frame_path = os.path.join(FRAME_DIR, 'latest_frame.jpg')
        self.manager = manager

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

    async def start_stream(self):
        self.running = True
        frame_count = 0
        process_interval = 1.0 / self.frame_rate

        print(f"Starting stream with {self.frame_rate} FPS")
        while self.running and self.capture.isOpened():
            start_time = time.time()
            
            ret, frame = self.capture.read()
            if not ret:
                print("Failed to read frame")
                break

            # Process frame and detect objects
            results = model(frame)[0]
            detections = results.boxes
            detection_count = len([box for box in detections if float(box.conf[0]) > self.conf_threshold])

            if detection_count > 0:
                frame = draw_detections(frame, results, self.conf_threshold)
                print(f"Frame {frame_count}: {detection_count} objects detected")

            # Broadcast frame to all connected clients
            await self.manager.broadcast_frame(frame)

            # Frame rate control
            processing_time = time.time() - start_time
            sleep_time = max(0, process_interval - processing_time)
            await asyncio.sleep(sleep_time)
            
            frame_count += 1

        print("Stream ended")

    def stop_stream(self):
        self.running = False
        if self.capture:
            self.capture.release()

# Controller for YouTube stream handling
streams = {}

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/start")
async def start_stream(request: Request):
    data = await request.json()
    url = data.get("url")
    if not url:
        return {"error": "URL is required"}

    # Stop all existing streams
    for stream_url, stream in list(streams.items()):
        stream.stop_stream()
        streams.pop(stream_url)
        print(f"Stopped stream: {stream_url}")

    yt_stream = YouTubeStream(url, frame_rate=2)
    try:
        yt_stream.initialize_stream()
    except Exception as e:
        return {"error": f"Failed to initialize stream: {e}"}

    asyncio.create_task(yt_stream.start_stream())
    streams[url] = yt_stream

    return {"message": "Stream started successfully"}

@app.post("/stop")
async def stop_stream(request: Request):
    """
    Stop a livestream. 
    Payload format: {"url": "<YouTube livestream URL>"}
    """
    data = await request.json()
    url = data.get("url")
    if not url:
        return {"error": "URL is required"}, 400

    yt_stream = streams.pop(url, None)
    if not yt_stream:
        return {"error": "Stream not found"}, 404

    yt_stream.stop_stream()
    return {"message": "Stream stopped successfully"}

@app.get("/frames")
async def list_frames():
    """
    List all frames saved from the livestream.
    """
    frames = [f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg")]
    return {"frames": frames}

@app.get("/streams")
async def get_streams():
    """Get available livestream URLs."""
    try:
        with open("static/live-cam-urls.json") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
