# backend/main.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import os
from collections import deque
import threading
import queue
import torch
import torch.nn as nn
from ultralytics import YOLO
import re
import time
import asyncio
import logging
import json
from datetime import datetime
import av
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole
from typing import Tuple, Dict, List

# Enhanced logging for WebRTC
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================== CONFIGURATION ======================
app = FastAPI(title="Gold Rubbing & Acid Detection API")

# Allow frontend (browser) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # In production → change to your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# At the very top of the file, after imports
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# Paths (adjust if needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_GOLD  = os.path.join(BASE_DIR, "models", "best_top2.pt")
MODEL_STONE = os.path.join(BASE_DIR, "models", "best_top_stone.pt")
MODEL_ACID  = os.path.join(BASE_DIR, "models", "best_aci_liq.pt")
# WebRTC audio model - 1D CNN for waveform detection
SOUND_MODEL_PATH = os.path.join(BASE_DIR, "models", "audio_model.pth")

MODEL_STONE = YOLO(MODEL_STONE).to(device)
MODEL_GOLD  = YOLO(MODEL_GOLD).to(device)
MODEL_ACID  = YOLO(MODEL_ACID).to(device)

# Detection settings
CONF_THRESH = 0.5
IMGSZ = 320

# Rubbing detection settings
THRESHOLD_FLUCTUATION = 2.0
NO_OF_FLUCTUATIONS = 3
WINDOW_SIZE = 10

# Colors
STONE_BOX_COLOR = (0, 0, 255)
GOLD_OVERLAY_COLOR = (0, 215, 255)

# Global state (shared between requests - simple approach for demo)
STATE = {
    "stage": "RUBBING",
    "rubbing_done": False,
    "recent_distances": deque(maxlen=WINDOW_SIZE),
    "prev_centroid": None,
    "sound_status": "Waiting...",
}

sound_queue = queue.Queue(maxsize=1)

# ====================== SOUND MODEL (WebRTC Audio) ======================
class WaveCNN1D(nn.Module):
    """1D CNN Model for raw waveform processing from WebRTC audio"""
    def __init__(self, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,    32, kernel_size=80, stride=4, padding=40), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32,   64, kernel_size=5,  stride=1, padding=2),  nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64,  128, kernel_size=5,  stride=1, padding=2),  nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=5,  stride=1, padding=2),  nn.ReLU(), nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        # x: (batch, 1, length)
        z = self.net(x).squeeze(-1)  # → (batch, 256)
        return self.fc(z)

def load_sound_model():
    """Load the waveform CNN model for audio inference"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaveCNN1D(n_classes=2).to(device)
    if not os.path.exists(SOUND_MODEL_PATH):
        logger.warning(f"Sound model not found: {SOUND_MODEL_PATH}")
        return None, device
    try:
        state_dict = torch.load(SOUND_MODEL_PATH, map_location=device, weights_only=True)
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"Model loaded from {SOUND_MODEL_PATH}")
        return model, device
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise RuntimeError(f"Cannot load model from {SOUND_MODEL_PATH}")

# ====================== WEBRTC AUDIO INFERENCE WORKER ======================
class WaveformInferenceWorker:
    """Processes audio chunks from WebRTC stream and performs rubbing sound detection"""
    def __init__(self, sample_rate=16000, window_seconds=2.0, hop_ratio=0.5, confidence_threshold=0.75):
        self.sample_rate = sample_rate
        self.window_sec = window_seconds
        self.hop_ratio = hop_ratio
        self.confidence_threshold = confidence_threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = load_sound_model()
        if self.model is None:
            raise RuntimeError("Failed to load audio model")
        self.model.to(self.device)
        self.model.eval()

        self.win_samples = max(8000, int(sample_rate * window_seconds))
        self.hop_samples = int(self.win_samples * hop_ratio)

        self._accum = []  # list of float32 samples
        
        logger.info(f"Waveform inference ready – window={window_seconds}s, hop={hop_ratio}, confidence_threshold={confidence_threshold}")

    def process_chunk(self, chunk: np.ndarray) -> Tuple[str, float]:
        """
        Process an audio chunk and return prediction
        chunk: np.float32 1D array
        Returns: (label, confidence)
        """
        self._accum.extend(chunk)

        if len(self._accum) >= self.win_samples:
            x = np.array(self._accum[-self.win_samples:], dtype=np.float32)

            # Peak normalization (same as test3_waveform.py)
            max_abs = np.max(np.abs(x)) if np.any(x) else 1.0
            if max_abs > 0:
                x = x / max_abs

            # To tensor: (1, 1, length)
            t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(t)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            idx = np.argmax(probs)
            conf = float(probs[idx])
            label = "OK" if idx == 0 else "NOK"
            
            # Only return prediction if confidence meets threshold
            if conf >= self.confidence_threshold:
                # Update global state
                STATE["sound_status"] = label
                
                # Slide window
                self._accum = self._accum[self.hop_samples:]
                return label, conf
            else:
                # Confidence too low, slide window but don't return prediction
                self._accum = self._accum[self.hop_samples:]
                return "WAIT", conf  # Return WAIT with actual confidence for logging

        return "WAIT", 0.0

    def reset(self):
        """Reset the accumulator buffer"""
        self._accum = []

# Global worker instance
worker = None

def init_worker():
    """Initialize the global worker instance"""
    global worker
    try:
        worker = WaveformInferenceWorker(window_seconds=2.0, hop_ratio=0.5, confidence_threshold=0.75)
        logger.info("Waveform inference worker initialized")
    except Exception as e:
        logger.error(f"Failed to initialize worker: {e}")
        worker = None

# ====================== DETECTION FUNCTIONS ======================
def process_rubbing_frame(frame, model_stone, model_gold):
    H, W = frame.shape[:2]
    annotated = frame.copy()
    gold_clipped_full = np.zeros((H, W), dtype=np.uint8)
    gold_mask_pct = 0.0
    stone_bbox = None

    # Stone detection
    try:
        res_s = model_stone.predict(frame, imgsz=IMGSZ, conf=CONF_THRESH, iou=0.45, verbose=False)[0]
        boxes = getattr(res_s, "boxes", None)
        if boxes is not None and getattr(boxes, "xyxy", None) is not None and len(boxes.xyxy) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            largest_area = 0
            largest_box = None
            for b in xyxy:
                x1, y1, x2, y2 = map(int, b)
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest_box = (x1, y1, x2, y2)
            if largest_box:
                x1, y1, x2, y2 = largest_box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), STONE_BOX_COLOR, 2)
                stone_bbox = (x1, y1, x2, y2)
    except Exception as e:
        print(f"Stone detection error: {e}")

    # Gold detection & overlay
    if stone_bbox:
        sx1, sy1, sx2, sy2 = stone_bbox
        try:
            sx1, sy1, sx2, sy2 = stone_bbox
            # add padding around bbox (20% of max side)
            w = sx2 - sx1
            h = sy2 - sy1
            pad = max(5, int(0.2 * max(w, h)))
            cx1 = max(0, sx1 - pad)
            cy1 = max(0, sy1 - pad)
            cx2 = min(annotated.shape[1], sx2 + pad)
            cy2 = min(annotated.shape[0], sy2 + pad)

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                raise ValueError("Empty crop for gold inference")

                        # Choose imgsz relative to crop to avoid unnecessary upscaling
            res_g = model_gold.predict(crop, imgsz=IMGSZ, conf=CONF_THRESH, iou=0.45, verbose=False)[0]
            masks = getattr(res_g, "masks", None)
            if masks is not None and getattr(masks, "data", None) is not None and len(masks.data) > 0:
                mask = masks.data[0].cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                # 1. Create mask in crop coordinates
                gold_mask_crop = (mask > 0.5).astype(np.uint8) * 255

                # 2. Create full-size mask with correct position
                gold_mask_full = np.zeros((H, W), dtype=np.uint8)

                # Paste the crop-sized mask into the correct location
                # (Important: resize mask back to the crop size first if needed!)
                crop_h, crop_w = cy2 - cy1, cx2 - cx1

                if gold_mask_crop.shape[:2] != (crop_h, crop_w):
                    gold_mask_crop = cv2.resize(
                        gold_mask_crop,
                        (crop_w, crop_h),
                        interpolation=cv2.INTER_NEAREST
                    )

                gold_mask_full[cy1:cy2, cx1:cx2] = gold_mask_crop

                # 3. Optional: clip only inside stone bounding box (your existing logic)
                stone_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.rectangle(stone_mask, (sx1, sy1), (sx2, sy2), 255, -1)

                gold_clipped_full = cv2.bitwise_and(gold_mask_full, stone_mask)
                annotated[gold_clipped_full > 0] = GOLD_OVERLAY_COLOR

                stone_area = max(1, (sx2 - sx1) * (sy2 - sy1))
                gold_mask_pct = (gold_clipped_full > 0).sum() / stone_area * 100
            else:
                # No masks returned for this frame
                print("[Gold] no masks detected in this frame")
        except Exception as e:
            print(f"Gold detection error: {e}")

    return annotated, {
        'mask': gold_clipped_full,
        'mask_pct': gold_mask_pct,
        'stone_bbox': stone_bbox
    }

def compute_rubbing(annotated, gold_info):
    global STATE

    if not gold_info or gold_info['stone_bbox'] is None:
        return annotated, False

    mask = gold_info['mask']
    if (mask > 0).sum() == 0:
        return annotated, False

    M = cv2.moments(mask)
    if M['m00'] == 0:
        return annotated, False

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.circle(annotated, (cx, cy), 5, (0,0,255), -1)

    sx1, sy1, sx2, sy2 = gold_info['stone_bbox']
    scx = (sx1 + sx2) / 2
    scy = (sy1 + sy2) / 2
    dist = np.hypot(cx - scx, cy - scy)

    STATE["recent_distances"].append(dist)

    rubbing = False
    if len(STATE["recent_distances"]) >= 3:
        diffs = np.diff(list(STATE["recent_distances"]))
        meaningful = np.abs(diffs) >= THRESHOLD_FLUCTUATION
        signs = np.sign(diffs)
        sign_changes = 0
        prev_sign = signs[0]
        for i in range(1, len(signs)):
            s = signs[i]
            if meaningful[i] and meaningful[i-1] and s != 0 and prev_sign != 0 and s != prev_sign:
                sign_changes += 1
            prev_sign = s if s != 0 else prev_sign
        rubbing = sign_changes >= NO_OF_FLUCTUATIONS

    return annotated, rubbing

def process_acid_frame(frame, model_acid):
    annotated = frame.copy()
    acid_detected = False

    try:
        results = model_acid(frame, imgsz=IMGSZ, conf=0.8, verbose=False)[0]
        if len(results.boxes) > 0:
            for box in results.boxes:
                conf = box.conf[0].item()
                if conf > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,255), 3)
                    cv2.putText(annotated, f"Acid {conf:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
                    acid_detected = True
    except Exception as e:
        print(f"Acid detection error: {e}")

    return annotated, acid_detected

# ====================== LOAD MODELS (once at startup) ======================
logger.info("Loading YOLO models...")
model_stone = YOLO(MODEL_STONE)
model_gold = YOLO(MODEL_GOLD)
model_acid = YOLO(MODEL_ACID)
logger.info("YOLO models loaded.")

# Initialize WebRTC audio worker
init_worker()

# WebRTC peer connections
pcs = set()

# ====================== WEBRTC AUDIO TRACK ======================
class AudioAnalysisTrack(MediaStreamTrack):
    """
    A MediaStreamTrack that intercepts audio frames from frontend,
    resamples them to 16kHz mono, and sends data to inference worker.
    """
    kind = "audio"

    def __init__(self, track, data_channel=None):
        super().__init__()
        self.track = track
        self.data_channel = data_channel
        # Resample to 16kHz mono float32 for the model
        self.resampler = av.AudioResampler(format='flt', layout='mono', rate=16000)

    async def recv(self):
        try:
            frame = await self.track.recv()
        except Exception:
            # Track ended
            self.stop()
            raise

        # Process audio for inference
        await self._process_frame(frame)

        # Return the original frame to keep the stream flowing
        return frame

    async def _process_frame(self, frame):
        """Process audio frame and send prediction via DataChannel"""
        if worker is None:
            return
            
        # Resample audio frame
        out_frames = self.resampler.resample(frame)
        for out_frame in out_frames:
            # Convert to numpy: to_ndarray returns (channels, samples), so (1, N) for mono
            chunk = out_frame.to_ndarray()[0]
            
            # Feed to worker
            label, conf = worker.process_chunk(chunk)
            
            if label != "WAIT":
                # Send result via DataChannel
                if self.data_channel and self.data_channel.readyState == "open":
                    msg = json.dumps({"label": label, "confidence": conf})
                    self.data_channel.send(msg)
                    logger.info(f"Audio prediction: {label} (conf={conf:.3f}, threshold=0.75) ✓")
            elif conf > 0:
                # Prediction made but below confidence threshold
                logger.debug(f"Audio prediction filtered: conf={conf:.3f} < 0.75 threshold")

# ====================== WEBRTC ENDPOINTS ======================
@app.post("/offer")
async def offer(request: Request):
    """WebRTC signaling endpoint - receives offer from frontend"""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    logger.info(f"Created peer connection {id(pc)}")

    # Container to share state between handlers
    state = {"channel": None}

    @pc.on("datachannel")
    def on_datachannel(channel):
        state["channel"] = channel
        logger.info(f"DataChannel received: {channel.label}")
        
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track received: {track.kind}")
        if track.kind == "audio":
            # Create local track that consumes the remote track and performs inference
            local_track = AudioAnalysisTrack(track, data_channel=state.get("channel"))
            
            # Use MediaBlackhole to consume the track
            recorder = MediaBlackhole() 
            recorder.addTrack(local_track)
            
            async def run_recorder():
                await recorder.start()
                
            asyncio.ensure_future(run_recorder())
            
            # Late binding for data channel if not ready yet
            if not state["channel"]:
                @pc.on("datachannel")
                def on_datachannel_late(channel):
                    state["channel"] = channel
                    local_track.data_channel = channel
                    logger.info("DataChannel attached to track (late)")

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

@app.on_event("shutdown")
async def on_shutdown():
    """Close all WebRTC connections on shutdown"""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if worker is not None else "degraded",
        "model": "loaded" if worker is not None else "failed",
        "worker_ready": worker is not None
    }

# ====================== API ENDPOINT ======================
@app.post("/process")
async def process_frame(frame: UploadFile = File(...)):
    global STATE

    # Read uploaded frame
    contents = await frame.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    annotated = img.copy()
    visual_ok = False
    acid_detected = False

    if not STATE["rubbing_done"]:
        # Stage 1: Rubbing
        annotated, info = process_rubbing_frame(img, model_stone, model_gold)
        annotated, is_rubbing = compute_rubbing(annotated, info)

        visual_ok = is_rubbing
        if visual_ok and STATE["sound_status"] == "OK":
            STATE["rubbing_done"] = True
            STATE["stage"] = "ACID"

        # Visual feedback
        cv2.putText(annotated, "STAGE 1: GOLD RUBBING", (10, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
        status_text = "Visual: OK" if visual_ok else "Visual: NOT OK"
        color = (0,255,0) if visual_ok else (0,0,255)
        cv2.putText(annotated, status_text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(annotated, f"Sound: {STATE['sound_status']}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        if STATE["rubbing_done"]:
            cv2.putText(annotated, "RUBBING CONFIRMED! → ACID TEST", (50, 240),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,255), 4)

    else:
        # Stage 2: Acid
        annotated, acid_detected = process_acid_frame(img, model_acid)
        cv2.putText(annotated, "STAGE 2: ACID DETECTION", (10, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,255), 2)

    # Encode result image
    _, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse({
        "image": jpg_as_text,
        "stage": STATE["stage"],
        "visual_ok": visual_ok,
        "sound_status": STATE["sound_status"],
        "acid_detected": acid_detected,
        "rubbing_done": STATE["rubbing_done"]
    })


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server with WebRTC audio support...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)