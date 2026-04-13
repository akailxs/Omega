import asyncio
import math
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Cortex Brain")

# Directory of the vault is defined in .env, fallback to current dir
vault_path_env = os.environ.get("OBSIDIAN_DIR", str(Path(__file__).parent.resolve()))
VAULT_DIR = Path(vault_path_env).resolve()
LOW_PASS_ALPHA = float(os.environ.get("CORTEX_LOW_PASS_ALPHA", "0.24"))
INERTIA_FRICTION = float(os.environ.get("CORTEX_INERTIA_FRICTION", "0.84"))
INERTIA_RESPONSE = float(os.environ.get("CORTEX_INERTIA_RESPONSE", "0.18"))
MASTER_HAND_PREFERENCE = os.environ.get("CORTEX_MASTER_HAND", "Right")
HAND_STICKINESS_SECONDS = float(os.environ.get("CORTEX_HAND_STICKINESS_SECONDS", "0.35"))
MAX_TRACKED_HANDS = 2


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def wrap_angle(angle_radians: float) -> float:
    while angle_radians <= -math.pi:
        angle_radians += 2 * math.pi
    while angle_radians > math.pi:
        angle_radians -= 2 * math.pi
    return angle_radians


@dataclass
class HandObservation:
    x: float
    y: float
    label: str = "Unknown"
    confidence: float = 0.0


class InertialLowPass2D:
    def __init__(
        self,
        alpha: float = LOW_PASS_ALPHA,
        friction: float = INERTIA_FRICTION,
        response: float = INERTIA_RESPONSE,
    ):
        self.alpha = alpha
        self.friction = friction
        self.response = response
        self.initialized = False
        self.filtered_x = 0.0
        self.filtered_y = 0.0
        self.output_x = 0.0
        self.output_y = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0

    def reset(self):
        self.initialized = False
        self.filtered_x = 0.0
        self.filtered_y = 0.0
        self.output_x = 0.0
        self.output_y = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0

    def update(self, raw_x: float, raw_y: float) -> tuple[float, float]:
        if not self.initialized:
            self.filtered_x = self.output_x = raw_x
            self.filtered_y = self.output_y = raw_y
            self.initialized = True
            return self.output_x, self.output_y

        self.filtered_x += (raw_x - self.filtered_x) * self.alpha
        self.filtered_y += (raw_y - self.filtered_y) * self.alpha

        dx = self.filtered_x - self.output_x
        dy = self.filtered_y - self.output_y

        self.velocity_x = (self.velocity_x * self.friction) + (dx * self.response)
        self.velocity_y = (self.velocity_y * self.friction) + (dy * self.response)

        if abs(self.velocity_x) < 0.001:
            self.velocity_x = 0.0
        if abs(self.velocity_y) < 0.001:
            self.velocity_y = 0.0

        self.output_x += self.velocity_x
        self.output_y += self.velocity_y
        return self.output_x, self.output_y


class InertialAngleFilter:
    def __init__(
        self,
        alpha: float = LOW_PASS_ALPHA,
        friction: float = INERTIA_FRICTION,
        response: float = INERTIA_RESPONSE,
    ):
        self.alpha = alpha
        self.friction = friction
        self.response = response
        self.initialized = False
        self.filtered = 0.0
        self.output = 0.0
        self.velocity = 0.0
        self.last_output = 0.0

    def reset(self):
        self.initialized = False
        self.filtered = 0.0
        self.output = 0.0
        self.velocity = 0.0
        self.last_output = 0.0

    def update(self, raw_angle: float) -> tuple[float, float]:
        if not self.initialized:
            normalized = wrap_angle(raw_angle)
            self.filtered = normalized
            self.output = normalized
            self.last_output = normalized
            self.initialized = True
            return self.output, 0.0

        filtered_delta = wrap_angle(raw_angle - self.filtered)
        self.filtered = wrap_angle(self.filtered + (filtered_delta * self.alpha))

        output_delta = wrap_angle(self.filtered - self.output)
        self.velocity = (self.velocity * self.friction) + (output_delta * self.response)
        if abs(self.velocity) < 0.0001:
            self.velocity = 0.0

        self.last_output = self.output
        self.output = wrap_angle(self.output + self.velocity)
        delta = wrap_angle(self.output - self.last_output)
        return self.output, delta


class GestureStreamProcessor:
    def __init__(self):
        self.master_smoother = InertialLowPass2D()
        self.secondary_smoother = InertialLowPass2D()
        self.rotation_filter = InertialAngleFilter()
        self.last_master_label: str | None = None
        self.last_master_seen_at = 0.0
        self.last_secondary_active = False

    def process(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        hands = self._extract_hands(payload)
        if not hands:
            self.master_smoother.reset()
            self.secondary_smoother.reset()
            self.rotation_filter.reset()
            self.last_master_label = None
            self.last_master_seen_at = 0.0
            self.last_secondary_active = False
            return None

        master_hand, secondary_hand = self._select_hands(hands)
        smoothed_x, smoothed_y = self.master_smoother.update(master_hand.x, master_hand.y)
        secondary_coords: tuple[float, float] | None = None
        if secondary_hand is not None:
            secondary_coords = self.secondary_smoother.update(
                secondary_hand.x,
                secondary_hand.y,
            )

        hands_count = 2 if secondary_hand is not None else 1
        rotation_angle, rotation_delta, tilt_angle = self._compute_rotation(
            smoothed_x,
            smoothed_y,
            secondary_coords,
        )

        message = {
            "type": "gesture",
            "action": "TRACK",
            "source": "python",
            "master_hand": master_hand.label,
            "hands_count": hands_count,
            "x": round(smoothed_x, 2),
            "y": round(smoothed_y, 2),
            "rotation": {
                "yaw_deg": round(math.degrees(rotation_angle), 2),
                "tilt_deg": round(math.degrees(tilt_angle), 2),
                "delta_deg": round(math.degrees(rotation_delta), 2),
            },
        }

        if secondary_hand is not None:
            secondary_x, secondary_y = secondary_coords or (secondary_hand.x, secondary_hand.y)
            message["secondary"] = {
                "x": round(secondary_x, 2),
                "y": round(secondary_y, 2),
                "label": secondary_hand.label,
            }
            self.last_secondary_active = True
        elif self.last_secondary_active:
            self.secondary_smoother.reset()
            self.last_secondary_active = False

        return message

    def _extract_hands(self, payload: dict[str, Any]) -> list[HandObservation]:
        hands_payload = payload.get("hands")
        if isinstance(hands_payload, list):
            hands: list[HandObservation] = []
            for hand_payload in hands_payload[:MAX_TRACKED_HANDS]:
                hand = self._build_hand_observation(hand_payload, payload)
                if hand is not None:
                    hands.append(hand)
            if hands:
                return hands

        legacy_hand = self._extract_legacy_hand(payload)
        return [legacy_hand] if legacy_hand is not None else []

    def _build_hand_observation(
        self,
        hand_payload: Any,
        root_payload: dict[str, Any],
    ) -> HandObservation | None:
        if not isinstance(hand_payload, dict):
            return None

        label = (
            hand_payload.get("label")
            or hand_payload.get("handedness")
            or hand_payload.get("hand")
            or "Unknown"
        )
        confidence = float(hand_payload.get("confidence", hand_payload.get("score", 0.0)) or 0.0)

        x = hand_payload.get("x")
        y = hand_payload.get("y")

        if x is None or y is None:
            index_point = hand_payload.get("index") or hand_payload.get("landmark") or {}
            if isinstance(index_point, dict):
                x = index_point.get("x")
                y = index_point.get("y")

        if x is None or y is None:
            return None

        norm_width = root_payload.get("frame_width") or root_payload.get("width")
        norm_height = root_payload.get("frame_height") or root_payload.get("height")
        raw_x = float(x)
        raw_y = float(y)

        if 0.0 <= raw_x <= 1.0 and 0.0 <= raw_y <= 1.0 and norm_width and norm_height:
            raw_x *= float(norm_width)
            raw_y *= float(norm_height)

        return HandObservation(x=raw_x, y=raw_y, label=str(label), confidence=confidence)

    def _extract_legacy_hand(self, payload: dict[str, Any]) -> HandObservation | None:
        raw_x = payload.get("master_x", payload.get("x"))
        raw_y = payload.get("master_y", payload.get("y"))
        if raw_x is None or raw_y is None:
            return None

        label = payload.get("master_hand") or payload.get("label") or "Unknown"
        confidence = float(payload.get("confidence", 1.0))
        return HandObservation(
            x=float(raw_x),
            y=float(raw_y),
            label=str(label),
            confidence=confidence,
        )

    def _select_hands(
        self,
        hands: list[HandObservation],
    ) -> tuple[HandObservation, HandObservation | None]:
        if len(hands) == 1:
            master = hands[0]
            self.last_master_label = master.label
            self.last_master_seen_at = time.monotonic()
            return master, None

        now = time.monotonic()
        preferred = next(
            (hand for hand in hands if hand.label == MASTER_HAND_PREFERENCE),
            None,
        )
        sticky = None
        if (
            self.last_master_label is not None
            and (now - self.last_master_seen_at) <= HAND_STICKINESS_SECONDS
        ):
            sticky = next(
                (hand for hand in hands if hand.label == self.last_master_label),
                None,
            )

        master = sticky or preferred or max(hands, key=lambda hand: hand.confidence)
        secondary_candidates = [hand for hand in hands if hand is not master]
        secondary = (
            max(secondary_candidates, key=lambda hand: hand.confidence)
            if secondary_candidates
            else None
        )

        self.last_master_label = master.label
        self.last_master_seen_at = now
        return master, secondary

    def _compute_rotation(
        self,
        smoothed_x: float,
        smoothed_y: float,
        secondary_coords: tuple[float, float] | None,
    ) -> tuple[float, float, float]:
        if secondary_coords is not None:
            secondary_x, secondary_y = secondary_coords
            raw_angle = math.atan2(secondary_y - smoothed_y, secondary_x - smoothed_x)
            distance = math.hypot(secondary_x - smoothed_x, secondary_y - smoothed_y)
            tilt = math.asin(clamp((secondary_y - smoothed_y) / max(distance, 1.0), -1.0, 1.0))
            angle, delta = self.rotation_filter.update(raw_angle)
            return angle, delta, tilt

        velocity_angle = math.atan2(
            self.master_smoother.velocity_y,
            self.master_smoother.velocity_x,
        )
        angle, delta = self.rotation_filter.update(velocity_angle)
        return angle, delta, 0.0


def extract_tags(content: str):
    # Match #tag format
    tags = re.findall(r'(#[a-zA-Z0-9_À-ÿ\-]+)', content)
    return list(set(tags))

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"Client connected. Active clients: {len(self.active_connections)}")
        await self.send_initial_state(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"Client disconnected. Active clients: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Failed to send to client: {e}")
                self.disconnect(connection)

    async def send_initial_state(self, websocket: WebSocket):
        print("Sending initial state to new client...")
        for root, dirs, files in os.walk(VAULT_DIR):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if file.endswith('.md'):
                    filepath = Path(root) / file
                    try:
                        content = filepath.read_text(encoding="utf-8")
                        rel_path = filepath.relative_to(VAULT_DIR)
                        tags = extract_tags(content)
                        msg = {
                            "action": "created",
                            "filename": file,
                            "path": str(rel_path),
                            "tags": tags
                        }
                        await websocket.send_text(json.dumps(msg))
                    except Exception as e:
                        print(f"Failed to read initial state of {filepath}: {e}")
        print("Initial state completely sent.")

manager = ConnectionManager()
loop = None

class MarkdownHandler(FileSystemEventHandler):
    def on_modified(self, event):
        self._process_event("modified", event.src_path, event.is_directory)

    def on_created(self, event):
        self._process_event("created", event.src_path, event.is_directory)

    def on_deleted(self, event):
        self._process_event("deleted", event.src_path, event.is_directory)

    def on_moved(self, event):
        self._process_event("deleted", event.src_path, event.is_directory)
        self._process_event("created", event.dest_path, event.is_directory)

    def _process_event(self, action: str, src_path: str, is_directory: bool):
        if is_directory or not src_path.endswith('.md'):
            return

        try:
            rel_path = Path(src_path).relative_to(VAULT_DIR)
        except ValueError:
            return

        if any(part.startswith('.') for part in rel_path.parts):
            return

        if loop is not None:
            asyncio.run_coroutine_threadsafe(self.broadcast_event(action, src_path, str(rel_path)), loop)

    async def broadcast_event(self, action: str, src_path: str, rel_path: str):
        path = Path(src_path)
        tags = []
        filename = path.name

        if action != "deleted":
            try:
                await asyncio.sleep(0.05)
                if path.exists():
                    content = path.read_text(encoding="utf-8")
                    tags = extract_tags(content)
                else:
                    return
            except Exception as e:
                print(f"Failed to read {src_path} on {action}: {e}")
                return

        msg = {
            "action": action,
            "filename": filename,
            "path": rel_path,
            "tags": tags
        }
        print(f"Broadcasting via WebSocket: {action} -> {rel_path} | Tags: {tags}")
        await manager.broadcast(json.dumps(msg))

@app.on_event("startup")
async def startup_event():
    global loop
    loop = asyncio.get_running_loop()

    event_handler = MarkdownHandler()
    observer = Observer()
    observer.schedule(event_handler, path=str(VAULT_DIR), recursive=True)
    observer.start()
    app.state.observer = observer
    print(f"Started monitoring Vault: {VAULT_DIR}")

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, "observer"):
        app.state.observer.stop()
        app.state.observer.join()
        print("Observer stopped.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    gesture_processor = GestureStreamProcessor()
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue

            if not isinstance(payload, dict):
                continue

            if payload.get("type") not in {"gesture", "gesture_raw"} and "hands" not in payload:
                continue

            processed = gesture_processor.process(payload)
            if processed is not None:
                await manager.broadcast(json.dumps(processed))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket execution error: {e}")
        manager.disconnect(websocket)

# Servez le répertoire entier pour avoir accès à mind_map.html, cortex_dashboard.html, mind_map.json
app.mount("/", StaticFiles(directory=VAULT_DIR, html=True), name="static")
