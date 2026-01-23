import os
import time
import threading
import queue
import json
import csv
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import cv2
import requests

from fastapi import FastAPI, Response, HTTPException, Header, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse

from insightface.app import FaceAnalysis

# =========================
# CONFIG (ENV)
# =========================
API_TOKEN = os.getenv("AGENT_TOKEN", "supersecret")
FACES_DIR = Path(os.getenv("FACES_DIR", "faces"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = DATA_DIR / "config.json"
DB_PATH = DATA_DIR / "access.db"
CSV_PATH = DATA_DIR / "access_log.csv"

ESP32CAM_IP = os.getenv("ESP32CAM_IP", "192.168.1.201")
ESP32CAM_STREAM_PORT = int(os.getenv("ESP32CAM_STREAM_PORT", "81"))
ESP32CAM_STREAM_PATH = os.getenv("ESP32CAM_STREAM_PATH", "/stream")

ESP32CTRL_IP = os.getenv("ESP32CTRL_IP", "192.168.1.202")
ESP32CTRL_PORT = int(os.getenv("ESP32CTRL_PORT", "80"))
ESP32CTRL_UNLOCK_PATH = os.getenv("ESP32CTRL_UNLOCK_PATH", "/unlock")

ESP32CAM_STREAM_URL = f"http://{ESP32CAM_IP}:{ESP32CAM_STREAM_PORT}{ESP32CAM_STREAM_PATH}"
ESP32_UNLOCK_URL = f"http://{ESP32CTRL_IP}:{ESP32CTRL_PORT}{ESP32CTRL_UNLOCK_PATH}"

# =========================
# DEFAULT RUNTIME SETTINGS (จะถูก override ด้วย config.json)
# =========================
DEFAULTS = {
    "VERIFY_TIME": 4.0,
    "SIM_THRESHOLD": 0.35,
    "TRIGGER_COOLDOWN": 5.0,
    "DISPLAY_WIDTH": 640,
    "FACE_DET_WIDTH": 320,
    "FACE_INFER_EVERY_N": 6,
    "COLLECT_HZ": 2.0,
    "PREVIEW_FPS": 8.0,

    # face trigger (แทนชูมือ)
    "FACE_TRIGGER_ENABLED": True,
    "FACE_PRESENT_SEC": 0.8,
    "FACE_CHECK_HZ": 6.0,
    "FACE_MIN_AREA": 0.015,
}

def safe_now():
    return time.time()

def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            return {**DEFAULTS, **json.loads(CONFIG_PATH.read_text(encoding="utf-8"))}
        except Exception:
            return DEFAULTS.copy()
    return DEFAULTS.copy()

def save_config(cfg: Dict[str, Any]):
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

CFG = load_config()

# =========================
# FASTAPI
# =========================
app = FastAPI(title="ESP32 Face Agent (Full)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def auth(token: Optional[str]):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

def pick_token(token: Optional[str], header_token: Optional[str]):
    return token or header_token

# =========================
# STATE
# =========================
state_lock = threading.Lock()
latest_frame = None
latest_jpeg = None

status = {
    "agent": "ok",
    "mode": "idle",
    "stream_url": ESP32CAM_STREAM_URL,
    "unlock_url": ESP32_UNLOCK_URL,
    "known_count": 0,
    "last_result": None,
    "last_error": None,
    "server_started_ts": safe_now(),
    "config": {},
    "face_trigger": {
        "last_face_ts": None,
        "face_hold_sec": 0.0,
        "last_trigger_ts": None,
    }
}

events_q = queue.Queue(maxsize=300)

def push_event(evt: dict):
    evt["ts"] = safe_now()
    try:
        events_q.put_nowait(evt)
    except Exception:
        pass

# =========================
# DB / CSV LOGGING
# =========================
def db_init():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS access_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        ok INTEGER NOT NULL,
        name TEXT,
        sim REAL,
        reason TEXT,
        trigger TEXT,
        unlock_ok INTEGER
    )
    """)
    conn.commit()
    conn.close()

def log_access(ok: bool, name: Optional[str], sim: float, reason: Optional[str], trigger: str, unlock_ok: Optional[bool]):
    ts = safe_now()
    db_init()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO access_log (ts, ok, name, sim, reason, trigger, unlock_ok) VALUES (?,?,?,?,?,?,?)",
        (ts, 1 if ok else 0, name, float(sim), reason, trigger, 1 if unlock_ok else 0 if unlock_ok is not None else None)
    )
    conn.commit()
    conn.close()

    # CSV append (สร้าง header ถ้ายังไม่มี)
    is_new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["ts", "ok", "name", "sim", "reason", "trigger", "unlock_ok"])
        w.writerow([ts, int(ok), name or "", float(sim), reason or "", trigger, "" if unlock_ok is None else int(unlock_ok)])

# =========================
# UTIL
# =========================
def resize_keep_aspect(frame, target_w):
    h, w = frame.shape[:2]
    if w == target_w:
        return frame
    scale = target_w / float(w)
    new_h = int(h * scale)
    return cv2.resize(frame, (target_w, new_h), interpolation=cv2.INTER_AREA)

def cosine_match(known_embeds, known_names, emb):
    sims = known_embeds @ emb
    best_idx = int(np.argmax(sims))
    return known_names[best_idx], float(sims[best_idx])

# =========================
# FACE MODEL + DB
# =========================
print("[AGENT] Initializing InsightFace...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(int(CFG["FACE_DET_WIDTH"]), int(CFG["FACE_DET_WIDTH"])))

def load_known_faces():
    embeds, names = [], []
    FACES_DIR.mkdir(parents=True, exist_ok=True)

    def add_face(img_path, person_name):
        img = cv2.imread(str(img_path))
        if img is None:
            return
        faces = face_app.get(img)
        if not faces:
            return
        embeds.append(faces[0].normed_embedding)
        names.append(person_name)

    for item in FACES_DIR.iterdir():
        if item.is_dir():
            person = item.name
            for f in item.iterdir():
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    add_face(f, person)
        else:
            if item.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                add_face(item, item.stem)

    if not embeds:
        return np.zeros((0, 512), dtype=np.float32), []

    return np.asarray(embeds, dtype=np.float32), names

known_embeds, known_names = load_known_faces()
with state_lock:
    status["known_count"] = len(known_names)
    status["config"] = CFG
print("[AGENT] Known faces:", len(known_names))

# =========================
# CAMERA READER THREAD
# =========================
stop_flag = False

def open_capture(url: str):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap

def camera_reader():
    global latest_frame, latest_jpeg

    display_w = int(CFG["DISPLAY_WIDTH"])
    jpeg_interval = 1.0 / max(1.0, float(CFG["PREVIEW_FPS"]))
    last_jpeg_ts = 0.0

    while not stop_flag:
        cap = open_capture(ESP32CAM_STREAM_URL)

        t0 = time.time()
        ok_open = False
        while time.time() - t0 < 2.0:
            if cap.isOpened():
                ok_open = True
                break
            time.sleep(0.05)

        if not ok_open:
            with state_lock:
                status["last_error"] = f"Cannot open stream: {ESP32CAM_STREAM_URL}"
            push_event({"type": "error", "message": "Cannot open stream"})
            try:
                cap.release()
            except Exception:
                pass
            time.sleep(1.0)
            continue

        fail_count = 0
        push_event({"type": "info", "message": "Stream connected"})

        while not stop_flag:
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                if fail_count >= 30:
                    push_event({"type": "warn", "message": "Stream lost -> reconnect"})
                    break
                time.sleep(0.02)
                continue

            fail_count = 0
            frame = resize_keep_aspect(frame, display_w)

            with state_lock:
                latest_frame = frame

            now = time.time()
            if now - last_jpeg_ts >= jpeg_interval:
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ok:
                    with state_lock:
                        latest_jpeg = buf.tobytes()
                last_jpeg_ts = now

        try:
            cap.release()
        except Exception:
            pass
        time.sleep(0.2)

threading.Thread(target=camera_reader, daemon=True).start()

# =========================
# VERIFY / UNLOCK
# =========================
verify_lock = threading.Lock()
last_trigger_ts = 0.0

def do_unlock(name: str, ms: int = 3000) -> bool:
    try:
        requests.get(ESP32_UNLOCK_URL, params={"name": name, "ms": ms}, timeout=2.0)
        return True
    except Exception as e:
        with state_lock:
            status["last_error"] = f"unlock failed: {e}"
        push_event({"type": "error", "message": f"unlock failed: {e}"})
        return False

def verify_once(trigger: str = "manual"):
    global last_trigger_ts, known_embeds, known_names, CFG

    now = safe_now()
    if now - last_trigger_ts < float(CFG["TRIGGER_COOLDOWN"]):
        return

    with verify_lock:
        with state_lock:
            status["mode"] = "verifying"

        # reload config each verify (เผื่อปรับจากเว็บ)
        CFG = load_config()
        with state_lock:
            status["config"] = CFG

        # reload known faces each verify (เผื่อเพิ่มคนผ่านเว็บ)
        known_embeds, known_names = load_known_faces()
        with state_lock:
            status["known_count"] = len(known_names)

        if len(known_names) == 0:
            res = {"ok": False, "reason": "no_db", "name": None, "sim": 0.0, "trigger": trigger}
            with state_lock:
                status["mode"] = "idle"
                status["last_result"] = {**res, "ts": safe_now()}
            push_event({"type": "verify", **res})
            log_access(False, None, 0.0, "no_db", trigger, None)
            return

        verify_time = float(CFG["VERIFY_TIME"])
        face_det_w = int(CFG["FACE_DET_WIDTH"])
        infer_every = int(CFG["FACE_INFER_EVERY_N"])
        collect_hz = float(CFG["COLLECT_HZ"])
        sim_th = float(CFG["SIM_THRESHOLD"])

        collected = []
        t0 = safe_now()
        last_collect = 0.0
        frame_id = 0

        while safe_now() - t0 < verify_time:
            with state_lock:
                f = None if latest_frame is None else latest_frame.copy()
            if f is None:
                time.sleep(0.02)
                continue

            frame_id += 1
            if frame_id % infer_every != 0:
                time.sleep(0.005)
                continue

            small = resize_keep_aspect(f, face_det_w)
            faces = face_app.get(small)
            if faces:
                main = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                if safe_now() - last_collect >= (1.0 / collect_hz):
                    collected.append(main.normed_embedding)
                    last_collect = safe_now()

            time.sleep(0.005)

        with state_lock:
            status["mode"] = "idle"

        if not collected:
            res = {"ok": False, "reason": "no_face", "name": None, "sim": 0.0, "trigger": trigger}
            with state_lock:
                status["last_result"] = {**res, "ts": safe_now()}
            push_event({"type": "verify", **res})
            log_access(False, None, 0.0, "no_face", trigger, None)
            return

        avg = np.mean(collected, axis=0)
        avg = avg / np.linalg.norm(avg)

        best_name, best_sim = cosine_match(known_embeds, known_names, avg)

        if best_sim >= sim_th:
            unlock_ok = do_unlock(best_name, 3000)
            last_trigger_ts = safe_now()
            with state_lock:
                status["face_trigger"]["last_trigger_ts"] = last_trigger_ts
            res = {"ok": True, "name": best_name, "sim": float(best_sim), "unlock": unlock_ok, "trigger": trigger}
            log_access(True, best_name, float(best_sim), None, trigger, unlock_ok)
        else:
            res = {"ok": False, "name": best_name, "sim": float(best_sim), "reason": "low_sim", "trigger": trigger}
            log_access(False, best_name, float(best_sim), "low_sim", trigger, None)

        with state_lock:
            status["last_result"] = {**res, "ts": safe_now()}
        push_event({"type": "verify", **res})

# =========================
# FACE-PRESENT TRIGGER LOOP
# =========================
def face_trigger_loop():
    global CFG
    hold = 0.0
    last_check = safe_now()

    while not stop_flag:
        CFG = load_config()
        enabled = bool(CFG["FACE_TRIGGER_ENABLED"])
        hz = float(CFG["FACE_CHECK_HZ"])
        interval = 1.0 / max(1.0, hz)

        time.sleep(interval)

        with state_lock:
            mode = status["mode"]
            f = None if latest_frame is None else latest_frame.copy()

        if not enabled or mode != "idle" or f is None:
            hold = 0.0
            last_check = safe_now()
            continue

        if safe_now() - last_trigger_ts < float(CFG["TRIGGER_COOLDOWN"]):
            hold = 0.0
            last_check = safe_now()
            continue

        now = safe_now()
        dt = now - last_check
        last_check = now

        face_det_w = int(CFG["FACE_DET_WIDTH"])
        min_area = float(CFG["FACE_MIN_AREA"])
        need_sec = float(CFG["FACE_PRESENT_SEC"])

        small = resize_keep_aspect(f, face_det_w)
        faces = face_app.get(small)

        face_ok = False
        if faces:
            sh, sw = small.shape[:2]
            main = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            x1, y1, x2, y2 = main.bbox
            area = max(0.0, (x2 - x1) * (y2 - y1))
            frame_area = float(sw * sh)
            if frame_area > 0 and (area / frame_area) >= min_area:
                face_ok = True

        if face_ok:
            hold += dt
            with state_lock:
                status["face_trigger"]["last_face_ts"] = now
                status["face_trigger"]["face_hold_sec"] = round(hold, 2)
        else:
            hold = 0.0
            with state_lock:
                status["face_trigger"]["face_hold_sec"] = 0.0

        if hold >= need_sec:
            hold = 0.0
            push_event({"type": "face_trigger", "message": "Face present -> start verify"})
            threading.Thread(target=verify_once, args=("face_present",), daemon=True).start()

threading.Thread(target=face_trigger_loop, daemon=True).start()

# =========================
# API: BASIC
# =========================
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/status")
def get_status(
    token: Optional[str] = Query(default=None),
    x_agent_token: Optional[str] = Header(default=None),
):
    auth(pick_token(token, x_agent_token))
    with state_lock:
        return {**status, "server_ts": int(safe_now())}

@app.post("/verify")
def start_verify(
    token: Optional[str] = Query(default=None),
    x_agent_token: Optional[str] = Header(default=None),
):
    auth(pick_token(token, x_agent_token))
    threading.Thread(target=verify_once, args=("manual_api",), daemon=True).start()
    return {"ok": True, "started": True}

@app.get("/stream.jpg")
def stream_jpg(
    token: Optional[str] = Query(default=None),
    x_agent_token: Optional[str] = Header(default=None),
):
    auth(pick_token(token, x_agent_token))
    with state_lock:
        jpg = latest_jpeg
    if not jpg:
        raise HTTPException(503, "No frame yet")
    return Response(content=jpg, media_type="image/jpeg")

@app.get("/events")
def sse_events(
    token: Optional[str] = Query(default=None),
    x_agent_token: Optional[str] = Header(default=None),
):
    auth(pick_token(token, x_agent_token))

    def gen():
        while True:
            evt = events_q.get()
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

# =========================
# API: PEOPLE (faces/)
# =========================
@app.get("/faces/list")
def faces_list(
    token: Optional[str] = Query(default=None),
    x_agent_token: Optional[str] = Header(default=None),
):
    auth(pick_token(token, x_agent_token))
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    people = []
    for p in FACES_DIR.iterdir():
        if p.is_dir():
            imgs = [x for x in p.iterdir() if x.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            people.append({"name": p.name, "count": len(imgs)})
    people.sort(key=lambda x: x["name"].lower())
    return {"ok": True, "people": people}

@app.post("/faces/upload")
async def faces_upload(
    token: Optional[str] = Query(default=None),
    x_agent_token: Optional[str] = Header(default=None),
    name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    auth(pick_token(token, x_agent_token))
    safe_name = "".join([c for c in name.strip() if c.isalnum() or c in ["_", "-", " "] ]).strip()
    if not safe_name:
        raise HTTPException(400, "Invalid name")

    target_dir = FACES_DIR / safe_name
    target_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for uf in files:
        ext = Path(uf.filename or "").suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            continue
        content = await uf.read()
        if not content:
            continue
        out = target_dir / f"{int(time.time()*1000)}_{saved}{ext}"
        out.write_bytes(content)
        saved += 1

    push_event({"type": "people", "message": f"Uploaded {saved} file(s) for {safe_name}"})
    return {"ok": True, "saved": saved, "name": safe_name}

@app.delete("/faces/delete")
def faces_delete(
    token: Optional[str] = Query(default=None),
    x_agent_token: Optional[str] = Header(default=None),
    name: str = Query(...),
):
    auth(pick_token(token, x_agent_token))
    target_dir = FACES_DIR / name
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(404, "Not found")
    # ลบทั้งโฟลเดอร์
    for f in target_dir.iterdir():
        try:
            f.unlink()
        except Exception:
            pass
    try:
        target_dir.rmdir()
    except Exception:
        pass
    push_event({"type": "people", "message": f"Deleted person: {name}"})
    return {"ok": True}

# =========================
# API: SETTINGS (config.json)
# =========================
@app.get("/config")
def get_config(
    token: Optional[str] = Query(default=None),
    x_agent_token: Optional[str] = Header(default=None),
):
    auth(pick_token(token, x_agent_token))
    cfg = load_config()
    return {"ok": True, "config": cfg}

@app.post("/config")
def set_config(
    payload: Dict[str, Any],
    token: Optional[str] = Query(default=None),
    x_agent_token: Optional[str] = Header(default=None),
):
    auth(pick_token(token, x_agent_token))
    cfg = load_config()

    # อนุญาตเฉพาะ key ที่เรารู้จัก
    for k in DEFAULTS.keys():
        if k in payload:
            cfg[k] = payload[k]

    save_config(cfg)
    with state_lock:
        status["config"] = cfg
    push_event({"type": "config", "message": "Config updated"})
    return {"ok": True, "config": cfg}

# =========================
# API: LOGS
# =========================
@app.get("/logs")
def get_logs(
    token: Optional[str] = Query(default=None),
    x_agent_token: Optional[str] = Header(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
):
    auth(pick_token(token, x_agent_token))
    db_init()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, ts, ok, name, sim, reason, trigger, unlock_ok FROM access_log ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()

    items = []
    for r in rows:
        items.append({
            "id": r[0],
            "ts": r[1],
            "ok": bool(r[2]),
            "name": r[3],
            "sim": r[4],
            "reason": r[5],
            "trigger": r[6],
            "unlock_ok": None if r[7] is None else bool(r[7]),
        })
    return {"ok": True, "items": items}

@app.get("/logs.csv")
def download_csv(
    token: Optional[str] = Query(default=None),
    x_agent_token: Optional[str] = Header(default=None),
):
    auth(pick_token(token, x_agent_token))
    if not CSV_PATH.exists():
        # create empty file with header
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts", "ok", "name", "sim", "reason", "trigger", "unlock_ok"])
    return FileResponse(CSV_PATH, media_type="text/csv", filename="access_log.csv")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    print("[AGENT] Stream:", ESP32CAM_STREAM_URL)
    print("[AGENT] Unlock:", ESP32_UNLOCK_URL)
    print("[AGENT] Data dir:", DATA_DIR)
    uvicorn.run(app, host="0.0.0.0", port=9000)
