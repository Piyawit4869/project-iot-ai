# print("เริ่มโปรแกรมแล้ว...")

# import cv2

# rtsp_url = "rtsp://YaHPwvyz:fGY7os86XYWKNNMA@192.168.1.39:554/live/ch0"
# cap = cv2.VideoCapture(rtsp_url)

# if not cap.isOpened():
#     print("เปิดกล้องไม่สำเร็จ")
#     exit()

# print("เปิดกล้องสำเร็จ! กำลังแสดงภาพ...")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("ดึงภาพไม่ได้")
#         break

#     # ⭐ ย่อขนาด เช่น เหลือ 640x360
#     frame_small = cv2.resize(frame, (1280, 720))

#     cv2.imshow("IP Camera Stream", frame_small)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# from flask import Flask, Response
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# rtsp_url = "rtsp://YaHPwvyz:fGY7os86XYWKNNMA@192.168.1.39:554/live/ch0"

# def generate_frames():
#     cap = cv2.VideoCapture(rtsp_url)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# @app.route("/video_feed")
# def video_feed():
#     return Response(generate_frames(),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)

import cv2
import time
import threading
import requests
from ultralytics import YOLO

ESP_IP = "http://192.168.1.40"  
ON_URL = f"{ESP_IP}/on"
MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.6      
DETECT_FRAMES = 3         

rtsp_url = "rtsp://YaHPwvyz:fGY7os86XYWKNNMA@192.168.1.39:554/live/ch0"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(rtsp_url)

latest_frame = None
annotated_frame = None

detect_status = "ไม่เจอ"
last_sent_status = None
detect_count = 0
running = True

def yolo_thread():
    global latest_frame, annotated_frame
    global detect_status, last_sent_status, detect_count, running

    while running:
        if latest_frame is None:
            time.sleep(0.01)
            continue

        frame_small = cv2.resize(latest_frame, (640, 640))
        results = model(frame_small, verbose=False)
        boxes = results[0].boxes

        found = False
        for box in boxes:
            if box.conf[0] >= CONF_THRESHOLD:
                found = True
                break

        if found:
            detect_count += 1
        else:
            detect_count = 0

        if detect_count >= DETECT_FRAMES:
            detect_status = "เจอ"
        else:
            detect_status = "ไม่เจอ"

        if detect_status != last_sent_status:
            try:
                if detect_status == "เจอ":
                    requests.get(ON_URL, timeout=1)
                    print("📡 ส่ง HTTP: ON")

                last_sent_status = detect_status

            except Exception as e:
                print("ส่งไป ESP32 ไม่ได้:", e)

        annotated_small = results[0].plot()
        annotated_frame = cv2.resize(annotated_small, (1280, 720))

        time.sleep(0.01)

thread = threading.Thread(target=yolo_thread, daemon=True)
thread.start()

print("อ่านภาพจากกล้อง")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ดึงภาพไม่ได้")
        break

    latest_frame = frame.copy()

    if annotated_frame is not None:
        cv2.imshow("YOLO Stream", annotated_frame)
    else:
        cv2.imshow("YOLO Stream", cv2.resize(frame, (1280, 720)))

    print(f"สถานะ: {detect_status} | detect_count: {detect_count}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cap.release()
cv2.destroyAllWindows()
print("⛔ ปิดโปรแกรม")




