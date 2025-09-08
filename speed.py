import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import time
import csv
import os
import matplotlib.pyplot as plt

# ========== SETTINGS ==========
VIDEO_SOURCE = 'veh2.mp4'   # change to 0 for webcam OR RTSP URL
SPEED_LIMIT = 80            # km/h overspeed threshold
DISTANCE_BETWEEN_LINES = 10 # meters (adjust based on real-world setup)

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Load class names
with open("coco.txt", "r") as file:
    class_list = file.read().split("\n")

# Setup tracker
tracker = Tracker()

# Setup video capture
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Output video writer
fps = int(cap.get(cv2.CAP_PROP_FPS))
output = cv2.VideoWriter('output_tracking.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps, (1020, 500))

# Reference lines
cy1 = 322
cy2 = 368
offset = 6

# Tracking data
vh_down, vh_up = {}, {}
counter, counter1 = [], []

# Ensure required folders exist
os.makedirs("violations", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

count = 0

# Open CSV for writing
with open('car_details.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Car ID", "Direction", "Speed (Km/h)", "Timestamp"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for efficiency
        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 500))

        # Run YOLO detection
        results = model.predict(frame, verbose=False)  # suppress YOLO logs
        boxes = results[0].boxes.data
        px = pd.DataFrame(boxes).astype("float")

        detections = []
        for _, row in px.iterrows():
            x1, y1, x2, y2, _, class_id = map(int, row[:6])
            class_name = class_list[class_id]
            if 'car' in class_name:
                detections.append([x1, y1, x2, y2])

        # Update tracker
        bbox_id = tracker.update(detections)

        for x3, y3, x4, y4, obj_id in bbox_id:
            cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

            # Going Down
            if cy1 - offset < cy < cy1 + offset:
                vh_down[obj_id] = time.time()

            if obj_id in vh_down and cy2 - offset < cy < cy2 + offset:
                elapsed_time = time.time() - vh_down[obj_id]
                if obj_id not in counter:
                    counter.append(obj_id)
                    speed_ms = DISTANCE_BETWEEN_LINES / elapsed_time
                    speed_kmh = speed_ms * 3.6
                    timestamp = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())  # safer filename

                    if speed_kmh > SPEED_LIMIT:
                        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 3)
                        cv2.putText(frame, "OVERSPEED!", (x3, y3 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        if y3 < y4 and x3 < x4:
                            violation_img = frame[max(0, y3):y4, max(0, x3):x4]
                            if violation_img.size > 0:
                                snap_path = f"violations/car_{obj_id}_{timestamp}.jpg"
                                cv2.imwrite(snap_path, violation_img)
                                print(f"[INFO] Overspeed snapshot saved: {snap_path}")

                    writer.writerow([obj_id, "Down", int(speed_kmh), timestamp])
                    file.flush()

                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID:{obj_id} {int(speed_kmh)}Km/h",
                                (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Going Up
            if cy2 - offset < cy < cy2 + offset:
                vh_up[obj_id] = time.time()

            if obj_id in vh_up and cy1 - offset < cy < cy1 + offset:
                elapsed_time = time.time() - vh_up[obj_id]
                if obj_id not in counter1:
                    counter1.append(obj_id)
                    speed_ms = DISTANCE_BETWEEN_LINES / elapsed_time
                    speed_kmh = speed_ms * 3.6
                    timestamp = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

                    if speed_kmh > SPEED_LIMIT:
                        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 3)
                        cv2.putText(frame, "OVERSPEED!", (x3, y3 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        if y3 < y4 and x3 < x4:
                            violation_img = frame[max(0, y3):y4, max(0, x3):x4]
                            if violation_img.size > 0:
                                snap_path = f"violations/car_{obj_id}_{timestamp}.jpg"
                                cv2.imwrite(snap_path, violation_img)
                                print(f"[INFO] Overspeed snapshot saved: {snap_path}")

                    writer.writerow([obj_id, "Up", int(speed_kmh), timestamp])
                    file.flush()

                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID:{obj_id} {int(speed_kmh)}Km/h",
                                (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw lines
        cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 1)
        cv2.putText(frame, 'L1', (277, cy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)
        cv2.putText(frame, 'L2', (182, cy2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display counts
        cv2.putText(frame, f"Down: {len(counter)}", (60, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Up: {len(counter1)}", (60, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Write & show
        output.write(frame)
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Cleanup
cap.release()
output.release()
cv2.destroyAllWindows()

# =================== STATS & PLOTS ===================
df = pd.read_csv("car_details.csv")

# 1. Vehicle count by direction
plt.figure(figsize=(6,4))
df["Direction"].value_counts().plot(kind="bar", color=["orange","blue"])
plt.title("Vehicle Count by Direction")
plt.xlabel("Direction")
plt.ylabel("Count")
plt.savefig("visualizations/vehicle_count.png")
plt.show()

# 2. Speed distribution
plt.figure(figsize=(6,4))
df["Speed (Km/h)"].plot(kind="hist", bins=10, color="green", edgecolor="black")
plt.title("Speed Distribution")
plt.xlabel("Speed (Km/h)")
plt.ylabel("Frequency")
plt.savefig("visualizations/speed_distribution.png")
plt.show()

# 3. Average speed per direction
plt.figure(figsize=(6,4))
df.groupby("Direction")["Speed (Km/h)"].mean().plot(kind="bar", color=["cyan","magenta"])
plt.title("Average Speed by Direction")
plt.ylabel("Average Speed (Km/h)")
plt.savefig("visualizations/avg_speed.png")
plt.show()

# 4. Top 5 fastest vehicles
top5 = df.sort_values(by="Speed (Km/h)", ascending=False).head(5)
plt.figure(figsize=(6,4))
plt.barh(top5["Car ID"].astype(str), top5["Speed (Km/h)"], color="red")
plt.title("Top 5 Fastest Vehicles")
plt.xlabel("Speed (Km/h)")
plt.ylabel("Car ID")
plt.savefig("visualizations/top5_fastest.png")
plt.show()

# 5. Speed trend over time
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df_sorted = df.sort_values("Timestamp")
plt.figure(figsize=(8,4))
plt.plot(df_sorted["Timestamp"], df_sorted["Speed (Km/h)"], marker="o", color="blue")
plt.title("Speed Trend Over Time")
plt.xlabel("Time")
plt.ylabel("Speed (Km/h)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/speed_trend.png")
plt.show()
