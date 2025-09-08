import cv2
import pandas as pd
import time
import csv
from ultralytics import YOLO
from tracker import Tracker


# Load the YOLO model
model = YOLO('yolov8s.pt')

# Read class names from the coco.txt file
with open("coco.txt", "r") as file:
    class_list = file.read().split("\n")

# Video capture setup
cap = cv2.VideoCapture('veh2.mp4')
# cap = cv2.VideoCapture("rtsp://username:password@ip:port/stream")

# Get the width, height, and FPS of the input video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter to save the output video
output = cv2.VideoWriter('output_tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (1020, 500))

# Object tracker initialization
tracker = Tracker()

# Define the lines (L1 and L2) for tracking vehicles
cy1 = 322
cy2 = 368
offset = 6

# Dictionaries and lists to track vehicle movement and counts
vh_down = {}
counter = []
vh_up = {}
counter1 = []

count = 0

# Open a CSV file to store car details
with open('car_details.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the headers of the CSV file
    writer.writerow(["Car ID", "Direction", "Speed (Km/h)", "Timestamp", "Plate Number"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for speed optimization
        count += 1
        if count % 3 != 0:
            continue

        # Resize the frame for easier handling
        frame = cv2.resize(frame, (1020, 500))

        # Run YOLO model prediction on the frame
        results = model.predict(frame)
        boxes = results[0].boxes.data  # Extract the detection boxes
        px = pd.DataFrame(boxes).astype("float")

        # List to store the detected objects
        detections = []

        # Iterate over each detection
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            class_id = int(row[5])
            class_name = class_list[class_id]

            # Filter to only detect cars
            if 'car' in class_name:
                detections.append([x1, y1, x2, y2])

        # Update tracker with current detections
        bbox_id = tracker.update(detections)

        # Process each tracked object
        for bbox in bbox_id:
            x3, y3, x4, y4, obj_id = bbox
            cx = int((x3 + x4) // 2)
            cy = int((y3 + y4) // 2)

            # Draw bounding box
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

            # Check if vehicle crosses L1 going down
            if cy1 - offset < cy < cy1 + offset:
                vh_down[obj_id] = time.time()

            if obj_id in vh_down:
                if cy2 - offset < cy < cy2 + offset:
                    elapsed_time = time.time() - vh_down[obj_id]
                    if obj_id not in counter:
                        counter.append(obj_id)
                        distance = 10  # meters (adjust based on setup)
                        speed_ms = distance / elapsed_time
                        speed_kmh = speed_ms * 3.6
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                        # Save details to CSV
                        print(f"Saving to CSV: ID={obj_id}, Direction=Down, Speed={speed_kmh:.2f}, Time={timestamp}")
                        writer.writerow([obj_id, "Down", int(speed_kmh), timestamp])
                        file.flush()  # Ensure data is written immediately

                        # Annotate frame
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(frame, f"ID: {obj_id}", (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(frame, f"{int(speed_kmh)} Km/h", (x4, y4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Check if vehicle crosses L2 going up
            if cy2 - offset < cy < cy2 + offset:
                vh_up[obj_id] = time.time()

            if obj_id in vh_up:
                if cy1 - offset < cy < cy1 + offset:
                    elapsed1_time = time.time() - vh_up[obj_id]
                    if obj_id not in counter1:
                        counter1.append(obj_id)
                        distance1 = 10  # meters
                        speed_ms1 = distance1 / elapsed1_time
                        speed_kmh1 = speed_ms1 * 3.6
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                        # Save details to CSV
                        SPEED_LIMIT = 80
                        if speed_kmh > SPEED_LIMIT:
                            # Draw red bounding box
                            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 3)
                            cv2.putText(frame, "OVERSPEED!", (x3, y3 - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            # Save snapshot
                            violation_img = frame[y3:y4, x3:x4]
                            cv2.imwrite(f"violations/car_{obj_id}_{timestamp}.jpg", violation_img)
                        


                        # Save details to CSV (same as before)
                        writer.writerow([obj_id, "Down", int(speed_kmh), timestamp])
                        file.flush()

                        # Annotate frame
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(frame, f"ID: {obj_id}", (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(frame, f"{int(speed_kmh1)} Km/h", (x4, y4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw tracking lines and annotate
        cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 1)
        cv2.putText(frame, 'L1', (277, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)
        cv2.putText(frame, 'L2', (182, 367), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Write frame to output
        output.write(frame)

        # Display frame
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
            break

# Release resources
cap.release()
output.release()
cv2.destroyAllWindows()
