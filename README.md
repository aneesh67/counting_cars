🚗 Vehicle Detection, Counting & Speed Estimation

This project is an AI-powered traffic monitoring system that detects, counts, and tracks vehicles in real-time using YOLOv8 and a custom-built tracking algorithm. It not only counts vehicles moving in both directions but also estimates their speed, identifies overspeed violations, and generates visual analytics for traffic insights.

🔹 Features

Real-time vehicle detection using YOLOv8

Custom tracker to assign unique IDs and avoid double-counting

Speed estimation based on vehicle crossing times between two reference lines

Overspeed detection & snapshots of violating vehicles (saved automatically)

CSV logging of vehicle details (ID, direction, speed, timestamp)

Visualization outputs:

Vehicle count by direction

Speed distribution histogram

Average speed by direction

Top 5 fastest vehicles

Speed trends over time

🔹 Tech Stack

Python, OpenCV, NumPy, Pandas, Matplotlib

YOLOv8 (Ultralytics) for object detection

Custom Tracker (tracker.py) for vehicle ID management

CSV + Plots for structured analytics

🔹 How It Works

YOLOv8 detects vehicles in each video frame.

Tracker.py assigns consistent IDs to detected vehicles.

Speed is calculated using time taken to cross two reference lines and a predefined real-world distance.

Overspeeding vehicles are highlighted in red, logged in CSV, and a snapshot is saved.

After processing, detailed visual reports are generated inside the visualizations/ folder.

🔹 Project Structure
📂 counting_cars
 ┣ 📜 speed.py          # Main script (detection, tracking, speed estimation, visualization)
 ┣ 📜 tracker.py        # Custom tracking algorithm
 ┣ 📜 coco.txt          # Class names for YOLO
 ┣ 📂 violations/       # Saved snapshots of overspeeding vehicles
 ┣ 📂 visualizations/   # Plots and analytics
 ┣ 📜 car_details.csv   # Logged vehicle data
 ┗ 📜 output_tracking.mp4 # Processed video with annotations

🔹 Future Enhancements

Deploy via FastAPI for real-time web service

Extend detection to classify vehicle types (car, truck, bike)

Use GPS-calibrated distances for higher accuracy in speed estimation

Integration with a dashboard for live monitoring
