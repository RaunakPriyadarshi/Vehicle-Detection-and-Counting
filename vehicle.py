import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import defaultdict
from datetime import datetime
import pandas as pd
import time

# Initialize video capture
cap = cv2.VideoCapture('video.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Define morphological operations kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Minimum contour area to be considered a vehicle
min_contour_area = 1500

# Line position for counting vehicles
line_y_position = 450

# Vehicle counters
toward_vehicle_count = 0
away_vehicle_count = 0

# Debounce count to avoid multiple counting
debounce_count = defaultdict(int)

# Timer setup
start_time = time.time()
save_interval = 20  # Save every 20 seconds
records = []

# Define the centroid tracking algorithm
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared
        self.counted = {}

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.counted[self.nextObjectID] = False
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.counted[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int((x + x + w) / 2.0)
            cY = int((y + y + h) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects

ct = CentroidTracker()

# Store previous centroids to check crossing
previous_centroids = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = bg_subtractor.apply(frame)

    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 0.2 < aspect_ratio < 4.0:  # Filter out unlikely objects based on aspect ratio
                rects.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # Check if the vehicle crossed the line
        if objectID in previous_centroids:
            prev_y = previous_centroids[objectID][1]
            if prev_y < line_y_position and centroid[1] >= line_y_position:
                if not ct.counted[objectID] and debounce_count[objectID] == 0:
                    toward_vehicle_count += 1
                    ct.counted[objectID] = True
                    debounce_count[objectID] = 5  # Set a debounce period
            elif prev_y > line_y_position and centroid[1] <= line_y_position:
                if not ct.counted[objectID] and debounce_count[objectID] == 0:
                    away_vehicle_count += 1
                    ct.counted[objectID] = True
                    debounce_count[objectID] = 5  # Set a debounce period

        if debounce_count[objectID] > 0:
            debounce_count[objectID] -= 1

        previous_centroids[objectID] = centroid

    # Draw the counting line
    cv2.line(frame, (0, line_y_position), (frame.shape[1], line_y_position), (255, 0, 0), 2)

    # Display vehicle counts
    cv2.putText(frame, f'Toward Vehicle Count: {toward_vehicle_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Away Vehicle Count: {away_vehicle_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Get current time and display it on the frame
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_time, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the video frames
    cv2.imshow('Vehicle Detection', frame)
    cv2.imshow('Foreground Mask', fg_mask)

    # Check if it's time to save the records
    if time.time() - start_time >= save_interval:
        records.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), toward_vehicle_count, away_vehicle_count])
        df = pd.DataFrame(records, columns=['Timestamp', 'Toward Vehicle Count', 'Away Vehicle Count'])
        df.to_excel('vehicle_counts.xlsx', index=False)

        # Reset counts and timer
        toward_vehicle_count = 0
        away_vehicle_count = 0
        start_time = time.time()

    if cv2.waitKey(30) & 0xFF == 27:
        break

# Save final records
records.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), toward_vehicle_count, away_vehicle_count])
df = pd.DataFrame(records, columns=['Timestamp', 'Toward Vehicle Count', 'Away Vehicle Count'])
df.to_excel('vehicle_counts.xlsx', index=False)

cap.release()
cv2.destroyAllWindows()
