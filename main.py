import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
from datetime import datetime
from djitellopy import Tello
import time

# ========== PATH SETUP ==========
BASE = r"C:\Users\andle\Desktop\FaceRecognitionRealTimeDatabase"
ENCODE_FILE = os.path.join(BASE, "EncodeFile.p")
BG = os.path.join(BASE, "Resources", "background.png")
MODES = os.path.join(BASE, "Resources", "Modes")
attendance_log = os.path.join(BASE, "attendance_log.txt")
os.makedirs(os.path.join(BASE, "SavedImages"), exist_ok=True)

# ========== RESOURCE LOADING ==========
imgBackground = cv2.imread(BG)
mode_imgs = [cv2.imread(os.path.join(MODES, f)) for f in sorted(os.listdir(MODES))]

# ========== TELLO DRONE SETUP ==========
try:
    tello = Tello()
    print("Attempting to connect to Tello...")
    tello.connect()
    print("Connection successful!")

    battery = tello.get_battery()
    print(f"Battery: {battery}%")

    if battery < 20:
        print("Warning: Low battery!")


    tello.streamon()
    time.sleep(2)  # Give time for stream to stabilize

except Exception as e:
    print(f"Failed to connect to Tello: {str(e)}")
    print("Please check:")
    print("1. You're connected to Tello's WiFi")
    print("2. The drone is powered on")
    print("3. Your firewall isn't blocking the connection")
    exit()

# ========== LOAD ENCODINGS ==========
print("Loading encodings...")
encodings, ids = pickle.load(open(ENCODE_FILE, 'rb'))
print("Encodings loaded successfully.")

# ========== STUDENT DATA ==========
student_data = {
    "963852": {"name": "Elon Musk", "major": "Physics", "starting_year": 2020, "total_attendance": 0,
               "standing": "Good", "year": 2},
    "123456": {"name": "Akira Hassan", "major": "Computer Science", "starting_year": "Autumn 2023",
               "total_attendance": 0, "standing": "Good", "year": 1},
    "191919": {"name": "Andleeb Muneer", "major": "Computer Science", "starting_year": "Autumn 2023",
               "total_attendance": 0, "standing": "Good", "year": 1},
    "852741": {"name": "Emily Blunt", "major": "Economics", "starting_year": "Autumn 2023",
               "total_attendance": 0, "standing": "Good", "year": 1}
}

# ========== PARAMETERS ==========
threshold = 0.42
modeType = 0
counter = 0
sid = None
crop = None

# ========== MAIN LOOP ==========
while True:
    # Get frame from Tello
    frame = tello.get_frame_read().frame
    if frame is None:
        continue

    # Tello provides frames in BGR format - we don't need to convert initially
    img = cv2.resize(frame, (640, 480))

    # Create small version for face recognition (convert to RGB for face_recognition)
    small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)  # Fixed typo: smmall → small

    # Face detection and recognition
    locs = face_recognition.face_locations(rgb_small)
    encs_cur = face_recognition.face_encodings(rgb_small, locs)

    # Background and mode UI setup - use original BGR image for display
    canvas = imgBackground.copy()
    canvas[162:162 + 480, 55:55 + 640] = img  # Original BGR image
    canvas[44:44 + 633, 808:808 + 414] = mode_imgs[modeType]

    # Rest of your existing face recognition and UI code...
    if encs_cur:
        dists = face_recognition.face_distance(encodings, encs_cur[0])
        idx = np.argmin(dists)

        if dists[idx] < threshold:
            sid = ids[idx]
            info = student_data.get(sid)

            if info:
                y1, x2, y2, x1 = [v * 4 for v in locs[0]]
                bbox = (55 + x1, 162 + y1, x2 - x1, y2 - y1)
                canvas = cvzone.cornerRect(canvas, bbox, rt=0)

                if counter == 0:
                    cvzone.putTextRect(canvas, "Loading", (275, 400))
                    cv2.imshow("Attendance", canvas)
                    cv2.waitKey(1)
                    modeType = 1
                    counter = 1

                if counter == 1:
                    crop = img[y1:y2, x1:x2]  # Use original BGR image
                    cv2.imwrite(os.path.join(BASE, "SavedImages", f"{sid}.jpg"), crop)
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(attendance_log, "a") as f:
                        f.write(f"{sid},{info['name']},{now}\n")
                    info["total_attendance"] += 1

                if 10 < counter < 20:
                    modeType = 2

                if counter <= 10:
                    cv2.putText(canvas, str(info["total_attendance"]), (861, 125), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (255, 255, 255), 1)
                    cv2.putText(canvas, info["major"], (1006, 550), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(canvas, sid, (1006, 493), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(canvas, info["standing"], (910, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(canvas, str(info["year"]), (1025, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100),
                                1)
                    cv2.putText(canvas, str(info["starting_year"]), (1125, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                (100, 100, 100), 1)
                    (w, _), _ = cv2.getTextSize(info["name"], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    if crop is not None:
                        canvas[175:175 + 216, 909:909 + 216] = cv2.resize(crop, (216, 216))
                    cv2.putText(canvas, info["name"], (808 + (414 - w) // 2, 445), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (50, 50, 50), 1)

                counter += 1
                if counter >= 20:
                    counter = 0
                    modeType = 0
            else:
                modeType = counter = 0
        else:
            modeType = counter = 0
    else:
        modeType = counter = 0

    # Display UI
    cv2.imshow("Attendance", canvas)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # This must be inside the while loop!
# Cleanup
try:
    tello.streamoff()
    tello.end()
except:
    pass
cv2.destroyAllWindows()
