# encode_generator.py

import os
import cv2
import face_recognition
import pickle

# Path to your images
folderPath = r"C:\Users\andle\Desktop\FaceRecognitionRealTimeDatabase\Images"

# List of student IDs (filenames without extension)
studentIds = ["963852", "123456", "191919", "852741"]

imgList = []
idsList = []

for studentId in studentIds:
    imgPathJpg = os.path.join(folderPath, f"{studentId}.jpg")
    imgPathPng = os.path.join(folderPath, f"{studentId}.png")

    if os.path.exists(imgPathJpg):
        imgPath = imgPathJpg
    elif os.path.exists(imgPathPng):
        imgPath = imgPathPng
    else:
        print(f"Image not found for ID: {studentId}")
        continue

    img = cv2.imread(imgPath)
    if img is None:
        print(f"Image cannot be read: {imgPath}")
        continue

    # Resize for consistency (optional but helps normalization)
    img = cv2.resize(img, (400, 400))

    # Show image for verification (optional)
    print(f"Encoding image for ID: {studentId}")
    cv2.imshow("Encoding", img)
    cv2.waitKey(500)  # Show for half a second

    imgList.append(img)
    idsList.append(studentId)

cv2.destroyAllWindows()

def findEncodings(imagesList, idsList):
    encodeList = []
    cleanIds = []

    for idx, img in enumerate(imagesList):
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgbImg)
        if len(encodings) == 0:
            print(f"No face found for ID: {idsList[idx]}")
            continue

        encodeList.append(encodings[0])
        cleanIds.append(idsList[idx])  # Only add ID if encoding was successful

    return encodeList, cleanIds

print("Encoding Started ...")
encodeListKnown, cleanedIdsList = findEncodings(imgList, idsList)
print("Encoding Complete")

encodeListKnownWithIds = [encodeListKnown, cleanedIdsList]

# Save encodings
with open("EncodeFile.p", 'wb') as file:
    pickle.dump(encodeListKnownWithIds, file)

print("✅ EncodeFile.p saved successfully")