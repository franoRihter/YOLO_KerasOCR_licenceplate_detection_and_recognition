from ultralytics import YOLO
import cv2
import os

# Load trained model
model = YOLO("runs_yolo/detect/train2/weights/best.pt")

# Predict on an image
results = model("/home/frano/Documents/diplomski_skupa/copilot/yolo/yolo11/5bac39f3-ford-focus-1.4-16v-plin-slika-175741817.jpg")  # returns a list of results
os.makedirs("crops", exist_ok=True)

for result in results:  # Loop through each result (usually one per image)
    img = result.orig_img  # Get original image (NumPy array)

    # Loop through each detected box
    for i, box in enumerate(result.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box.tolist())  # Convert tensor to list, then ints
        crop = img[y1:y2, x1:x2]

        # Save and show crop
        filename = f"crops/crop_{i}.jpg"
        cv2.imwrite(filename, crop)
        cv2.imshow(f"Crop {i}", crop)

cv2.waitKey(0)
cv2.destroyAllWindows()
