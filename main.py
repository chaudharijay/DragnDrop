import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import cvzone

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Initialize hand detector
detector = HandDetector(detectionCon=1, maxHands=1)
colorR = (255, 0, 255)  # Rectangle color

# Define rectangle properties
class DragRect:
    def __init__(self, posCenter, size=(200, 200)):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # Check if the index finger tip is in the rectangle area
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor  # Update rectangle position

# Create a list of draggable rectangles
rectList = [DragRect([x * 250 + 150, 150]) for x in range(5)]

while True:
    success, img = cap.read()
    if not success:
        break  # Exit if frame not read correctly

    img = cv2.flip(img, 1)  # Flip the image horizontally
    img = detector.findHands(img)  # Detect hands
    lmList, _ = detector.findPosition(img)  # Get landmark positions

    if lmList:
        # Calculate distance between landmarks 8 and 12 (index and middle fingers)
        l, _, _ = detector.findDistance(8, 12, img, draw=False)
        if l < 60:  # If fingers are close enough
            cursor = lmList[8]  # Get index finger tip landmark
            # Update rectangle positions
            for rect in rectList:
                rect.update(cursor)

    # Draw transparent rectangles
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    # Blend the original image with the new rectangles
    out = img.copy()
    alpha = 0.2
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    # Display the output image
    cv2.imshow("Image", out)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
        break

# Release resources
cap.release()
cv2.destroyAllWindows()