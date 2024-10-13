import jetson.utils
import jetson.inference
import cv2
import numpy as np
import pygame
import time

# Initialize sound library (pygame)
pygame.mixer.init()
pygame.mixer.music.load("alert.wav")

# Set up camera and display window using Jetson utils
camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")  # Use your video device
display = jetson.utils.glDisplay()

# Loop for processing camera input
while display.IsOpen():
    # Capture two frames for motion detection
    img1, width, height = camera.CaptureRGBA()
    time.sleep(0.1)  # Add small delay to simulate frame difference
    img2, width, height = camera.CaptureRGBA()

    # Convert Jetson image format (RGBA) to OpenCV format (BGR)
    frame1 = jetson.utils.cudaToNumpy(img1, width, height, 4)
    frame2 = jetson.utils.cudaToNumpy(img2, width, height, 4)

    frame1 = cv2.cvtColor(frame1.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    frame2 = cv2.cvtColor(frame2.astype(np.uint8), cv2.COLOR_RGBA2BGR)

    # Motion detection logic
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected motion and play sound
    for c in contours:
        if cv2.contourArea(c) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        pygame.mixer.music.play()  # Play sound when motion is detected

    # Convert the OpenCV frame back to RGBA for displaying with Jetson utils
    display_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGBA)
    jetson_frame = jetson.utils.cudaFromNumpy(display_frame)

    # Render the image to the display window
    display.Render(jetson_frame)

    # Check if 'q' is pressed to quit
    if cv2.waitKey(10) == ord('q'):
        break

# Clean up resources
camera.Close()
display.Close()
pygame.mixer.quit()
