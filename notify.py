import cv2
import pygame

# Initialize the sound system (pygame)
pygame.mixer.init()
pygame.mixer.music.load("alert.wav")

# Open the camera (change 1 to 0 if you're using the primary camera)
cam = cv2.VideoCapture(1)

while cam.isOpened():
    # Capture two consecutive frames
    ret, frame1 = cam.read()
    ret, frame2 = cam.read()

    # Compute the absolute difference between the two frames
    diff = cv2.absdiff(frame1, frame2)
    
    # Convert to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary threshold to get a binary image
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    
    # Dilate the threshold image to fill in small gaps
    dilated = cv2.dilate(thresh, None, iterations=3)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through contours and draw bounding boxes for large enough contours
    for c in contours:
        if cv2.contourArea(c) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Play sound alert asynchronously
        if not pygame.mixer.music.get_busy():  # Prevents replaying sound too quickly
            pygame.mixer.music.play()

    # Display the resulting frame with bounding boxes
    cv2.imshow('Granny Cam', frame1)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(10) == ord('q'):
        break

# Release the camera and close any open windows
cam.release()
cv2.destroyAllWindows()
pygame.mixer.quit()  # Properly shut down pygame mixer
