import cv2

# Open face cascade classifier
faceCascade = cv2.CascadeClassifier("./data/classifiers/haarcascade_frontalface_default.xml")

# Open WebCam
cap = cv2.VideoCapture(0)

# Frame reduction factor
reduction_factor = 4

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Main loop
while True:
    ret, frame = cap.read()

    # Reduce size of frame to 1/reduction_factor for faster image processing.
    small_frame = cv2.resize(frame, None, fx=1/reduction_factor, fy=1/reduction_factor, interpolation=cv2.INTER_AREA)
    # Convert to grayscale for faster face detection
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1,1),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles on all idenfied faces 
    for (x, y, w, h) in faces:
        # Multiply by the reduction_factor the coordinates in order to fit the original image size 
        x *= reduction_factor
        y *= reduction_factor
        w *= reduction_factor
        h *= reduction_factor
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    text = "No faces found"
    text_color = (0, 0, 255, 255) 
    if len(faces) > 0:
        text = "Faces found: {}".format(len(faces))
        text_color = (0, 255, 0, 255)

    cv2.putText(frame, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)
    
    cv2.imshow('cam_test', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
