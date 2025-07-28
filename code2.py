import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.models import load_model
import pytesseract
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Twilio setup from .env file
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_number = os.getenv('TWILIO_PHONE_NUMBER')

client = Client(account_sid, auth_token)

def send_sms(to_number, message_body):
    message = client.messages.create(
        body=message_body,
        from_=twilio_number,
        to=to_number
    )
    print(f"Message sent to {to_number}: {message.sid}")

# Predefined license plates and associated phone numbers
predefined_plates = {
    "AP01CD1234": os.getenv("RECIPIENT_PHONE_NUMBER")
}

# Load YOLO model
net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load CNN model
model = load_model('helmet-nonhelmet_cnn.h5')
print('Helmet detection model loaded!')

COLORS = [(0, 255, 0), (0, 0, 255)]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Optional video writer
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter('output.avi', fourcc, 5, (888, 500))

def helmet_or_nohelmet(helmet_roi):
    try:
        helmet_roi = cv2.resize(helmet_roi, (224, 224))
        helmet_roi = np.array(helmet_roi, dtype='float32') / 255.0
        helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
        prediction = model.predict(helmet_roi)[0][0]
        return 1 if prediction > 0.5 else 0
    except Exception as e:
        print(f"Error processing helmet ROI: {e}")
        return None

def detect_license_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return pytesseract.image_to_string(binary, config='--psm 8')

# Replace with your actual ESP32-CAM stream URL
ip_camera_url = os.getenv("ESP32_CAM_URL")

cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("Error: Unable to open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from stream.")
        break

    frame = imutils.resize(frame, height=500)
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences, boxes, classIds = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x, center_y = int(detection[0]*width), int(detection[1]*height)
                w, h = int(detection[2]*width), int(detection[3]*height)
                x, y = int(center_x - w/2), int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = [int(c) for c in COLORS[classIds[i]]]

            if classIds[i] == 0:  # Bike
                helmet_roi = frame[max(0, y):y+h, max(0, x):x+w]
                helmet_status = helmet_or_nohelmet(helmet_roi)

                if helmet_status == 1:
                    print("No helmet detected! Alert!")
                    cv2.putText(frame, "No Helmet", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[1], 2)
                else:
                    print("Helmet detected.")
                    cv2.putText(frame, "Helmet", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)

            elif classIds[i] == 1:  # Number plate
                plate_roi = frame[y:y+h, x:x+w]
                detected_plate = detect_license_plate(plate_roi).strip()
                print(f"Detected License Plate: {detected_plate}")

                if detected_plate in predefined_plates and helmet_status == 1:
                    send_sms(predefined_plates[detected_plate], "Wear your helmet! This is your final warning!!!")

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
