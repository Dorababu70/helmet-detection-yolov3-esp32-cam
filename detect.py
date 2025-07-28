import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.models import load_model
import pytesseract
from twilio.rest import Client

# Allow TensorFlow GPU growth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Twilio setup (credentials removed for security)
account_sid = 'YOUR_TWILIO_ACCOUNT_SID'
auth_token = 'YOUR_TWILIO_AUTH_TOKEN'
client = Client(account_sid, auth_token)

def send_sms(to_number, message_body):
    message = client.messages.create(
        body=message_body,
        from_='YOUR_TWILIO_PHONE_NUMBER',
        to=to_number
    )
    print(f"Message sent to {to_number}: {message.sid}")

# Predefined license plates and associated phone numbers
predefined_plates = {
    "AP01CD1234": "PHONE_NUMBER_1",
    # Add more entries if needed
}

# Load YOLO model
net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load CNN model
model = load_model('helmet-nonhelmet_cnn.h5')
print('Helmet detection model loaded!')

# Load a test image
image_path = '1.jpg'  # Replace with your test image path
img = cv2.imread(image_path)

COLORS = [(0, 255, 0), (0, 0, 255)]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define video writer (optional)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter('output.avi', fourcc, 5, (888, 500))

# Helmet detection function
def helmet_or_nohelmet(helmet_roi):
    try:
        helmet_roi = cv2.resize(helmet_roi, (224, 224))
        helmet_roi = np.array(helmet_roi, dtype='float32')
        helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
        helmet_roi = helmet_roi / 255.0
        prediction = model.predict(helmet_roi)[0][0]
        return 1 if prediction > 0.5 else 0
    except Exception as e:
        print(f"Error processing helmet ROI: {e}")
        return None

# License plate detection
def detect_license_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return pytesseract.image_to_string(binary, config='--psm 8')

# Process the image
img = imutils.resize(img, height=500)
height, width = img.shape[:2]

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

confidences = []
boxes = []
classIds = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.3:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            classIds.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        color = [int(c) for c in COLORS[classIds[i]]]
        if classIds[i] == 0:  # Bike
            helmet_roi = img[max(0, y):y + h, max(0, x):x + w]
            helmet_status = helmet_or_nohelmet(helmet_roi)

            if helmet_status == 1:
                print("No helmet detected! Alert!")
                cv2.putText(img, "No Helmet", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                print("Helmet detected.")
                cv2.putText(img, "Helmet", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        elif classIds[i] == 1:  # Number plate
            plate_roi = img[y:y + h, x:x + w]
            detected_plate = detect_license_plate(plate_roi).strip()
            print(f"Detected License Plate: {detected_plate}")

            if detected_plate in predefined_plates:
                phone_number = predefined_plates[detected_plate]
                if helmet_status == 1:
                    send_sms(phone_number, "Wear your helmet! This is your final warning!!!")

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

# Save and display result
cv2.imwrite('output_image.jpg', img)
cv2.imshow("Helmet Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
