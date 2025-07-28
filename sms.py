from twilio.rest import Client
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Read credentials and numbers from environment variables
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_number = os.getenv('TWILIO_PHONE_NUMBER')
recipient_number = os.getenv('RECIPIENT_PHONE_NUMBER')

# Create Twilio client
client = Client(account_sid, auth_token)

# Send SMS
message = client.messages.create(
    body="Wear your helmet! This is your final warning!",
    from_=twilio_number,
    to=recipient_number
)

# Print message SID
print("Message sent successfully! SID:", message.sid)
