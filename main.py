import cvzone
import cv2  
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
 
st.set_page_config(layout="wide")
st.title("AI Gesture Calculator")
 
col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Activate Project', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

genai.configure(api_key="AIzaSyAu7w2tMO4kIAiB-RDMh8vywmF8OqBjpQk")
model = genai.GenerativeModel('gemini-1.5-flash')
 
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    st.error("Error: Unable to access the camera.")

cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None
 
def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(current_pos), tuple(prev_pos), (0, 255, 0), 10)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(canvas)
        prev_pos = None
    return current_pos, canvas
 
def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        if np.count_nonzero(canvas) == 0:
            return "Canvas is empty. Draw something first!"
        
        pil_image = Image.fromarray(canvas)
        if pil_image.size == (0, 0):
            return "Canvas is empty. Draw something first!"
        
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return ""

prev_pos = None
canvas = None

while True:
    success, img = cap.read()
    if not success or img is None or img.size == 0:
        st.error("Error: Unable to capture video from the camera.")
        continue
    
    img = cv2.flip(img, 1)
    
    if canvas is None:
        canvas = np.zeros_like(img)
 
    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)
    else:
        output_text = ""

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")
    
    if output_text:
        output_text_area.text(output_text)

    cv2.waitKey(1)


# import cvzone
# import cv2  
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import google.generativeai as genai
# from PIL import Image
# import streamlit as st
 
 
# st.set_page_config(layout="wide")
# st.title("AI Gesture Calculator")
 
# col1, col2 = st.columns([3,2])
# with col1:
#     run = st.checkbox('Activate Project', value=True)
#     FRAME_WINDOW = st.image([])
 
# with col2:
#     st.title("Answer")
#     output_text_area = st.subheader("")
 
 
# genai.configure(api_key="AIzaSyAu7w2tMO4kIAiB-RDMh8vywmF8OqBjpQk")
# model = genai.GenerativeModel('gemini-1.5-flash')
 
# # Initialize the webcam to capture video
# # The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)
 
# # Initialize the HandDetector class with the given parameters
# detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)
 
 
# def getHandInfo(img):
#     # Find hands in the current frame
#     # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
#     # The 'flipType' parameter flips the image, making it easier for some detections
#     hands, img = detector.findHands(img, draw=False, flipType=True)
 
#     # Check if any hands are detected
#     if hands:
#         # Information for the first hand detected
#         hand = hands[0]  # Get the first hand detected
#         lmList = hand["lmList"]  # List of 21 landmarks for the first hand
#         # Count the number of fingers up for the first hand
#         fingers = detector.fingersUp(hand)
#         print(fingers)
#         return fingers, lmList
#     else:
#         return None
 
# def draw(info,prev_pos,canvas):
#     fingers, lmList = info
#     current_pos= None
#     if fingers == [0, 1, 0, 0, 0]:
#         current_pos = lmList[8][0:2]
#         if prev_pos is None: prev_pos = current_pos
#         cv2.line(canvas,current_pos,prev_pos,(0, 255, 0),10)
#     elif fingers == [1, 0, 0, 0, 0]:
#         canvas = np.zeros_like(img)
 
#     return current_pos, canvas
 
# def sendToAI(model,canvas,fingers):
#     if fingers == [1,1,1,1,0]:
#         pil_image = Image.fromarray(canvas)
#         response = model.generate_content(["Solve this math problem", pil_image])
#         return response.text
 
 
# prev_pos= None
# canvas=None
# image_combined = None
# output_text= ""
# # Continuously get frames from the webcam
# while True:
#     # Capture each frame from the webcam
#     # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
#     success, img = cap.read()
#     img = cv2.flip(img, 1)
 
#     if canvas is None:
#         canvas = np.zeros_like(img)
 
 
#     info = getHandInfo(img)
#     if info:
#         fingers, lmList = info
#         prev_pos,canvas = draw(info, prev_pos,canvas)
#         output_text = sendToAI(model,canvas,fingers)
 
#     image_combined= cv2.addWeighted(img,0.7,canvas,0.3,0)
#     FRAME_WINDOW.image(image_combined,channels="BGR")
 
#     if output_text:
#         output_text_area.text(output_text)

#     # Keep the window open and update it for each frame; wait for 1 millisecond between frames
#     cv2.waitKey(1) # wait key imtiyaz




# import cvzone
# import cv2
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import google.generativeai as genai
# from PIL import Image
# import streamlit as st
# import sqlite3
# from datetime import datetime

# # Initialize Streamlit page
# st.set_page_config(layout="wide")
# st.text("AI Gesture Calculator")

# # Initialize the database
# def init_db():
#     conn = sqlite3.connect('gesture_calculator.db')
#     cursor = conn.cursor()
#     # Create a table to store calculations
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS calculations (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
#             expression TEXT,
#             result TEXT
#         )
#     ''')
#     conn.commit()
#     return conn

# conn = init_db()
# cursor = conn.cursor()

# # Streamlit layout
# col1, col2 = st.columns([3,2])
# with col1:
#     run = st.checkbox('Activate Project', value=True)
#     FRAME_WINDOW = st.image([])
 
# with col2:
#     st.title("Answer")
#     output_text_area = st.subheader("")
    
#     st.title("AI Calculator Gesture")
#     history = cursor.execute('SELECT timestamp, expression, result FROM calculations ORDER BY id DESC LIMIT 10').fetchall()
    
#     if history:
#         for record in history:
#             timestamp, expression, result = record
#             st.write(f"**{timestamp}**: {expression} = {result}")
#     else:
#         st.write("No calculations yet.")

# # Configure Generative AI
# genai.configure(api_key="AIzaSyAu7w2tMO4kIAiB-RDMh8vywmF8OqBjpQk")  # Replace with your actual API key
# model = genai.GenerativeModel('gemini-1.5-flash')

# # Initialize the webcam
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

# # Initialize HandDetector
# detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# def getHandInfo(img):
#     hands, img = detector.findHands(img, draw=False, flipType=True)
#     if hands:
#         hand = hands[0]
#         lmList = hand["lmList"]
#         fingers = detector.fingersUp(hand)
#         print(fingers)
#         return fingers, lmList
#     else:
#         return None

# def draw(info, prev_pos, canvas, img):
#     fingers, lmList = info
#     current_pos = None
#     if fingers == [0, 1, 0, 0, 0]:
#         current_pos = lmList[8][0:2]
#         if prev_pos is None:
#             prev_pos = current_pos
#         cv2.line(canvas, tuple(current_pos), tuple(prev_pos), (0, 0, 255), 10)
#         prev_pos = current_pos
#     elif fingers == [1, 0, 0, 0, 0]:
#         canvas = np.zeros_like(img)
#         prev_pos = None
#     return prev_pos, canvas

# def sendToAI(model, canvas, fingers):
#     if fingers == [1,1,1,1,0]:
#         pil_image = Image.fromarray(canvas)
#         # Depending on how you process the canvas to get the expression
#         # Here, we're sending the image to the AI model
#         response = model.generate_content(["Solve this math problem", pil_image])
#         return response.text
#     return None

# prev_pos = None
# canvas = None
# output_text = ""

# if run:
#     while True:
#         success, img = cap.read()
#         img = cv2.flip(img, 1)
 
#         if canvas is None:
#             canvas = np.zeros_like(img)
 
#         info = getHandInfo(img)
#         if info:
#             fingers, lmList = info
#             prev_pos, canvas = draw(info, prev_pos, canvas, img)
#             result = sendToAI(model, canvas, fingers)
#             if result:
#                 output_text = result
#                 output_text_area.text(output_text)
                
#                 # Insert into database
#                 expression = "Parsed Expression"  # Implement your parsing logic here
#                 result_text = output_text
#                 cursor.execute('''
#                     INSERT INTO calculations (expression, result)
#                     VALUES (?, ?)
#                 ''', (expression, result_text))
#                 conn.commit()
                
#                 # Refresh history
#                 st.experimental_rerun()
 
#         image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
#         FRAME_WINDOW.image(image_combined, channels="BGR")
 
#         cv2.waitKey(1)
