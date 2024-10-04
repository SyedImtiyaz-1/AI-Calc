# import cvzone
# import cv2  
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import google.generativeai as genai
# from PIL import Image
# import streamlit as st
 
# st.set_page_config(layout="wide")
# st.title("AI Gesture Calculator")
 
# col1, col2 = st.columns([3, 2])
# with col1:
#     run = st.checkbox('Activate Project', value=True)
#     FRAME_WINDOW = st.image([])

# with col2:
#     st.title("Answer")
#     output_text_area = st.subheader("")

# genai.configure(api_key="AIzaSyAu7w2tMO4kIAiB-RDMh8vywmF8OqBjpQk")
# model = genai.GenerativeModel('gemini-1.5-flash')
 
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     st.error("Error: Unable to access the camera.")

# cap.set(3, 1280)
# cap.set(4, 720)
# detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# def getHandInfo(img):
#     hands, img = detector.findHands(img, draw=False, flipType=True)
#     if hands:
#         hand = hands[0]
#         lmList = hand["lmList"]
#         fingers = detector.fingersUp(hand)
#         return fingers, lmList
#     return None
 
# def draw(info, prev_pos, canvas):
#     fingers, lmList = info
#     current_pos = None
#     if fingers == [0, 1, 0, 0, 0]:
#         current_pos = lmList[8][0:2]
#         if prev_pos is None:
#             prev_pos = current_pos
#         cv2.line(canvas, tuple(current_pos), tuple(prev_pos), (0, 255, 0), 10)
#     elif fingers == [1, 0, 0, 0, 0]:
#         canvas = np.zeros_like(canvas)
#         prev_pos = None
#     return current_pos, canvas
 
# def sendToAI(model, canvas, fingers):
#     if fingers == [1, 1, 1, 1, 0]:
#         if np.count_nonzero(canvas) == 0:
#             return "Canvas is empty. Draw something first!"
        
#         pil_image = Image.fromarray(canvas)
#         if pil_image.size == (0, 0):
#             return "Canvas is empty. Draw something first!"
        
#         response = model.generate_content(["Solve this math problem", pil_image])
#         return response.text
#     return ""

# prev_pos = None
# canvas = None

# while True:
#     success, img = cap.read()
#     if not success or img is None or img.size == 0:
#         st.error("Error: Unable to capture video from the camera.")
#         continue
    
#     img = cv2.flip(img, 1)
    
#     if canvas is None:
#         canvas = np.zeros_like(img)
 
#     info = getHandInfo(img)
#     if info:
#         fingers, lmList = info
#         prev_pos, canvas = draw(info, prev_pos, canvas)
#         output_text = sendToAI(model, canvas, fingers)
#     else:
#         output_text = ""

#     image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
#     FRAME_WINDOW.image(image_combined, channels="BGR")
    
#     if output_text:
#         output_text_area.text(output_text)

#     cv2.waitKey(1)

# import cvzone
# import cv2
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import google.generativeai as genai
# from PIL import Image
# import streamlit as st

# # Configure Streamlit page
# st.set_page_config(layout="wide")
# st.title("AI Gesture Calculator")

# # Layout setup
# col1, col2 = st.columns([3, 2])
# with col1:
#     run = st.checkbox('Activate Project', value=True)
#     FRAME_WINDOW = st.image([])

# with col2:
#     st.title("Answer")
#     output_text_area = st.subheader("")

# # Configure Generative AI
# genai.configure(api_key="AIzaSyAu7w2tMO4kIAiB-RDMh8vywmF8OqBjpQk")  # Replace with your actual API key
# model = genai.GenerativeModel('gemini-1.5-flash')

# # Initialize Hand Detector
# detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# # Initialize session state for canvas and previous position
# if 'canvas' not in st.session_state:
#     st.session_state.canvas = None
# if 'prev_pos' not in st.session_state:
#     st.session_state.prev_pos = None

# def getHandInfo(img):
#     hands, img = detector.findHands(img, draw=False, flipType=True)
#     if hands:
#         hand = hands[0]
#         lmList = hand["lmList"]
#         fingers = detector.fingersUp(hand)
#         return fingers, lmList
#     return None

# def draw(info):
#     fingers, lmList = info
#     current_pos = None
#     if fingers == [0, 1, 0, 0, 0]:  # Index finger up
#         current_pos = lmList[8][0:2]  # Get fingertip position
#         if st.session_state.prev_pos is None:
#             st.session_state.prev_pos = current_pos
#         cv2.line(st.session_state.canvas, tuple(current_pos), tuple(st.session_state.prev_pos), (0, 255, 0), 10)
#         st.session_state.prev_pos = current_pos
#     elif fingers == [1, 0, 0, 0, 0]:  # Thumb up
#         st.session_state.canvas = np.zeros_like(st.session_state.canvas)
#         st.session_state.prev_pos = None

# def sendToAI(model, canvas, fingers):
#     if fingers == [1, 1, 1, 1, 0]:  # All fingers up
#         if np.count_nonzero(canvas) == 0:
#             return "Canvas is empty. Draw something first!"

#         pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
#         response = model.generate_content(["Solve this math problem", pil_image])
#         return response.text
#     return ""

# # Main logic
# if run:
#     # Start capturing video from the webcam
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         st.error("Error: Unable to access the camera.")
#     else:
#         if st.session_state.canvas is None:
#             ret, img = cap.read()
#             st.session_state.canvas = np.zeros_like(img)

#         while True:
#             success, img = cap.read()
#             if not success:
#                 st.error("Error: Unable to capture video from the camera.")
#                 break

#             img = cv2.flip(img, 1)  # Mirror the image

#             info = getHandInfo(img)
#             if info:
#                 fingers, lmList = info
#                 draw(info)
#                 output_text = sendToAI(model, st.session_state.canvas, fingers)
#             else:
#                 output_text = ""

#             image_combined = cv2.addWeighted(img, 0.7, st.session_state.canvas, 0.3, 0)
#             FRAME_WINDOW.image(cv2.cvtColor(image_combined, cv2.COLOR_BGR2RGB))

#             if output_text:
#                 output_text_area.text(output_text)

#             # Check for Streamlit's run state to break the loop
#             if not run:
#                 break

#         cap.release()
import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st

# Configure Streamlit page
st.set_page_config(layout="wide")
st.title("AI Gesture Calculator")

# Layout setup
col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Activate Project', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

# Configure Generative AI
genai.configure(api_key="AIzaSyAu7w2tMO4kIAiB-RDMh8vywmF8OqBjpQk")  # Replace with your actual API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize Hand Detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# Initialize session state for canvas and previous position
if 'canvas' not in st.session_state:
    st.session_state.canvas = None
if 'prev_pos' not in st.session_state:
    st.session_state.prev_pos = None

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up
        current_pos = lmList[8][0:2]  # Get fingertip position
        if st.session_state.prev_pos is None:  # Start new gesture
            st.session_state.prev_pos = current_pos
        cv2.line(st.session_state.canvas, tuple(current_pos), tuple(st.session_state.prev_pos), (0, 255, 0), 10)
        st.session_state.prev_pos = current_pos  # Update previous position
    else:
        # If not drawing, reset previous position
        st.session_state.prev_pos = None
        if fingers == [1, 0, 0, 0, 0]:  # Thumb up gesture (to clear canvas)
            st.session_state.canvas = np.zeros_like(st.session_state.canvas)

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:  # All fingers up
        if np.count_nonzero(canvas) == 0:
            return "Canvas is empty. Draw something first!"

        pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return ""

# Main logic
if run:
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Unable to access the camera.")
    else:
        if st.session_state.canvas is None:
            ret, img = cap.read()
            st.session_state.canvas = np.zeros_like(img)

        while True:
            success, img = cap.read()
            if not success:
                st.error("Error: Unable to capture video from the camera.")
                break

            img = cv2.flip(img, 1)  # Mirror the image

            info = getHandInfo(img)
            if info:
                fingers, lmList = info
                draw(info)
                output_text = sendToAI(model, st.session_state.canvas, fingers)
            else:
                output_text = ""

            image_combined = cv2.addWeighted(img, 0.7, st.session_state.canvas, 0.3, 0)
            FRAME_WINDOW.image(cv2.cvtColor(image_combined, cv2.COLOR_BGR2RGB))

            if output_text:
                output_text_area.text(output_text)

            # Check for Streamlit's run state to break the loop
            if not run:
                break

        cap.release()
