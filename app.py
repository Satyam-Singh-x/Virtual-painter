import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import HandTrackingModule as htm

# Set page config
st.set_page_config(
    page_title="🖌️ Virtual Painter",
    page_icon="🎨",
    layout="centered"
)

# Stylish custom CSS
st.markdown("""
    <style>
    body {
        background-color: #fafafa;
    }
    .stButton>button {
        background-color: #6c5ce7;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 12px;
        border: none;
    }
    .stImage img {
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🖌️ Virtual Painter App")
st.write("Use your hand to draw, erase, and select colors in real time using your webcam.")

# Brush and eraser settings
brushThickness = 15
eraserThickness = 50

folderPath = 'header-images'
mylist = os.listdir(folderPath)
overlay = [cv2.cvtColor(cv2.imread(os.path.join(folderPath, imgPath)), cv2.COLOR_BGR2RGB) for imgPath in mylist]

if 'xp' not in st.session_state:
    st.session_state.xp = 0
    st.session_state.yp = 0
    st.session_state.drawColor = (200, 162, 200)  # Light purple by default
    st.session_state.header = overlay[2]
    st.session_state.streaming = False

detector = htm.HandDetector(detectionCon=0.85)
canvas_width, canvas_height = 960, 540
imgCanvas = np.zeros((canvas_height, canvas_width, 3), np.uint8)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button('🟢 Start Painting', key='start'):
        st.session_state.streaming = True
with col2:
    if st.button('🔴 Stop Painting', key='stop'):
        st.session_state.streaming = False

frame_window = st.image([], width=960)

cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 720)

while st.session_state.streaming:
    success, img = cap.read()
    if not success:
        st.warning("⚠️ Cannot access webcam.")
        break

    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        fingers = detector.fingersUp()

        # Selection mode
        if fingers[1] and fingers[2]:
            cv2.rectangle(img, (x1, y1 - 18), (x2, y2 + 18), st.session_state.drawColor, cv2.FILLED)
            st.session_state.xp, st.session_state.yp = 0, 0

            if y1 < st.session_state.header.shape[0] + 50:
                if 100 < x1 < 260:
                    st.session_state.header = overlay[2]
                    st.session_state.drawColor = (200, 162, 200)  # Light Purple
                elif 320 < x1 < 470:
                    st.session_state.header = overlay[0]
                    st.session_state.drawColor = (0, 165, 255)  # Orange
                elif 520 < x1 < 700:
                    st.session_state.header = overlay[1]
                    st.session_state.drawColor = (0, 255, 0)  # Green
                elif 720 < x1 < 900:
                    st.session_state.header = overlay[3]
                    st.session_state.drawColor = (0, 0, 0)  # Eraser (Black)

        # Drawing mode
        elif fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, st.session_state.drawColor, cv2.FILLED)
            if st.session_state.xp == 0 and st.session_state.yp == 0:
                st.session_state.xp, st.session_state.yp = x1, y1

            thickness = eraserThickness if st.session_state.drawColor == (0, 0, 0) else brushThickness
            cv2.line(imgCanvas, (st.session_state.xp, st.session_state.yp), (x1, y1), st.session_state.drawColor, thickness)
            st.session_state.xp, st.session_state.yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    h, w, _ = st.session_state.header.shape
    img[0:h, 0:w] = st.session_state.header

    frame_window.image(img, channels="BGR")

cap.release()
st.success("✅ Streaming stopped. Thank you for using Virtual Painter!")
