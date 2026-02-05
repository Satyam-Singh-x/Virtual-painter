🎨 Virtual Painter App

An elegant, real-time virtual painting web app powered by Streamlit, OpenCV, and Hand Tracking.

Draw, erase, and change colors using hand gestures without any external hardware—just your webcam.

Live Video Demo: https://www.youtube.com/watch?v=iE6sB0k3Wu4

🚀 Features

✋ Gesture-Controlled Drawing

Two fingers up → Select tools (colors or eraser)

Index finger up → Drawing mode

🎨 Multiple brush colors including Light Purple, Orange, Green, and Eraser

🖼️ Dynamic, modern UI with stylish design

💻 Real-time interaction via webcam

✅ Clean session management (start & stop streaming easily)

🎯 Technologies Used

Python 3.x

Streamlit

OpenCV

NumPy


Custom Hand Tracking Module (based on Mediapipe)

⚡ Live Demo

You can run the app locally using:

streamlit run app.py

📋 Setup Instructions

Clone the repository:

git clone https://github.com/Satyam-Singh-x/virtual-painter.git
cd virtual-painter


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py

🎨 How to Use

Click Start Painting to activate the webcam.

Use gestures:

✌️ Two fingers → Select tool (color or eraser).

☝️ One finger → Draw on the canvas.

Click Stop Painting when finished.

🧱 Project Structure
virtual-painter/
│
├──virtual-painter.py       #basic code for the app
├── app.py                  # Main Streamlit app
├── HandTrackingModule.py   # Custom hand detection module
├── header-images/          # Header images for UI
├── requirements.txt        # Python dependencies
└── README.md               # Project description

📜 License

MIT License – Feel free to use, modify, and distribute.

⭐ Star this project if you like it!

Made with ❤️ by Satyam

Btech Jadavpur University

Contact me:

Email: singhsatyam.0912@gmail.com

