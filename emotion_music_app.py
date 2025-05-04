import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import time
import random
from PIL import Image
import os
import sys

# Add the directory containing your data collection and training scripts to path
sys.path.append(r"C:\Users\srini\Downloads")

# Set page configuration
st.set_page_config(
    page_title="Emotion Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .title-text {
        font-size: 50px !important;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle-text {
        font-size: 25px !important;
        color: #3B82F6;
        text-align: center;
        margin-bottom: 30px;
    }
    .emotion-label {
        font-size: 36px !important;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .happy {
        background-color: #FEF3C7;
        color: #D97706;
    }
    .sad {
        background-color: #DBEAFE;
        color: #2563EB;
    }
    .angry {
        background-color: #FEE2E2;
        color: #DC2626;
    }
    .neutral {
        background-color: #D1FAE5;
        color: #059669;
    }
    .surprised {
        background-color: #E0E7FF;
        color: #4F46E5;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 18px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Title and intro
st.markdown("<h1 class='title-text'>Emotion Music Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>We'll detect your emotions and suggest music to match your mood!</p>", unsafe_allow_html=True)

# Create emotion to music mapping
emotion_to_music = {
    "happy": [
        {"title": "Happy", "artist": "Pharrell Williams", "url": "https://open.spotify.com/track/60nZcImufyMA1MKQZ2FfW"},
        {"title": "Can't Stop the Feeling!", "artist": "Justin Timberlake", "url": "https://open.spotify.com/track/1WkMMavIMc4JZ8cfMmxHkI"},
        {"title": "Good as Hell", "artist": "Lizzo", "url": "https://open.spotify.com/track/3Yh9lZcWyKrK9GZ43JYHzE"},
        {"title": "Walking on Sunshine", "artist": "Katrina & The Waves", "url": "https://open.spotify.com/track/05wIrZSwuaVWhcv5FfqeH0"},
        {"title": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars", "url": "https://open.spotify.com/track/32OlwWuMpZ6b0aN2RZOeMS"}
    ],
    "sad": [
        {"title": "Someone Like You", "artist": "Adele", "url": "https://open.spotify.com/track/4qJSYQ4USQQnuLK9YMmJGr"},
        {"title": "Fix You", "artist": "Coldplay", "url": "https://open.spotify.com/track/7LVHVU3tWfcxj5aiPFEW4Q"},
        {"title": "Hurt", "artist": "Johnny Cash", "url": "https://open.spotify.com/track/28cnXtME493VX9NOw9cIUh"},
        {"title": "When the Party's Over", "artist": "Billie Eilish", "url": "https://open.spotify.com/track/43zdsphuZLzwA9k4DJhU0I"},
        {"title": "Skinny Love", "artist": "Bon Iver", "url": "https://open.spotify.com/track/2IMA6FGoDi45ebdjDY8UTs"}
    ],
    "angry": [
        {"title": "Break Stuff", "artist": "Limp Bizkit", "url": "https://open.spotify.com/track/5cZqsjdxiRKCIpou2SawGe"},
        {"title": "Killing In The Name", "artist": "Rage Against The Machine", "url": "https://open.spotify.com/track/59WN2psjkt1tyaxjspN8fp"},
        {"title": "Bodies", "artist": "Drowning Pool", "url": "https://open.spotify.com/track/2IEHhKGKoRMK9l6WZJUlTf"},
        {"title": "Before I Forget", "artist": "Slipknot", "url": "https://open.spotify.com/track/2eOuL8KesWzQgmBxkS9nQ9"},
        {"title": "You're Going Down", "artist": "Sick Puppies", "url": "https://open.spotify.com/track/38YlJ9lp9uwAOExXPRQKg2"}
    ],
    "neutral": [
        {"title": "Weightless", "artist": "Marconi Union", "url": "https://open.spotify.com/track/0t3ZvGKNFJAe4allGS0QFI"},
        {"title": "Intro", "artist": "The xx", "url": "https://open.spotify.com/track/2UvF1vv5I7AP67X03qqYOf"},
        {"title": "Daylight", "artist": "Matt and Kim", "url": "https://open.spotify.com/track/2nCIlS4H95X8b2QsjQyCxI"},
        {"title": "Porcelain", "artist": "Moby", "url": "https://open.spotify.com/track/1DxEbt6OSbrkRcgeu0Dsyu"},
        {"title": "Comptine d'un autre été", "artist": "Yann Tiersen", "url": "https://open.spotify.com/track/2O8avu9JbzIaNQ0Hfu8y0w"}
    ],
    "surprised": [
        {"title": "Wow.", "artist": "Post Malone", "url": "https://open.spotify.com/track/7xQAfvXzm3AkraOtGPWIZg"},
        {"title": "Thunderclouds", "artist": "LSD ft. Sia", "url": "https://open.spotify.com/track/6Q8cLXFzHeLQ5tPAXxBTJN"},
        {"title": "Surprise Yourself", "artist": "Jack Garratt", "url": "https://open.spotify.com/track/0ZrNa6CGgJGUvz4Ku7it0B"},
        {"title": "Supermassive Black Hole", "artist": "Muse", "url": "https://open.spotify.com/track/3lPr8ghNDBLc2uDsRQfGTL"},
        {"title": "Wow", "artist": "Beck", "url": "https://open.spotify.com/track/40GXjcTHBiVpGGQns2Mdyt"}
    ]
}

# Function to get emotion color class
def get_emotion_class(emotion):
    if emotion == "happy":
        return "happy"
    elif emotion == "sad":
        return "sad"
    elif emotion == "angry":
        return "angry"
    elif emotion == "neutral":
        return "neutral"
    elif emotion == "surprised":
        return "surprised"
    else:
        return "neutral"

# Initialize session state variables
if 'emotion' not in st.session_state:
    st.session_state.emotion = "neutral"
if 'start_webcam' not in st.session_state:
    st.session_state.start_webcam = False
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = time.time()
if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'labels' not in st.session_state:
    st.session_state.labels = None

# Load model only once when app starts
@st.cache_resource
def load_emotion_model():
    try:
        # Path to your model file - update this path as needed
        model_path = os.path.join(r"C:\Users\srini\Downloads", "model.h5")
        labels_path = os.path.join(r"C:\Users\srini\Downloads", "labels.npy")
        
        # Check if model files exist
        if os.path.exists(model_path) and os.path.exists(labels_path):
            model = load_model(model_path)
            labels = np.load(labels_path)
            return model, labels
        else:
            # For demo purposes, we'll simulate detection if model files don't exist
            st.warning("Model files not found. Running in demo mode with simulated emotions.")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load MediaPipe components
@st.cache_resource
def load_mediapipe():
    holistic = mp.solutions.holistic
    hands = mp.solutions.hands
    holis = holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    drawing = mp.solutions.drawing_utils
    return holistic, hands, holis, drawing

# Function to process landmarks for model input
def process_landmarks_for_prediction(res):
    """Process MediaPipe landmarks into the format expected by the model"""
    lst = []
    
    # Process face landmarks
    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        # Process left hand landmarks
        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        # Process right hand landmarks
        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)
                
        return np.array(lst).reshape(1, -1)
    
    return None

# Function to detect emotion from facial landmarks
def detect_emotion(res, model=None, labels=None):
    """Detect emotion from facial landmarks using model if available"""
    
    # Only update emotion every 3 seconds to avoid flickering
    current_time = time.time()
    if current_time - st.session_state.last_detection_time < 3:
        return st.session_state.emotion
    
    st.session_state.last_detection_time = current_time
    
    # If we have a trained model and labels, use them
    if model is not None and labels is not None and res.face_landmarks:
        try:
            # Process landmarks into format expected by the model
            features = process_landmarks_for_prediction(res)
            
            if features is not None:
                # Make prediction
                prediction = model.predict(features, verbose=0)  # Set verbose=0 to suppress output
                predicted_label = labels[np.argmax(prediction)]
                
                # Map your original labels to our emotion categories
                # Update this mapping based on your actual label names
                emotion_mapping = {
                    "happy": "happy",
                    "sad": "sad",
                    "angry": "angry",
                    "neutral": "neutral",
                    "surprised": "surprised",
                    # Add more mappings as needed for your model's classes
                }
                
                # Get the mapped emotion or use the original if no mapping exists
                st.session_state.emotion = emotion_mapping.get(predicted_label, predicted_label)
                return st.session_state.emotion
        except Exception as e:
            st.error(f"Error during emotion prediction: {e}")
            # Fall back to demo mode if prediction fails
    
    # Demo mode: randomly choose an emotion
    emotions = ["happy", "sad", "angry", "neutral", "surprised"]
    weights = [0.3, 0.2, 0.15, 0.25, 0.1]  # Make happy and neutral more common
    st.session_state.emotion = random.choices(emotions, weights=weights, k=1)[0]
    
    return st.session_state.emotion

# Get MediaPipe components
holistic, hands, holis, drawing = load_mediapipe()

# Page layout with columns for webcam and recommendations
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.button("Toggle Webcam" if st.session_state.start_webcam else "Start Webcam"):
        st.session_state.start_webcam = not st.session_state.start_webcam
        if st.session_state.start_webcam:
            st.session_state.model, st.session_state.labels = load_emotion_model()
    
    # Placeholder for webcam feed
    webcam_placeholder = st.empty()
    
    # Display emotion when detected
    if st.session_state.emotion:
        emotion_class = get_emotion_class(st.session_state.emotion)
        st.markdown(f"<div class='emotion-label {emotion_class}'>Current Mood: {st.session_state.emotion.upper()}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#1E3A8A;'>Music Recommendations</h3>", unsafe_allow_html=True)
    
    if st.button("Get Music Recommendations"):
        st.session_state.recommendations = emotion_to_music.get(st.session_state.emotion, [])
        st.session_state.show_recommendations = True
    
    if st.session_state.show_recommendations and st.session_state.recommendations:
        for idx, song in enumerate(st.session_state.recommendations):
            st.markdown(f"""
            <div style='padding:10px; margin-bottom:10px; background-color:#f8fafc; border-radius:5px;'>
                <p style='margin:0; font-weight:bold;'>{idx+1}. {song['title']}</p>
                <p style='margin:0; color:#6B7280;'>Artist: {song['artist']}</p>
                <p style='margin:0;'><a href='{song['url']}' target='_blank'>Listen on Spotify</a></p>
            </div>
            """, unsafe_allow_html=True)
    elif st.session_state.show_recommendations:
        st.info("No recommendations available for this emotion.")
    else:
        st.info("Click the button to get music recommendations based on your detected emotion.")
    st.markdown("</div>", unsafe_allow_html=True)

# If webcam is active
if st.session_state.start_webcam:
    try:
        # Use OpenCV to capture from device 0
        cap = cv2.VideoCapture(0)
        
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            st.error("Could not open webcam. Please check your camera permissions.")
        else:
            # Read a frame
            ret, frame = cap.read()
            
            if ret:
                # Process the frame
                frame = cv2.flip(frame, 1)  # Flip horizontally for mirror effect
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                res = holis.process(rgb_frame)
                
                # Draw landmarks
                if res.face_landmarks:
                    drawing.draw_landmarks(rgb_frame, res.face_landmarks, holistic.FACEMESH_CONTOURS)
                    
                    # Get emotion from landmarks using the model if available
                    emotion = detect_emotion(res, st.session_state.model, st.session_state.labels)
                    
                if res.left_hand_landmarks:
                    drawing.draw_landmarks(rgb_frame, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
                
                if res.right_hand_landmarks:
                    drawing.draw_landmarks(rgb_frame, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
                
                # Display the frame
                webcam_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
            
            # Release the webcam when done
            cap.release()
    
    except Exception as e:
        st.error(f"Error accessing webcam: {e}")
        st.info("If you're running this app locally, make sure to grant camera access.")
else:
    # Display a placeholder image when webcam is off
    placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
    placeholder_image[:] = (240, 240, 240)  # Light gray background
    
    # Add text to the placeholder
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Click 'Start Webcam' to begin"
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    
    # Get the text position to center it
    textX = (placeholder_image.shape[1] - textsize[0]) // 2
    textY = (placeholder_image.shape[0] + textsize[1]) // 2
    
    cv2.putText(placeholder_image, text, (textX, textY), font, 1, (0, 0, 0), 2)
    
    # Display the placeholder
    webcam_placeholder.image(placeholder_image, channels="RGB", use_column_width=True)

# Footer
st.markdown("""
<div style='text-align:center; margin-top:30px; padding:20px; color:#6B7280;'>
    <p>Created with ❤️ | Emotion Music Recommender</p>
    <p style='font-size:12px;'>Note: This is a demonstration app. For a production version, a properly trained emotion recognition model would be used.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with instructions
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>How It Works</h2>", unsafe_allow_html=True)
    st.markdown("""
    1. **Start the webcam** by clicking the button
    2. Position your face clearly in the frame
    3. The app will detect your emotions using facial recognition
    4. Click "Get Music Recommendations" to see songs that match your mood
    5. Click on any song link to listen on Spotify
    
    **Privacy Note:** Your video is processed locally and is not stored or sent to any server.
    """)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>About</h3>", unsafe_allow_html=True)
    st.markdown("""
    This app uses:
    - MediaPipe for facial landmark detection
    - Deep learning for emotion classification
    - Curated music recommendations based on emotional state
    
    **Detected Emotions:**
    - Happy 😊
    - Sad 😢
    - Angry 😠
    - Neutral 😐
    - Surprised 😲
    """)
