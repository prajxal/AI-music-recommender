import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 10):
        print("WARNING: This application was designed for Python 3.10+")
        print(f"Your current Python version is {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All packages installed successfully!")
    except subprocess.CalledProcessError:
        print("ERROR: Failed to install required packages.")
        print("Please try manually installing them using: pip install -r requirements.txt")
        sys.exit(1)

def check_model_files():
    """Check if model files exist"""
    model_path = os.path.join(r"C:\Users\srini\Downloads", "model.h5")
    labels_path = os.path.join(r"C:\Users\srini\Downloads", "labels.npy")
    
    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found at {model_path}")
        print("The application will run in demo mode with simulated emotions.")
    
    if not os.path.exists(labels_path):
        print(f"WARNING: Labels file not found at {labels_path}")
        print("The application will run in demo mode with simulated emotions.")
    
    if os.path.exists(model_path) and os.path.exists(labels_path):
        print("Model files found. Application will use your trained emotion model.")

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("WARNING: Could not access webcam. Please check your camera permissions.")
        else:
            print("Camera access successful.")
        cap.release()
    except ImportError:
        print("WARNING: OpenCV not installed properly. Camera functionality may not work.")

def main():
    """Main setup function"""
    print("=" * 50)
    print("Emotion Music Recommender Setup")
    print("=" * 50)
    
    check_python_version()
    install_requirements()
    check_model_files()
    check_camera()
    
    print("\nSetup complete! You can now run the application with:")
    print("streamlit run emotion_music_app.py")
    print("=" * 50)

if __name__ == "__main__":
    main()
