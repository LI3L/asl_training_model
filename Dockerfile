# Use the official TensorFlow image as the base
FROM tensorflow/tensorflow:2.15.0

# Install C++ hex dump utility and heavy GUI dependencies for OpenCV on Ubuntu
RUN apt-get update && apt-get install -y \
    xxd \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install standard opencv-python and pin numpy to prevent TF 2.15 crashes.
# pyserial is for run_model.py --car / serial_bridge.py talking to the Mega.
RUN pip install pandas matplotlib scipy "numpy<2" opencv-python pyserial

WORKDIR /app

# Default command (can be overridden)
CMD ["python", "src/test_model.py", "webcam"]