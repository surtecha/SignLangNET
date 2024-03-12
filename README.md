<p align="center">
  <img src="https://github.com/surtecha/SignLangNET/assets/91011302/ff899ebf-09c1-487f-8242-47cfe6cf062e" alt="SignLangNET logo" width="250"/>
</p>
<h1 align="center">SignLangNET</h1>


SignLangNET is a project aimed at interpreting sign language gestures using deep learning techniques. It utilizes the MediaPipe library for hand and body pose estimation and employs Long Short-Term Memory (LSTM) networks for sequence modeling.

## Overview

This project consists of three main components:

1. **Data Collection**: Utilizes webcam input to collect sequences of hand and body pose keypoints corresponding to various sign language gestures.

2. **Model Training**: Trains an LSTM neural network to classify sign language gestures based on the collected keypoints.

3. **Real-time Gesture Interpretation**: Interprets sign language gestures in real-time using the trained model.

## Dependencies

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- MediaPipe
- TensorFlow
- Scikit-learn

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/surtecha/SignLangNET.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Collection

1. Run `utils.py` to collect training data. This script records sequences of hand and body pose keypoints for different sign language gestures.

2. Follow on-screen instructions to perform gestures in front of the webcam. Press 'q' to stop recording and move to the next gesture.

### Model Training

1. Run `model.py` to train the LSTM model using the collected data.

2. The trained model will be saved as `saved_model.h5`.

### Real-time Gesture Interpretation

1. Run `detect.py` to interpret sign language gestures in real-time using the trained model.

2. Perform gestures in front of the webcam, and the predicted gestures will be displayed on the screen.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- This project utilizes the [MediaPipe](https://github.com/google/mediapipe) library developed by Google.
- Inspiration for this project comes from efforts to make technology more accessible for individuals with disabilities.
