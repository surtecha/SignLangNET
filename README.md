<p align="center">
  <img src="https://github.com/surtecha/SignLangNET/assets/91011302/ff899ebf-09c1-487f-8242-47cfe6cf062e" alt="SignLangNET logo" width="250"/>
</p>
<h1 align="center">SignLangNET</h1>

SignLangNET is a project aimed at interpreting sign language gestures using deep learning techniques. It utilizes the MediaPipe library for hand and body pose estimation and employs Long Short-Term Memory (LSTM) networks for sequence modeling.

## Table of Contents

- [Overview](#overview)
- [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Collection](#data-collection)
  - [Model Training](#model-training)
  - [Real-time Gesture Interpretation](#real-time-gesture-interpretation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project consists of three main components:

1. **Data Collection**: Utilizes webcam input to collect sequences of hand and body pose keypoints corresponding to various sign language gestures. The `utils.py` file must be run first to capture the data. A total of 40 videos will be captured, each 30 frames long. The captured data will be stored in the `DATA_PATH` folder, with each frame of the action stored in .npy format.

2. **Model Training**: Trains an LSTM neural network to classify sign language gestures based on the collected keypoints. Run the `model.py` file to start training the model after collecting the data.

3. **Real-time Gesture Interpretation**: Interprets sign language gestures in real-time using the trained model. Run the `detect.py` file after training the model.

## Long Short-Term Memory (LSTM)

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture designed to overcome the vanishing gradient problem and efficiently capture long-range dependencies in sequential data. LSTMs are well-suited for sequence modeling tasks like time series forecasting, natural language processing, and gesture recognition.

In an LSTM network, each LSTM unit maintains a cell state that can store information over long sequences, which allows it to remember important patterns and relationships in the data. Additionally, LSTMs have gating mechanisms (input gate, forget gate, output gate) that control the flow of information and gradients, enabling them to learn and retain information over many time steps.

For more information on LSTMs, refer to the [LSTM Wikipedia page](https://en.wikipedia.org/wiki/Long_short-term_memory).

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

**Note for Windows Users:** If you are using a Windows machine, please modify the `requirements.txt` file by replacing the TensorFlow packages with Windows-specific versions.

## Usage

### Data Collection

1. Run `utils.py` to collect training data. Follow on-screen instructions to perform gestures in front of the webcam. The program will exit automatically after recording all the gestures.

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
