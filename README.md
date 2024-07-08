# Handwritten Digit Recognition

This project uses a neural network model to recognize handwritten digits. The project consists of three main parts: a GUI application, a training script, and a testing script. 


## Features

1. **GUI Application **:
    - Users can draw digits on a canvas.
    - The drawn digit is recognized and the result is displayed on the screen.
    - The canvas can be cleared.
    - The result is saved as `Result.png`.

2. **Training Code **:
    - Trains the model on a dataset of handwritten digits.
    - Saves the trained model as `handwritten_digit_recognition_model.h5`.

3. **Testing Code **:
    - Recognizes digits from a video or camera stream.
    - Displays the recognized digits and their confidence levels on the video stream in real-time.

## Requirements

- Python 3.7
- OpenCV
- TensorFlow
- Numpy
- Pillow

## Setup

1. Install the required Python packages:

    ```sh
    pip install opencv-python-headless tensorflow numpy pillow
    ```

2. Download the `handwritten_digit_recognition_model.h5` model file and place it in the project directory.

## Usage

### GUI Application

1. Run the `handwr_dig_reg_GUI_App.py` file:

    ```sh
    python handwr_dig_reg_GUI_App.py
    ```

2. Draw a digit on the canvas that appears, and click the "Recognize Digit" button to start the recognition process. The recognized digit will be saved as `Result_Prediction.jpg`.

### Training Script

1. Run the `handwr_dig_reg_train.py` file to train the model:

    ```sh
    python handwr_dig_reg_train.py
    ```

2. The trained model will be saved as `handwritten_digit_recognition_model.h5`.

### Testing Script

1. Run the `handwr_dig_reg_test.py` file to start recognizing digits from a video stream:

    ```sh
    python handwr_dig_reg_test.py
    ```

2. Watch the video stream and see the recognized digits displayed in real-time on the screen.

## Project Files

- **handwr_dig_reg_GUI_App.py**: GUI application code.
- **handwr_dig_reg_train.py**: Training script code.
- **handwr_dig_reg_test.py**: Testing script code.

## Author

- **Hilal Işık**

This project provides a simple GUI and video processing application using TensorFlow and OpenCV to recognize handwritten digits. The GUI application allows users to draw digits and recognize them, while the video processing application recognizes digits in real-time from a video stream.
