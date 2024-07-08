import tensorflow as tf
import numpy as np
import cv2

# Loading the saved model
model = tf.keras.models.load_model('handwritten_digit_recognition_model.h5')

# Getting video or camera capture
cap = cv2.VideoCapture("digits.mp4")

if not cap.isOpened():
    print("Video is unavailable")
else:
    print("Video capturing started.")

    # Simple function that will close the video capturing tab when user pressed the corresponding or 'x' button
    def can_exit(capt_key, capt_prop):
        if capt_key == ord('q') or capt_prop < 1:
            return True

        return False

    # Simple function that will convert the number to string
    def get_number_as_text(curr_number):
        if curr_number == 0:
            return "(zero)"
        elif curr_number == 1:
            return "(one)"
        elif curr_number == 2:
            return "(two)"
        elif curr_number == 3:
            return "(three)"
        elif curr_number == 4:
            return "(four)"
        elif curr_number == 5:
            return "(five)"
        elif curr_number == 6:
            return "(six)"
        elif curr_number == 7:
            return "(seven)"
        elif curr_number == 8:
            return "(eight)"
        elif curr_number == 9:
            return "(nine)"

    # Simple function that will create the texts which are going to be displayed on the capture screen
    def get_text_to_display(curr_number, curr_accuracy, curr_threshold):
        number_detected_text = "Current number: "
        if curr_accuracy > curr_threshold:
            number_detected_text += str(curr_number) + " " + get_number_as_text(curr_number) + "."
        else:
            number_detected_text += "No number detected."

        accuracy_detected_text = "Accuracy: " + "%" + str(round((curr_accuracy*100), 2))
        threshold_detected_text = "Threshold set to: %" + str(round((curr_threshold*100), 2))
        return number_detected_text, accuracy_detected_text, threshold_detected_text


    threshold = 0.75

    print("Starting to detecting numbers.\n")

    counter = 0
    while True:
        # Capture started
        success, frame = cap.read()
        counter = counter + 1
        if counter % 2 == 0:
            if frame is None:
                print("Video ended.")
                break

            # Captured image is preprocessed as well as the dataset which is used for training
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

            new_img = tf.keras.utils.normalize(resized, axis=1)
            IMG_SIZE = 28
            new_img = np.array(new_img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

            # Detecting initiated and both number and accuracy are saved
            predictions = model.predict(new_img)
            number = np.argmax(predictions)
            accuracy = predictions[0][number]

            bigger_frame = cv2.resize(frame, (0, 0), fx=4, fy=4)

            # Texts which are going to be printed on the display are saved and displayed
            number_text, accuracy_text, threshold_text = get_text_to_display(number, accuracy, threshold)
            cv2.putText(bigger_frame, number_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv2.putText(bigger_frame, accuracy_text, (5, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv2.putText(bigger_frame, threshold_text, (5, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

            smaller_frame = cv2.resize(bigger_frame, (0, 0), fx=0.5, fy=0.5)

            # The prepared display screen is shown
            cv2.imshow("Video", smaller_frame)

            captured_key = cv2.waitKey(1)
            captured_property = cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE)

            # Previous defined function to detect whether the user wants to close the program or not, if yes closing.
            if can_exit(captured_key, captured_property):
                print("Closed by user.")
                break

# After the video ends or capture is aborted,
# Current captured source is released to use for another programs and all windows are deleted on the screen
cap.release()
cv2.destroyAllWindows()
