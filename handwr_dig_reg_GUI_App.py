import cv2
import numpy as np
from tkinter import *
from PIL import ImageGrab, Image

# Loading model.
from tensorflow.keras.models import load_model
model = load_model('handwritten_digit_recognition_model.h5')
print("Model successfully loaded. Executing GUI App.")

# Creating tkinter window.
root = Tk()
root.resizable(0, 0)
root.title("Handwritten Digit Recognition GUI App")

# Variables that will be used later.
lastx, lasty = None, None

# Creating a canvas for drawing
cv = Canvas(root, width=640, height=480, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)


# Canvas drawing method
    # This method will be called while drawing.
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    # Do the canvas drawings
    cv.create_line((lastx, lasty, x, y), width=8, fill='black',
                   capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


# Activate event method
    # This method will be called whenever something is done.
def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y


# Tkinter provides this event binding, whenever happens, execution start.
cv.bind('<Button-1>', activate_event)


# Clear canvas command
    # This method will be called whenever clear button is pressed.
def clear_widget():
    global cv
    cv.delete("all")


# Recognize Digit method
    # This method will be called whenever we want to predict number.
def Recognize_Digit():
    # Variables
    filename = 'Result.png'
    widget = cv

    # Getting widget coordinates
        # x, y -> Top left coordinates
        # x1, y1 -> Bottom right coordinates
    x = root.winfo_rootx() + widget.winfo_x()
    y = root.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    # Grab the image, crop it according the requirement and save in png format
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)


    # Read the image in color format
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    # Convert the image in grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Applying Otsu thresholding
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find Contours method will help to extract the contours from the image
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        # Getting bounding box of each number
        x, y, w, h = cv2.boundingRect(cnt)
        # Create rectangle around it
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left
        # Extract the image ROI
        roi = th[y - top:y + h + bottom, x - left:x + w + right]
        # Resize roi image to 28x28 pixels.
        img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        # Reshaping the image to support our model input.
        img = img.reshape(1, 28, 28, 1)
        # Normalizing the image to support our model input.
        img = img / 255.0
        # It is time to predict the result
        pred = model.predict([img])[0]
        # Get max value.
        final_pred = np.argmax(pred)
        data = f"The drawn number is: {final_pred}"
        # Write the prediction.
        Font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), Font, fontScale, color, thickness)

    convertedImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    savingImage = Image.fromarray(convertedImg)
    savingImage.save("Result_Prediction.jpg")
    cv2.imshow("Prediction", image)
    cv2.waitKey(0)


# Adding buttons, labels and set all commands.
btn_save = Button(text='Recognize Digit', command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text="Clear Widget", command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)

root.mainloop()
