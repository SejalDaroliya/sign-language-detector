import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('asl_recognition_model.h5')

# Define the labels
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
          'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
          'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
          'u', 'v', 'w', 'x', 'y', 'z']  # list of labels in the same order as the folders

# Create the GUI window
window = tk.Tk()
window.title("ASL Recognition")

# Create a label to display the input image
input_label = tk.Label(window, text="Input Image")
input_label.pack()

# Create a label to display the output prediction
output_label = tk.Label(window, text="Prediction")
output_label.pack()

# Create a label to display the confidence score
confidence_label = tk.Label(window, text="Confidence Score")
confidence_label.pack()

# Create a button to select the input image
def select_image():
    filepath = filedialog.askopenfilename()
    input_image = Image.open(filepath)
    input_image = input_image.resize((224, 224))
    input_image = np.array(input_image) / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    prediction = model.predict(input_image)
    predicted_label = np.argmax(prediction[0])
    output_label.config(text=f"Prediction: {labels[predicted_label]}")
    confidence_label.config(text=f"Confidence Score: {prediction[0][predicted_label]:.2f}")

select_button = tk.Button(window, text="Select Image", command=select_image)
select_button.pack()

# Start the GUI event loop
window.mainloop()