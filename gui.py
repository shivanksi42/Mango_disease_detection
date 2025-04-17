import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tkinter import messagebox
import numpy as np
from tensorflow.keras.models import load_model, Model
import tensorflow as tf


class CNNEnsemble(tf.keras.Model):
    def __init__(self, model_paths=None, models=None):
        super(CNNEnsemble, self).__init__()
        if models is not None:
            self.models = models
        elif model_paths is not None:
            self.models = [load_model(path) for path in model_paths]
        else:
            raise ValueError("Either model_paths or models must be provided")
        
    def call(self, inputs):
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(inputs, training=False)
            predictions.append(pred)
        
        # Average the predictions
        ensemble_pred = tf.reduce_mean(predictions, axis=0)
        return ensemble_pred
    
    def save(self, filepath):
        # Save all individual models
        for i, model in enumerate(self.models):
            model.save(f"{filepath}_model_{i}")
            
    @classmethod
    def load(cls, filepath, num_models):
        # Load all individual models
        models = []
        for i in range(num_models):
            model = load_model(f"{filepath}_model_{i}")
            models.append(model)
        return cls(models=models)
    

# fold_model_paths = [f'model_fold_{i}.h5' for i in range(1,6)]
# ensemble_model = CNNEnsemble(model_paths=fold_model_paths)


# Load all four models
models = {
    
    "DenseNet121": tf.keras.models.load_model("DenseNet121_model.h5"),
    
}


selected_model_name = "DenseNet121"
model = models[selected_model_name]


class_labels = {
    0: "Anthracnose",
    1: "Bacterial Canker",
    2: "Cutting Weevil",
    3: "Die Back",
    4: "Gall Midge",
    5: "Healthy",
    6: "Powdery Mildew",
    7: "Sooty Mould"
}

root = tk.Tk()
root.title("Mango Leaf Disease Detection System")
root.minsize(800, 600)
root.configure(bg="#e0f7fa")


img_path = ""
img_display = None


def upload_image():
    global img_path, img_display
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if img_path:
        try:
            img = Image.open(img_path)
            img = img.resize((300, 300))
            img_display = ImageTk.PhotoImage(img)
            image_label.config(image=img_display)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")


def perform_detection():
    global model
    if not img_path:
        messagebox.showerror("Error", "Please upload an image first!")
        return
    
    # Load and preprocess the image
    img = Image.open(img_path)
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make predictions
    predictions = model.predict(img) if selected_model_name == "DenseNet121" else model(img)
    predicted_class_idx = tf.argmax(predictions[0]).numpy()
    predicted_class = class_labels[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx])
    
    # Display results
    result_label.config(text=f"Detected Disease: {predicted_class}")
    confidence_label.config(text=f"Confidence: {confidence:.2f}")
    selected_model_label.config(text=f"Using Model: {selected_model_name}")


def reset_gui():
    global img_display, img_path
    image_label.config(image="")
    result_label.config(text="")
    confidence_label.config(text="")
    selected_model_label.config(text="Using Model: None")
    img_display = None
    img_path = ""


def on_model_selection(event):
    global model, selected_model_name
    selected_model_name = model_selection_var.get()
    model = models[selected_model_name]
    selected_model_label.config(text=f"Using Model: {selected_model_name}")


# GUI Layout
title_label = tk.Label(root, text="Mango Leaf Disease Detection System", font=("Arial", 18, "bold"), bg="#e0f7fa")
title_label.pack(pady=20)

# Dropdown for model selection
model_selection_var = tk.StringVar(value=selected_model_name)
model_dropdown = tk.OptionMenu(root, model_selection_var, *models.keys(), command=on_model_selection)
model_dropdown.config(font=("Arial", 12), bg="#4caf50", fg="white")
model_dropdown.pack(pady=10)

# Label to show the selected model
selected_model_label = tk.Label(root, text=f"Using Model: {selected_model_name}", font=("Arial", 14), bg="#e0f7fa")
selected_model_label.pack(pady=10)

image_label = tk.Label(root, bg="#b2ebf2")
image_label.pack(pady=10)

upload_button = tk.Button(root, text="Upload Image", font=("Arial", 12), command=upload_image, bg="#00897b", fg="white")
upload_button.pack(pady=10)

detect_button = tk.Button(root, text="Detect Disease", font=("Arial", 12), command=perform_detection, bg="#00796b", fg="white")
detect_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#e0f7fa")
result_label.pack(pady=10)

confidence_label = tk.Label(root, text="", font=("Arial", 12), bg="#e0f7fa")
confidence_label.pack(pady=10)

# Reset and Exit buttons
button_frame = tk.Frame(root, bg="#e0f7fa")
button_frame.pack(pady=20)
reset_button = tk.Button(button_frame, text="Reset", font=("Arial", 12), command=reset_gui, bg="#ff8a65", fg="white")
reset_button.pack(side="left", padx=10)
exit_button = tk.Button(button_frame, text="Exit", font=("Arial", 12), command=root.quit, bg="#d32f2f", fg="white")
exit_button.pack(side="right", padx=10)

root.mainloop()





















#starter code for this file 
# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
# import numpy as np
# import tensorflow as tf
# from tkinter import messagebox

# model = tf.keras.models.load_model("DenseNet121_model.h5")



# root = tk.Tk()
# root.title("Mango Leaf Disease Detection System")
# root.minsize(800, 600)
# root.configure(bg="#e0f7fa")



# img_path = ""
# img_display = None
# class_labels = {
#     0: "Anthracnose",
#     1: "Bacterial Canker",
#     2: "Cutting Weevil",
#     3: "Die Back",
#     4: "Gall Midge",
#     5: "Healthy",
#     6: "Powdery Mildew",
#     7: "Sooty Mould"
# }


# def upload_image():
#     global img_path, img_display
#     img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
#     if img_path:
#         try:
#             img = Image.open(img_path)
#             img = img.resize((300, 300))
#             img_display = ImageTk.PhotoImage(img)
#             image_label.config(image=img_display)
#         except Exception as e:
#             messagebox.showerror("Error", f"Failed to load image: {str(e)}")


# def perform_detection():
#     if not img_path:
#         messagebox.showerror("Error", "Please upload an image first!")
#         return
#     img = Image.open(img_path)
#     img = img.resize((256, 256))
#     img = np.array(img) / 255.0
#     img = np.expand_dims(img, axis=0)
    
#     predictions = model.predict(img)
#     predicted_class_idx = np.argmax(predictions)
#     predicted_class = class_labels[predicted_class_idx]
#     confidence = predictions[0][predicted_class_idx]
#     result_label.config(text=f"Detected Disease: {predicted_class}")
#     confidence_label.config(text=f"Confidence: {confidence:.2f}")

# def reset_gui():    
#     global img_display, img_path
#     image_label.config(image="")
#     result_label.config(text="")
#     confidence_label.config(text="")
#     img_display = None
#     img_path = ""


# title_label = tk.Label(root, text="Mango Leaf Disease Detection System", font=("Arial", 18, "bold"), bg="#e0f7fa")
# title_label.pack(pady=20)

# image_label = tk.Label(root, bg="#b2ebf2")
# image_label.pack(pady=10)

# upload_button = tk.Button(root, text="Upload Image", font=("Arial", 12), command=upload_image, bg="#00897b", fg="white")
# upload_button.pack(pady=10)

# detect_button = tk.Button(root, text="Detect Disease", font=("Arial", 12), command=perform_detection, bg="#00796b", fg="white")
# detect_button.pack(pady=10)

# result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#e0f7fa")
# result_label.pack(pady=10)

# confidence_label = tk.Label(root, text="", font=("Arial", 12), bg="#e0f7fa")
# confidence_label.pack(pady=10)

# button_frame = tk.Frame(root, bg="#e0f7fa")
# button_frame.pack(pady=20)
# reset_button = tk.Button(button_frame, text="Reset", font=("Arial", 12), command=reset_gui, bg="#ff8a65", fg="white")
# reset_button.pack(side="left", padx=10)
# exit_button = tk.Button(button_frame, text="Exit", font=("Arial", 12), command=root.quit, bg="#d32f2f", fg="white")
# exit_button.pack(side="right", padx=10)

# root.mainloop()







# version 2
# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
# import numpy as np
# import tensorflow as tf
# from tkinter import messagebox




# # Load all four models
# models = {
#     "VGG16": tf.keras.models.load_model("VGG16_model.h5"),
#     "VGG19": tf.keras.models.load_model("VGG19_model.h5"),
#     "DenseNet121": tf.keras.models.load_model("DenseNet121_model.h5"),
#     "ResNet50": tf.keras.models.load_model("ResNet50_model.h5"),
# }


# selected_model_name = "DenseNet121"
# model = models[selected_model_name]


# class_labels = {
#     0: "Anthracnose",
#     1: "Bacterial Canker",
#     2: "Cutting Weevil",
#     3: "Die Back",
#     4: "Gall Midge",
#     5: "Healthy",
#     6: "Powdery Mildew",
#     7: "Sooty Mould"
# }

# root = tk.Tk()
# root.title("Mango Leaf Disease Detection System")
# root.minsize(800, 600)
# root.configure(bg="#e0f7fa")


# img_path = ""
# img_display = None


# def upload_image():
#     global img_path, img_display
#     img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
#     if img_path:
#         try:
#             img = Image.open(img_path)
#             img = img.resize((300, 300))
#             img_display = ImageTk.PhotoImage(img)
#             image_label.config(image=img_display)
#         except Exception as e:
#             messagebox.showerror("Error", f"Failed to load image: {str(e)}")


# def perform_detection():
#     global model
#     if not img_path:
#         messagebox.showerror("Error", "Please upload an image first!")
#         return
    
#     # Load and preprocess the image
#     img = Image.open(img_path)
#     img = img.resize((256, 256))
#     img = np.array(img) / 255.0
#     img = np.expand_dims(img, axis=0)
    
#     # Make predictions
#     predictions = model.predict(img)
#     predicted_class_idx = np.argmax(predictions)
#     predicted_class = class_labels[predicted_class_idx]
#     confidence = predictions[0][predicted_class_idx]
    
#     # Display results
#     result_label.config(text=f"Detected Disease: {predicted_class}")
#     confidence_label.config(text=f"Confidence: {confidence:.2f}")
#     selected_model_label.config(text=f"Using Model: {selected_model_name}")


# def reset_gui():
#     global img_display, img_path
#     image_label.config(image="")
#     result_label.config(text="")
#     confidence_label.config(text="")
#     selected_model_label.config(text="Using Model: None")
#     img_display = None
#     img_path = ""


# def on_model_selection(event):
#     global model, selected_model_name
#     # Update the selected model based on the dropdown selection
#     selected_model_name = model_selection_var.get()
#     model = models[selected_model_name]
#     selected_model_label.config(text=f"Using Model: {selected_model_name}")


# # GUI Layout
# title_label = tk.Label(root, text="Mango Leaf Disease Detection System", font=("Arial", 18, "bold"), bg="#e0f7fa")
# title_label.pack(pady=20)

# # Dropdown for model selection
# model_selection_var = tk.StringVar(value=selected_model_name)
# model_dropdown = tk.OptionMenu(root, model_selection_var, *models.keys(), command=on_model_selection)
# model_dropdown.config(font=("Arial", 12), bg="#4caf50", fg="white")
# model_dropdown.pack(pady=10)

# # Label to show the selected model
# selected_model_label = tk.Label(root, text=f"Using Model: {selected_model_name}", font=("Arial", 14), bg="#e0f7fa")
# selected_model_label.pack(pady=10)

# image_label = tk.Label(root, bg="#b2ebf2")
# image_label.pack(pady=10)

# upload_button = tk.Button(root, text="Upload Image", font=("Arial", 12), command=upload_image, bg="#00897b", fg="white")
# upload_button.pack(pady=10)

# detect_button = tk.Button(root, text="Detect Disease", font=("Arial", 12), command=perform_detection, bg="#00796b", fg="white")
# detect_button.pack(pady=10)

# result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#e0f7fa")
# result_label.pack(pady=10)

# confidence_label = tk.Label(root, text="", font=("Arial", 12), bg="#e0f7fa")
# confidence_label.pack(pady=10)

# # Reset and Exit buttons
# button_frame = tk.Frame(root, bg="#e0f7fa")
# button_frame.pack(pady=20)
# reset_button = tk.Button(button_frame, text="Reset", font=("Arial", 12), command=reset_gui, bg="#ff8a65", fg="white")
# reset_button.pack(side="left", padx=10)
# exit_button = tk.Button(button_frame, text="Exit", font=("Arial", 12), command=root.quit, bg="#d32f2f", fg="white")
# exit_button.pack(side="right", padx=10)

# root.mainloop()