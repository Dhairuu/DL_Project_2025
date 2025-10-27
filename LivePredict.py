from Utils import *
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import os


class DrawingApp:
    def __init__(self, master):
        self.master = master
        master.title("MNIST Live Predictor")
        self.canvas_size = 1000
        self.pen_width = 80

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas = tk.Canvas(master, bg="black", width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        tk.Button(master, text="Predict", command=self.predict_digit).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(master, text="Clear", command=self.clear_canvas).pack(side=tk.RIGHT, padx=10, pady=10)
        self.result_label = tk.Label(master, text="Draw a digit (0-9)", font=('Arial', 14))
        self.result_label.pack(side=tk.TOP, pady=5)

        self.l1 = layers(784)
        self.l2 = layers(128) 
        self.l3 = layers(64)
        self.l4 = layers(10) 

        self.l2.weights(self.l1.layerSize)
        self.l3.weights(self.l2.layerSize)
        self.l4.weights(self.l3.layerSize)
        
        try:
            with open('parameters.json','r') as file:
                Js_Object = js.load(file)

            self.l2.W = np.array(Js_Object['W2'])
            self.l2.b = np.array(Js_Object['B2'])
            self.l3.W = np.array(Js_Object['W3'])
            self.l3.b = np.array(Js_Object['B3'])
            self.l4.W = np.array(Js_Object['W4'])
            self.l4.b = np.array(Js_Object['B4'])
            self.result_label.config(text="Model Loaded. Draw a digit.")
        except FileNotFoundError:
            self.result_label.config(text="ERROR: parameters.json not found. Run Training.py first.")
            
    def paint(self, event):
        x, y = event.x, event.y
        r = self.pen_width // 2
        
        x1, y1 = x - r, y - r
        x2, y2 = x + r, y + r

        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white", width=0)
        
        self.draw.ellipse([x1, y1, x2, y2], fill=255, outline=255) 

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit (0-9)")


    def predict_digit(self):
        bbox = self.image.getbbox()
        if bbox is None:
            self.result_label.config(text="Please draw a digit first.")
            return

        final_img = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        img_array = np.array(final_img).astype(np.float64)
        
        input_vector = (img_array / 255.0).reshape(784, 1) # Reshape to (784, 1) column vector
        
        self.l1.A = input_vector 
        m = 1

        forwardPropogation(self.l1, self.l2, self.l3, self.l4, m)
        
        prediction = get_predictions(self.l4.A)
        
        predicted_digit = prediction[0]
        self.result_label.config(text=f"Prediction: {predicted_digit}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
