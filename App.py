from flask import Flask, render_template, request
import cv2
import numpy as np
from transformers import pipeline
import torch

# Initialize Flask app and sentiment analysis pipeline
app = Flask(__name__)
sentiment_analyzer = pipeline("sentiment-analysis")

# Load pre-trained YOLOv3 model (or you can use any other model)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getOutputsNames()]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Read the uploaded image
        file = request.files['image']
        img = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        # Perform object detection (using YOLO for example)
        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        # Process detected objects
        object_labels = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    object_labels.append(class_id)
        
        # Sentiment analysis of detected objects (assuming the object label is a text)
        sentiment_results = {}
        for label in object_labels:
            object_name = get_object_name(label)  # Assuming a function to map class_id to object name
            sentiment = sentiment_analyzer(object_name)
            sentiment_results[object_name] = sentiment[0]['label']
        
        return render_template('index.html', image=image, sentiment_results=sentiment_results)
    
    return render_template('index.html')

def get_object_name(class_id):
    # For simplicity, let's assume a mapping from class_id to object names for detection (you can replace it with actual labels from COCO or your dataset)
    labels = ["person", "car", "dog", "cat", "bottle", "chair", "tv", "table"]
    return labels[class_id % len(labels)]

if __name__ == '__main__':
    app.run(debug=True)
