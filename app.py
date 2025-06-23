from flask import Flask, request, jsonify, render_template_string
from tensorflow import keras
from PIL import Image
import numpy as np
import io
import base64
import re
import os

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
try:
    model = keras.models.load_model('mnist_digit_recognizer.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# HTML template as a string (no external files needed)
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Digit Recognition</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #canvas { border: 1px solid black; background-color: black; }
        .buttons { margin: 10px 0; }
        button { padding: 8px 15px; margin-right: 10px; }
        #result { margin-top: 20px; font-size: 18px; }
    </style>
</head>
<body>
    <h1>Handwritten Digit Recognition</h1>
    <p>Draw a digit below and click Predict</p>
    <canvas id="canvas" width="280" height="280"></canvas>
    <div class="buttons">
        <button id="predict">Predict</button>
        <button id="clear">Clear</button>
    </div>
    <div id="result">Prediction: </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        
        // Setup canvas
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        
        // Drawing functions
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        function startDrawing(e) {
            isDrawing = true;
            draw(e);
        }
        
        function draw(e) {
            if (!isDrawing) return;
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(e.offsetX, e.offsetY);
        }
        
        function stopDrawing() {
            isDrawing = false;
            ctx.beginPath();
        }
        
        // Clear canvas
        document.getElementById('clear').addEventListener('click', function() {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('result').textContent = 'Prediction: ';
        });
        
        // Predict digit
        document.getElementById('predict').addEventListener('click', function() {
            const imageData = canvas.toDataURL('image/png');
            
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('result').textContent = 'Error: ' + data.error;
                } else {
                    document.getElementById('result').textContent = 
                        `Prediction: ${data.digit} (${data.confidence}% confidence)`;
                }
            })
            .catch(error => {
                document.getElementById('result').textContent = 'Error: ' + error;
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # Process image
        image_data = re.sub('^data:image/.+;base64,', '', data['image'])
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess for MNIST model
        image = image.convert('L').resize((28, 28))
        image_array = np.array(image)
        image_array = 255 - image_array  # Invert colors
        image_array = image_array.astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=(0, -1))
        
        # Make prediction
        pred = model.predict(image_array)
        digit = np.argmax(pred)
        confidence = float(np.max(pred)) * 100
        
        return jsonify({
            'digit': int(digit),
            'confidence': round(confidence, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)