import os
# Force CPU execution & lower TF memory footprint before importing TensorFlow/Keras
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gc
import re
import base64
import numpy as np
from PIL import Image
import io
import traceback
from flask import Flask, render_template_string, request, jsonify
import keras

app = Flask(__name__)

# Search and assign the correct available model file
MODEL_PATH = None
for candidate in ["digit_recognizer.keras", "digit_recognizer.h5", "model.keras", "model.h5"]:
    if os.path.exists(candidate):
        MODEL_PATH = candidate
        break

model = None
if MODEL_PATH:
    try:
        model = keras.models.load_model(MODEL_PATH)
        print(f"Model successfully loaded from {MODEL_PATH}")
        
        # Warm-up pass to pre-allocate TensorFlow internal graph in RAM
        dummy_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
        _ = model.predict(dummy_input, verbose=0)
        print("Model warm-up complete.")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
else:
    print("Warning: No valid .keras or .h5 model file found in project root.")

def preprocess_image(image_bytes):
    """
    Preprocess image to match model input shape: (1, 28, 28, 1)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # Auto-crop bounding box to center content if drawing
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        # Add padding to maintain aspect ratio
        w, h = img.size
        max_dim = max(w, h)
        padded_img = Image.new('L', (max_dim, max_dim), 0)
        padded_img.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
        img = padded_img
        
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Shape: (1, 28, 28, 1)
    return img_array

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Keras CNN Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            max-width: 900px;
            width: 100%;
            background: var(--card-bg);
            border-radius: 20px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
            padding: 40px;
            text-align: center;
        }

        h1 {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(to right, #a5b4fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle {
            color: var(--text-secondary);
            margin-bottom: 30px;
        }

        .workspace {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            align-items: start;
        }

        .canvas-card, .result-card {
            background: rgba(15, 23, 42, 0.6);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid #334155;
        }

        canvas {
            background: #000;
            border-radius: 12px;
            cursor: crosshair;
            touch-action: none;
            border: 2px solid #334155;
            box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.6);
        }

        .controls {
            margin-top: 15px;
            display: flex;
            gap: 10px;
            justify-content: center;
        }

        button, .file-label {
            background: var(--accent-color);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 0.9rem;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        button:hover, .file-label:hover {
            background: var(--accent-hover);
            transform: translateY(-2px);
        }

        .btn-secondary {
            background: #475569;
        }

        .btn-secondary:hover {
            background: #334155;
        }

        input[type="file"] {
            display: none;
        }

        .result-box {
            min-height: 280px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }

        .prediction-class {
            font-size: 5rem;
            font-weight: 800;
            color: #818cf8;
            margin: 10px 0;
        }

        .confidence-meter {
            width: 100%;
            background: #334155;
            border-radius: 8px;
            height: 10px;
            overflow: hidden;
            margin-top: 10px;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #6366f1, #22c55e);
            width: 0%;
            transition: width 0.4s ease;
        }

        .prob-list {
            margin-top: 20px;
            width: 100%;
            text-align: left;
            font-size: 0.85rem;
            color: var(--text-secondary);
            max-height: 150px;
            overflow-y: auto;
        }

        .prob-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
    </style>
</head>
<body>

    <div class="container">
        <h1>CNN Model Predictor</h1>
        <p class="subtitle">Draw or upload an image to test the neural network</p>

        <div class="workspace">
            <div class="canvas-card">
                <canvas id="paintCanvas" width="280" height="280"></canvas>
                <div class="controls">
                    <button class="btn-secondary" onclick="clearCanvas()">Clear</button>
                    <label class="file-label">
                        Upload
                        <input type="file" id="imageInput" accept="image/*" onchange="uploadImage(event)">
                    </label>
                    <button onclick="predictCanvas()">Predict</button>
                </div>
            </div>

            <div class="result-card">
                <h3>Prediction Output</h3>
                <div class="result-box" id="resultBox">
                    <p style="color: var(--text-secondary);">Draw or upload to see results</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('paintCanvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;

        // Initialize Canvas
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = "white";
        ctx.lineWidth = 18;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        function startDrawing(e) {
            isDrawing = true;
            draw(e);
        }

        function stopDrawing() {
            isDrawing = false;
            ctx.beginPath();
        }

        function draw(e) {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX || e.touches[0].clientX) - rect.left;
            const y = (e.clientY || e.touches[0].clientY) - rect.top;

            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
        }

        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mousemove', draw);

        canvas.addEventListener('touchstart', startDrawing);
        canvas.addEventListener('touchend', stopDrawing);
        canvas.addEventListener('touchmove', draw);

        function clearCanvas() {
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('resultBox').innerHTML = `<p style="color: var(--text-secondary);">Draw or upload to see results</p>`;
        }

        async function predictCanvas() {
            const dataURL = canvas.toDataURL('image/png');
            sendPredictionRequest({ image: dataURL });
        }

        function uploadImage(e) {
            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(event) {
                const img = new Image();
                img.onload = function() {
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    sendPredictionRequest({ image: event.target.result });
                }
                img.src = event.target.result;
            }
            reader.readAsDataURL(file);
        }

        async function sendPredictionRequest(payload) {
            const resultBox = document.getElementById('resultBox');
            resultBox.innerHTML = '<p>Processing...</p>';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const responseText = await response.text();

                if (!response.ok) {
                    throw new Error(`Server returned status ${response.status}: ${responseText}`);
                }

                if (!responseText) {
                    throw new Error("Empty response received from server.");
                }

                const data = JSON.parse(responseText);

                if (data.error) throw new Error(data.error);

                let probListHTML = '';
                data.probabilities.forEach((prob, idx) => {
                    probListHTML += `
                        <div class="prob-row">
                            <span>Class ${idx}</span>
                            <span>${(prob * 100).toFixed(1)}%</span>
                        </div>`;
                });

                resultBox.innerHTML = `
                    <div>
                        <div class="prediction-class">${data.prediction}</div>
                        <p style="color: var(--text-secondary);">Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
                        <div class="confidence-meter">
                            <div class="confidence-fill" style="width: ${data.confidence * 100}%"></div>
                        </div>
                    </div>
                    <div class="prob-list">${probListHTML}</div>
                `;
            } catch (err) {
                resultBox.innerHTML = `<p style="color: #ef4444; word-break: break-all;">Error: ${err.message}</p>`;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model file not found or failed to load on backend.'}), 500

    try:
        data = request.get_json(silent=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data payload received.'}), 400

        image_data = data['image']
        
        # Clean up Base64 string prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        processed_input = preprocess_image(image_bytes)
        
        preds = model.predict(processed_input, verbose=0)[0]
        pred_class = int(np.argmax(preds))
        confidence = float(np.max(preds))
        
        # Free memory immediately
        gc.collect()

        return jsonify({
            'prediction': pred_class,
            'confidence': confidence,
            'probabilities': preds.tolist()
        }), 200

    except Exception as e:
        print("Backend prediction error:\n", traceback.format_exc())
        return jsonify({'error': f'Prediction processing error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
