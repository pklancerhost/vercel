import base64
import io

from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageDraw, UnidentifiedImageError
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

app = Flask(__name__)

WEIGHTS = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
MODEL = fasterrcnn_resnet50_fpn(weights=WEIGHTS)
MODEL.eval()
CATEGORIES = WEIGHTS.meta["categories"]

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Image Object Detector</title>
    <style>
        :root { color-scheme: light; font-family: Arial, sans-serif; }
        body { margin: 0; min-height: 100vh; background: #eef3f8; color: #17202a; }
        main { width: min(900px, calc(100% - 32px)); margin: 40px auto; }
        h1 { margin-bottom: 8px; }
        .intro { color: #52616b; }
        .controls, .results { background: white; border: 1px solid #d7e0e8; border-radius: 12px; padding: 20px; }
        .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
        input[type=file] { max-width: 100%; }
        button { border: 0; border-radius: 7px; padding: 11px 18px; background: #087f5b; color: white; font-weight: bold; cursor: pointer; }
        button:disabled { opacity: .6; cursor: wait; }
        .results { margin-top: 18px; display: none; }
        img { display: block; max-width: 100%; max-height: 560px; margin: 0 auto 18px; }
        .top { font-size: 20px; font-weight: bold; margin-bottom: 12px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 9px; border-top: 1px solid #e4e9ee; }
        .message { color: #52616b; }
        @media (max-width: 520px) { main { margin: 22px auto; } .controls { align-items: stretch; flex-direction: column; } }
    </style>
</head>
<body>
    <main>
        <h1>Image Object Detector</h1>
        <p class="intro">Upload an image to detect objects with confidence above 50%.</p>
        <form class="controls" id="form">
            <input id="image" name="image" type="file" accept="image/*" required>
            <button id="button" type="submit">Analyze image</button>
        </form>
        <section class="results" id="results">
            <img id="preview" alt="Annotated detection result">
            <div class="top" id="top"></div>
            <div id="message" class="message"></div>
            <table id="detections"><thead><tr><th>Object</th><th>Confidence</th><th>Box</th></tr></thead><tbody></tbody></table>
        </section>
    </main>
    <script>
        const form = document.getElementById('form');
        const button = document.getElementById('button');
        const results = document.getElementById('results');
        const preview = document.getElementById('preview');
        const topResult = document.getElementById('top');
        const message = document.getElementById('message');
        const table = document.querySelector('#detections tbody');

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            button.disabled = true;
            button.textContent = 'Analyzing...';
            results.style.display = 'block';
            message.textContent = 'Processing image...';
            table.replaceChildren();
            try {
                const response = await fetch('/predict', { method: 'POST', body: new FormData(form) });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'Could not analyze image');
                const detections = data.detections || [];
                preview.src = data.imageData;
                if (!detections.length) {
                    topResult.textContent = 'No object detected with enough confidence.';
                    message.textContent = '';
                    return;
                }
                topResult.textContent = `Detected Object: ${detections[0].label} (${detections[0].confidence.toFixed(2)}%)`;
                message.textContent = `${detections.length} detection${detections.length === 1 ? '' : 's'}`;
                for (const item of detections) {
                    const row = table.insertRow();
                    row.insertCell().textContent = item.label;
                    row.insertCell().textContent = `${item.confidence.toFixed(2)}%`;
                    row.insertCell().textContent = `[${item.box.map(value => value.toFixed(2)).join(', ')}]`;
                }
            } catch (error) {
                topResult.textContent = 'Detection failed';
                message.textContent = error.message;
                preview.removeAttribute('src');
            } finally {
                button.disabled = false;
                button.textContent = 'Analyze image';
            }
        });
    </script>
</body>
</html>"""


def detect_objects(image):
    image_tensor = transforms.ToTensor()(image)

    with torch.inference_mode():
        result = MODEL([image_tensor])[0]

    detections = []
    for box, score, label_id in zip(result["boxes"], result["scores"], result["labels"]):
        confidence = float(score) * 100
        if confidence < 50:
            continue
        detections.append({
            "label": CATEGORIES[int(label_id)],
            "confidence": confidence,
            "box": [float(value) for value in box],
        })

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections


def annotate_image(image, detections):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for detection in detections:
        box = [int(value) for value in detection["box"]]
        draw.rectangle(box, outline=(0, 220, 110), width=5)
        draw.text((box[0] + 6, box[1] + 6), detection["label"], fill=(0, 220, 110))
    output = io.BytesIO()
    result.save(output, format="JPEG", quality=90)
    return base64.b64encode(output.getvalue()).decode("ascii")


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.post("/predict")
def predict():
    uploaded_file = request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify(error="Please select an image"), 400

    try:
        image = Image.open(uploaded_file.stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="The uploaded file is not a valid image"), 400

    detections = detect_objects(image)
    return jsonify(
        detections=detections,
        imageData="data:image/jpeg;base64," + annotate_image(image, detections),
    )


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="Image must be smaller than 20 MB"), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
