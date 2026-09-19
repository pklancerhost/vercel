import base64
import json
import urllib.request
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image, ImageDraw


# ------------------------------
# Real object detection setup for common objects
# ------------------------------
COCO_CATEGORIES = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
MODEL = None


def build_model():
    """Load a COCO-trained object detection model once and reuse it."""
    global MODEL
    if MODEL is not None:
        return MODEL

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    MODEL = fasterrcnn_resnet50_fpn(weights=weights)
    MODEL.eval()
    return MODEL


def prepare_image(image_path):
    """Open the image and convert it to the format expected by the object detector."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transforms.ToTensor()(image)
    return [image_tensor]


def predict_object_label(image_path):
    """Return all detected objects with confidence and box information."""
    model = build_model()
    image_batch = prepare_image(image_path)

    with torch.no_grad():
        output = model(image_batch)

    result = output[0]
    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["labels"]

    keep = scores >= 0.1
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    detections = []
    for box, score, label_id in zip(boxes, scores, labels):
        label_index = int(label_id.item())
        label = COCO_CATEGORIES[label_index]
        confidence = float(score.item()) * 100
        detections.append({
            "label": label,
            "confidence": confidence,
            "box": box.tolist(),
        })

    if not detections:
        return [], "object", 0.0, None

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    detections = detections[:10]
    top = detections[0]
    return detections, top["label"], top["confidence"], top["box"]


def create_highlighted_image(image, boxes=None):
    """Draw green boxes around all detected objects when boxes are available."""
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(value) for value in box]
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 110), width=4)

    return result


# ------------------------------
# Simple web page
# ------------------------------
INDEX_HTML = """<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Simple Image Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6fb;
            display: flex;
            justify-content: center;
            padding: 30px;
        }
        .box {
            background: white;
            border-radius: 16px;
            padding: 24px;
            width: 900px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }
        h2 {
            text-align: center;
            margin-bottom: 20px;
        }
        .controls {
            display: flex;
            justify-content: center;
            gap: 12px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        input[type="file"] {
            border: 1px solid #cbd5e1;
            padding: 8px;
            border-radius: 8px;
            background: #f8fafc;
        }
        button {
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 18px;
            cursor: pointer;
            font-size: 16px;
        }
        .result {
            margin-top: 18px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
        }
        .label {
            font-size: 18px;
            font-weight: bold;
            margin: 8px 0;
        }
        .object { color: #1d4ed8; }
        .background { color: #8b4513; }
        img {
            display: block;
            max-width: 100%;
            margin: 0 auto 20px auto;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="box">
        <h2>Upload an image</h2>

        <div class="controls">
            <input id="imageInput" type="file" accept="image/*">
            <button id="analyzeBtn">Analyze</button>
        </div>

        <img id="resultImage" alt="result image" style="display:none;">

        <div class="result">
            <div id="objectLabel" class="label object">Top object: </div>
            <div id="objectList" class="label object"></div>
        </div>
    </div>

    <script>
        const imageInput = document.getElementById('imageInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const resultImage = document.getElementById('resultImage');
        const objectLabel = document.getElementById('objectLabel');
        const objectList = document.getElementById('objectList');

        analyzeBtn.addEventListener('click', async () => {
            const file = imageInput.files[0];
            if (!file) {
                alert('Please select an image first.');
                return;
            }

            analyzeBtn.disabled = true;
            analyzeBtn.textContent = 'Analyzing...';
            objectLabel.textContent = 'Top object: Processing...';
            objectList.textContent = '';

            const form = new FormData();
            form.append('image', file);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: form
                });

                const data = await response.json();
                const detections = data.detections || [];

                if (detections.length === 0) {
                    objectLabel.textContent = 'Top object: No object detected';
                    objectList.textContent = 'No detection results';
                    resultImage.style.display = 'none';
                    return;
                }

                const top = detections[0];
                objectLabel.textContent = 'Top object: ' + top.label + ' (' + Number(top.confidence).toFixed(2) + '%)';
                objectList.innerHTML = detections
                    .map(item => '<div>• ' + item.label + ' (' + Number(item.confidence).toFixed(2) + '%)</div>')
                    .join('');
                resultImage.src = data.imageData;
                resultImage.style.display = 'block';
            } catch (error) {
                objectLabel.textContent = 'Top object: Error';
                objectList.textContent = 'Could not analyze the image';
                alert('Error: ' + error.message);
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Analyze';
            }
        });
    </script>
</body>
</html>
"""


# ------------------------------
# Local web server
# ------------------------------
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404, "Not Found")
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(400, "Expected an uploaded image")
            return

        try:
            boundary = content_type.split("boundary=")[-1].encode()
            total_bytes = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(total_bytes)

            parts = data.split(b"--" + boundary)
            image_data = None

            for part in parts:
                if b"Content-Disposition" in part and b"filename" in part:
                    headers, body = part.split(b"\r\n\r\n", 1)
                    body = body.split(b"\r\n--", 1)[0]
                    image_data = body
                    break

            if image_data is None:
                self.send_error(400, "No image uploaded")
                return

            with open("uploaded_image_temp.png", "wb") as file:
                file.write(image_data)

            detections, label, confidence, box = predict_object_label("uploaded_image_temp.png")
            original_image = Image.open("uploaded_image_temp.png").convert("RGB")
            box_list = [item["box"] for item in detections]
            highlighted_image = create_highlighted_image(original_image, box_list)

            buffer = BytesIO()
            highlighted_image.save(buffer, format="PNG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

            response = {
                "label": label,
                "confidence": f"{confidence:.2f}",
                "detections": [
                    {
                        "label": item["label"],
                        "confidence": f"{item['confidence']:.2f}",
                    }
                    for item in detections
                ],
                "imageData": f"data:image/png;base64,{base64_image}",
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))

        except Exception as error:
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(error)}).encode("utf-8"))


def main():
    """Start the local server."""
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("Open your browser at: http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
