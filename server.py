from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('best.pt')  # Załaduj model YOLO

@app.route('/video_feed', methods=['POST'])
def video_feed():
    try:
        file = request.files['image']
        in_frame = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(in_frame, cv2.IMREAD_COLOR)

        # Wykrywanie obiektów za pomocą YOLO
        results = model(frame)

        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Pobranie współrzędnych detekcji
            confidences = result.boxes.conf.cpu().numpy()  # Pobranie pewności detekcji
            classes = result.boxes.cls.cpu().numpy()  # Pobranie klas detekcji

            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    "topLeft": {"x": x1, "y": y1},
                    "bottomRight": {"x": x2, "y": y2},
                    "confidence": float(conf),
                    "label": model.names[int(cls)] if hasattr(model, 'names') else f"class_{int(cls)}"
                })

        print("Wysłane do klienta:", detections)
        return jsonify(detections)

    except Exception as e:
        print(f"Błąd przetwarzania: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
