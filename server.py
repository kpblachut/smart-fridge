from flask import Flask, request, Response
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('best.pt')  # Load the YOLO model (make sure to have the weights)

@app.route('/video_feed', methods=['POST'])
def video_feed():
    # Get the video frame from the request
    file = request.files['frame']
    in_frame = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(in_frame, cv2.IMREAD_COLOR)

    # Process the frame with YOLO
    results = model(frame)
    annotated_frame = results[0].plot()  # Annotate the frame

    # Encode the frame to send back
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)