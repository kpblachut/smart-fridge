import cv2
import requests
import numpy as np

SERVER_URL = 'http://127.0.0.1:5000/video_feed'

def main():
    cap = cv2.VideoCapture(0)  # Open the camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)

        # Send the frame to the server
        response = requests.post(SERVER_URL, files={'frame': buffer.tobytes()})

        # Decode the response from the server
        if response.status_code == 200:
            analyzed_frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow('Analyzed Frame', analyzed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()