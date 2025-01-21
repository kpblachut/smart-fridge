import cv2
import requests
import numpy as np

SERVER_URL = 'http://192.168.0.94:5000/video_feed'

def main():
    cap = cv2.VideoCapture(0)  # Otwórz kamerę

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Błąd: Nie udało się przechwycić obrazu.")
            break

        # Kodowanie klatki do formatu JPEG
        _, buffer = cv2.imencode('.jpg', frame)

        try:
            # Wysyłanie klatki do serwera Flask
            response = requests.post(SERVER_URL, files={'image': buffer.tobytes()})

            if response.status_code == 200:
                # Parsowanie odpowiedzi JSON z serwera
                detections = response.json()
                print("Wykryte obiekty:", detections)

                # Rysowanie wykrytych obiektów na obrazie
                for detection in detections:
                    top_left = detection['topLeft']
                    bottom_right = detection['bottomRight']
                    label = detection.get('label', 'Unknown')
                    confidence = detection.get('confidence', 0.0)

                    # Rysowanie prostokąta wokół wykrytego obiektu
                    cv2.rectangle(frame, 
                                  (top_left['x'], top_left['y']), 
                                  (bottom_right['x'], bottom_right['y']), 
                                  (0, 255, 0), 2)

                    # Dodanie etykiety wykrytego obiektu
                    cv2.putText(frame, f"{label} ({confidence:.2f})", 
                                (top_left['x'], top_left['y'] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.9, (0, 255, 0), 2)

            else:
                print("Błąd odpowiedzi z serwera:", response.status_code)

        except Exception as e:
            print("Błąd komunikacji z serwerem:", str(e))

        # Wyświetlenie analizowanego obrazu
        cv2.imshow('Analyzed Frame', frame)

        # Naciśnięcie klawisza 'q' powoduje zakończenie działania
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
