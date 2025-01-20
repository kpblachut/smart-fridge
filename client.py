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

        # Pobierz dodatkowe dane od użytkownika
        print("Podaj ilość produktu (np. 1.5):")
        quantity = input("> ")
        try:
            quantity = float(quantity)
        except ValueError:
            print("Nieprawidłowa ilość. Ustawiono domyślną wartość 1.")
            quantity = 1.0

        print("Podaj jednostkę (np. liters, kg):")
        unit = input("> ")

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)

        # Send the frame and additional data to the server
        response = requests.post(SERVER_URL,
                                 files={'frame': buffer.tobytes()},
                                 data={'quantity': quantity, 'unit': unit})

        # Decode the response from the server
        if response.status_code == 200:
            print("Produkt dodany do lodówki.")
        else:
            print(f"Błąd: {response.status_code}, {response.text}")

        # Wyświetl obraz z kamerki
        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
