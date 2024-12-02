import json
from ultralytics import YOLO

# Wczytaj model
model = YOLO('best.pt')

# Przetwarzanie wideo
results = model(source="./test2.mp4", show=True, conf=0.4)

# Przygotowanie danych do JSON
output_data = []
for frame in results:
    for detection in frame.boxes:
        xyxy = detection.xyxy.tolist()[0]  # Współrzędne
        label = detection.cls.tolist()[0]  # Klasa
        confidence = detection.conf.tolist()[0]  # Pewność (opcjonalnie)
        output_data.append({
            "xyxy": xyxy,
            "label": label,
            "confidence": confidence
        })

# Zapis wyników do pliku JSON
with open("output.json", "w") as json_file:
    json.dump(output_data, json_file, indent=4)