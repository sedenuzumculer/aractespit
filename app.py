from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

model = YOLO('yolov8n.pt')

parking_slots = {
    "A-1": "OCCUPIED",
    "A-2": "AVAILABLE",
    "A-3": "AVAILABLE",
    "A-4": "OCCUPIED",
    "A-5": "AVAILABLE",
    "A-6": "OCCUPIED",
    "A-7": "OCCUPIED",
    "A-8": "OCCUPIED",
    "A-9": "OCCUPIED",
    "A-10": "OCCUPIED",
    "A-11": "OCCUPIED",
    "A-12": "OCCUPIED",
    "A-13": "OCCUPIED",
    "A-14": "OCCUPIED",
    "A-15": "OCCUPIED",
    "A-16": "OCCUPIED",
    "A-17": "OCCUPIED",
    "A-18": "OCCUPIED",
}

parking_spaces = {
    "A-1": [11, 1795, 690, 2261],
    "A-2": [698, 1991, 1214, 2236],
    "A-3": [1320, 1900, 1934, 2100],
    "A-4": [1876, 1736, 2392, 2048],
    "A-5": [2343, 1895, 2916, 2065],
    "A-6": [698, 1991, 1214, 2236],
    "A-7": [2556, 1745, 3186, 1991],
    "A-8": [2965, 1705, 3578, 1942],
    "A-9": [3300, 1690, 3845, 1925],
    "A-10": [170, 1723, 487, 1814],
    "A-11": [501, 1710, 841, 1873],
    "A-12": [859, 1705, 1126, 1868],
    "A-13": [1186, 1691, 1394, 1850],
    "A-14": [1435, 1696, 1612, 1818],
    "A-15": [1662, 1691, 1825, 1832],
    "A-16": [1884, 1678, 1830, 1823],
    "A-17": [1875, 1700, 2052, 1796],
    "A-18": [2412, 1691, 2587, 1796],
}

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/update_parking', methods=['POST'])
def update_parking():
    data = request.json
    slot = data['slot']
    status = data['status']
    parking_slots[slot] = status
    return jsonify(parking_slots)

@app.route('/get_parking_status', methods=['GET'])
def get_parking_status():
    return jsonify(parking_slots)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'Resim bulunamadı'}), 400
    
    image_file = request.files['image']
    image_bytes = image_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)

    def box_overlaps_space(box, space, threshold=0.5):
        """Bir sınırlayıcı kutunun bir park yeri bölgesiyle örtüşüp örtüşmediğini kontrol eder."""
        space_x1, space_y1, space_x2, space_y2 = space

        box_x1, box_y1, box_x2, box_y2 = box

        x_overlap = max(0, min(box_x2, space_x2) - max(box_x1, space_x1))
        y_overlap = max(0, min(box_y2, space_y2) - max(box_y1, space_y1))

        intersection_area = x_overlap * y_overlap
        box_area = (box_x2 - box_x1) * (box_y2 - box_y1)
        space_area = (space_x2 - space_x1) * (space_y2 - space_y1)
        union_area = box_area + space_area - intersection_area

        iou = intersection_area / union_area

        return iou > threshold

    for slot, space in parking_spaces.items():
        is_empty = True
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if model.names[int(box.cls[0])] == 'car' and box_overlaps_space(box.xyxy[0], space):
                    is_empty = False
                    break
        parking_slots[slot] = "AVAILABLE" if is_empty else "OCCUPIED"

    return jsonify(parking_slots)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
