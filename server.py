from flask import Flask, request, jsonify
import cv2
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'  # SQLite jako baza danych
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)

# Modele baz danych
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    #category = db.Column(db.String(50))
    #added_date = db.Column(db.Date, default=db.func.current_date)

class Recipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    instructions = db.Column(db.Text)

class Fridge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    quantity = db.Column(db.Numeric(10, 2), nullable=False)
    #added_date = db.Column(db.Date, default=db.func.current_date)

    __table_args__ = (
        db.CheckConstraint("status IN ('available', 'used', 'restored')", name="status_check"),
    )

class RecipeProduct(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    recipe_id = db.Column(db.Integer, db.ForeignKey('recipe.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Numeric(10, 2), nullable=False)
    unit = db.Column(db.String(50))

# Inicjalizacja YOLO
model = YOLO('best.pt')  # Podmień na właściwą ścieżkę do Twojego modelu

@app.before_request
def create_tables():
    if not hasattr(app, 'db_initialized'):
        db.create_all()
        app.db_initialized = True

@app.route('/api/products/add', methods=['POST'])
def add_product():
    data = request.json
    name = data.get('name')
    #category = data.get('category', 'General')
    quantity = data.get('quantity', 1.0)

    if not name:
        return jsonify({'error': 'Product name is required'}), 400

    # Spróbuj znaleźć produkt
    product = Product.query.filter_by(name=name).first()
    if not product:
        # Stwórz nowy
        product = Product(name=name)
        db.session.add(product)
        db.session.commit()

    # Tu zawsze tworzysz nowy wiersz w Fridge (jeśli tak chcesz):
    fridge_item = Fridge(product_id=product.id, status='available', quantity=quantity)
    db.session.add(fridge_item)
    db.session.commit()

    return jsonify({
        'message': 'Product added to fridge',
        'product': {
            'name': product.name,
            'quantity': quantity
        }
    }), 200

@app.route('/api/fridge', methods=['GET'])
def get_fridge():
    """
    Pobierz wszystkie produkty w lodówce.
    """
    fridge_items = Fridge.query.filter_by(status='available').all()
    result = []
    for item in fridge_items:
        product = Product.query.get(item.product_id)
        result.append({
            'id': item.id,
            'name': product.name,
            # 'category': product.category,
            'quantity': float(item.quantity),
            # 'added_date': item.added_date.isoformat()
        })

    return jsonify(result), 200

@app.route('/api/fridge/<int:product_id>', methods=['DELETE'])
def remove_from_fridge(product_id):
    """
    Usuń produkt z lodówki na podstawie ID produktu.
    """
    fridge_item = Fridge.query.filter_by(product_id=product_id, status='available').first()
    if not fridge_item:
        return jsonify({'error': 'Product not found in fridge'}), 404

    db.session.delete(fridge_item)
    db.session.commit()

    return jsonify({'message': 'Product removed from fridge'}), 200

@app.route('/detect', methods=['POST'])
def detect():
    """
    Odbiera obraz z kamery (plik 'image') i zwraca wykryte obiekty (etykieta, pewność i koordynaty).
    Nie dodaje produktu do bazy – to robi osobny endpoint /api/products/add.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    file = request.files['image']
    in_frame = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(in_frame, cv2.IMREAD_COLOR)

    results = model(frame)
    detections_data = []

    for detection in results[0].boxes:
        x1, y1, x2, y2 = detection.xyxy.tolist()[0]
        label_idx = int(detection.cls.tolist()[0])
        confidence = float(detection.conf.tolist()[0])

        # Nazwa klasy z modelu (jeśli jest dostępna)
        if hasattr(model, 'names') and label_idx in model.names:
            class_name = model.names[label_idx]
        else:
            class_name = f"Product_{label_idx}"

        detections_data.append({
            'label': class_name,
            'confidence': confidence,
            'topLeft': {'x': x1, 'y': y1},
            'bottomRight': {'x': x2, 'y': y2}
        })

    return jsonify(detections_data), 200

@app.route('/recipes', methods=['GET'])
def get_recipes():
    fridge_items = Fridge.query.filter_by(status="available").all()
    fridge_dict = {item.product_id: item.quantity for item in fridge_items}

    recipes = Recipe.query.all()
    possible_recipes = []

    for recipe in recipes:
        recipe_products = RecipeProduct.query.filter_by(recipe_id=recipe.id).all()
        can_make = True

        for rp in recipe_products:
            if rp.product_id not in fridge_dict or fridge_dict[rp.product_id] < rp.quantity:
                can_make = False
                break

        if can_make:
            possible_recipes.append({
                'id': recipe.id,
                'name': recipe.name,
                'description': recipe.description,
                'instructions': recipe.instructions
            })

    return jsonify(possible_recipes)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)