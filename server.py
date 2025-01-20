from flask import Flask, request, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'  # SQLite jako baza danych
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Modele baz danych
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50))
    added_date = db.Column(db.Date, default=db.func.current_date)

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
    added_date = db.Column(db.Date, default=db.func.current_date)

    __table_args__ = (
        db.CheckConstraint("status IN ('available', 'used', 'restored')", name="status_check"),
    )

class RecipeProduct(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    recipe_id = db.Column(db.Integer, db.ForeignKey('recipe.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Numeric(10, 2), nullable=False)
    unit = db.Column(db.String(50))

# YOLO model
model = YOLO('best.pt')

@app.before_request
def create_tables():
    if not hasattr(app, 'db_initialized'):
        db.create_all()
        app.db_initialized = True

@app.route('/video_feed', methods=['POST'])
def video_feed():
    file = request.files['frame']
    in_frame = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(in_frame, cv2.IMREAD_COLOR)

    # Process the frame with YOLO
    results = model(frame)
    for detection in results[0].boxes:
        label = detection.cls.tolist()[0]  # Klasa
        confidence = detection.conf.tolist()[0]

        # Przykład: Użycie label jako nazwy produktu
        product_name = f"Product_{int(label)}"

        # Dodaj produkt do bazy, jeśli nie istnieje
        product = Product.query.filter_by(name=product_name).first()
        if not product:
            product = Product(name=product_name, category="General")
            db.session.add(product)
            db.session.commit()

        # Dodaj produkt do lodówki
        fridge_item = Fridge.query.filter_by(product_id=product.id).first()
        if fridge_item:
            fridge_item.quantity += 1  # Zwiększ ilość (przykładowo)
        else:
            # Użytkownik podaje ilość i jednostkę
            quantity = float(request.form.get('quantity', 1.0))
            fridge_item = Fridge(product_id=product.id, status="available", quantity=quantity)
            db.session.add(fridge_item)
        db.session.commit()

    return jsonify({'message': 'Products processed and added to fridge.'})

@app.route('/recipes', methods=['GET'])
def get_recipes():
    # Pobierz wszystkie produkty z lodówki
    fridge_items = Fridge.query.filter_by(status="available").all()
    fridge_dict = {item.product_id: item.quantity for item in fridge_items}

    # Znajdź przepisy, które mogą być wykonane
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
