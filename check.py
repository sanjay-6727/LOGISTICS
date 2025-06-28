
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash
from app import db, Partner, Order, Vehicle, ROAD_NETWORK, map_location_to_node, encrypt_data, allocate_vehicle, a_star_route, calculate_route_distance

# Initialize Flask app for database context
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///logistics.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Valid locations for orders
LOCATIONS = ['pallikaranai', 'tambaram', 'delhi', 'warehouse', 'city_center', 'chennai', 'mumbai']
PRODUCTS = ['Electronics', 'Furniture', 'Groceries', 'Clothing', 'Appliances', 'Books', 'Pharmaceuticals']
URGENCIES = ['High', 'Medium', 'Low']
STATUSES = ['Pending', 'In Transit', 'Delivered']

def ensure_default_partner():
    """Ensure at least one partner exists in the database."""
    with app.app_context():
        partner = Partner.query.first()
        if not partner:
            partner = Partner(
                username='sample_partner',
                password_hash=generate_password_hash('password123')
            )
            db.session.add(partner)
            db.session.commit()
            print("Created default partner: sample_partner")
        return partner.id

def generate_random_order(partner_id, vehicles):
    """Generate a single random order."""
    # Random attributes
    product = random.choice(PRODUCTS)
    pickup_location = random.choice(LOCATIONS)
    delivery_location = random.choice([loc for loc in LOCATIONS if loc != pickup_location])
    preferred_time = datetime.now(ZoneInfo("UTC")) + timedelta(days=random.randint(0, 7), hours=random.randint(0, 23))
    urgency = random.choice(URGENCIES)
    weight = round(random.uniform(0.5, 1000), 1)
    volume = round(random.uniform(0.1, 50), 1)
    status = random.choices(STATUSES, weights=[0.3, 0.4, 0.3])[0]  # Weighted for realism
    created_at = datetime.now(ZoneInfo("UTC")) - timedelta(days=random.randint(0, 30))

    # Encrypt locations
    encrypted_pickup, pickup_key = encrypt_data(pickup_location)
    encrypted_delivery, delivery_key = encrypt_data(delivery_location)

    # Map locations to nodes
    try:
        pickup_node = map_location_to_node(pickup_location)
        delivery_node = map_location_to_node(delivery_location)
    except ValueError as e:
        print(f"Error mapping locations: {e}")
        return None

    # Allocate vehicle
    vehicle, _ = allocate_vehicle(weight, volume, urgency, [pickup_node, delivery_node])
    if not vehicle:
        print(f"No suitable vehicle for order: {product}, {weight}kg, {volume}m³")
        return None

    # Calculate route
    route, cost, travel_time = a_star_route(pickup_node, delivery_node, preferred_time, vehicle)
    if not route:
        print(f"No route found from {pickup_location} to {delivery_location}")
        return None

    # Simulate order progress
    eta = preferred_time + timedelta(hours=travel_time)
    dispatch_time = None
    current_location = pickup_node
    fuel_consumed = 0.0
    last_updated = created_at

    if status == 'In Transit':
        dispatch_time = created_at + timedelta(hours=random.uniform(0, 2))
        progress = random.uniform(0.1, 0.9)  # Partial progress
        segment_index = min(int(len(route) * progress), len(route) - 2)
        current_location = route[segment_index]
        distance_traveled = calculate_route_distance(route[:segment_index + 1])
        fuel_consumed = distance_traveled * vehicle.fuel_rate
        last_updated = dispatch_time + timedelta(hours=progress * travel_time)
    elif status == 'Delivered':
        dispatch_time = created_at + timedelta(hours=random.uniform(0, 2))
        current_location = delivery_node
        distance_traveled = calculate_route_distance(route)
        fuel_consumed = distance_traveled * vehicle.fuel_rate
        last_updated = dispatch_time + timedelta(hours=travel_time)

    # Create order
    order = Order(
        partner_id=partner_id,
        product=product,
        pickup_location=pickup_location,
        delivery_location=delivery_location,
        encrypted_pickup_location=encrypted_pickup,
        pickup_location_key=pickup_key,
        encrypted_delivery_location=encrypted_delivery,
        delivery_location_key=delivery_key,
        preferred_time=preferred_time,
        urgency=urgency,
        weight=weight,
        volume=volume,
        status=status,
        route=json.dumps(route),
        cost=cost,
        vehicle_id=vehicle.id,
        eta=eta,
        estimated_travel_time=travel_time,
        dispatch_time=dispatch_time,
        current_location=current_location,
        fuel_consumed=fuel_consumed,
        created_at=created_at,
        last_updated=last_updated
    )
    return order

def add_sample_deliveries(num_orders=100):
    """Add sample deliveries to the database."""
    with app.app_context():
        # Ensure a partner exists
        partner_id = ensure_default_partner()

        # Get available vehicles
        vehicles = Vehicle.query.all()
        if not vehicles:
            print("No vehicles found in the database. Please add vehicles first.")
            return

        # Generate and insert orders
        orders_added = 0
        for i in range(num_orders):
            order = generate_random_order(partner_id, vehicles)
            if order:
                db.session.add(order)
                orders_added += 1
                if orders_added % 10 == 0:
                    db.session.commit()
                    print(f"Added {orders_added} orders...")
        
        # Final commit
        db.session.commit()
        print(f"Successfully added {orders_added} sample deliveries.")

if __name__ == "__main__":
    try:
        add_sample_deliveries(100)
    except Exception as e:
        print(f"Error adding sample deliveries: {str(e)}")
