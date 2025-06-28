import sqlite3
from datetime import datetime, timedelta
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from app import db, Order, Vehicle, encrypt_data  # Import from your app.py
import json
import logging
from zoneinfo import ZoneInfo

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup (needed for app context)
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/sanja/LOG/instance/logistics.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Mock road network from app.py (for route calculation)
ROAD_NETWORK = {
    'A': {
        'B': {'distance': 5, 'traffic_multiplier': 1.2, 'toll': 2, 'restricted': False, 'speed_limit': 60},
        'C': {'distance': 10, 'traffic_multiplier': 1.5, 'toll': 0, 'restricted': False, 'speed_limit': 50}
    },
    'B': {
        'A': {'distance': 5, 'traffic_multiplier': 1.2, 'toll': 2, 'restricted': False, 'speed_limit': 60},
        'D': {'distance': 8, 'traffic_multiplier': 1.0, 'toll': 1, 'restricted': True, 'speed_limit': 70}
    },
    'C': {
        'A': {'distance': 10, 'traffic_multiplier': 1.5, 'toll': 0, 'restricted': False, 'speed_limit': 50},
        'D': {'distance': 3, 'traffic_multiplier': 1.1, 'toll': 0, 'restricted': False, 'speed_limit': 80}
    },
    'D': {
        'B': {'distance': 8, 'traffic_multiplier': 1.0, 'toll': 1, 'restricted': True, 'speed_limit': 70},
        'C': {'distance': 3, 'traffic_multiplier': 1.1, 'toll': 0, 'restricted': False, 'speed_limit': 80}
    }
}

def calculate_route_distance(route):
    distance = 0
    for i in range(len(route) - 1):
        start, end = route[i], route[i + 1]
        edge = ROAD_NETWORK.get(start, {}).get(end, {})
        distance += edge.get('distance', 0)
    return distance

def add_orders(num_orders=20):
    with app.app_context():
        # Fetch available vehicles
        vehicles = Vehicle.query.all()
        if not vehicles:
            logger.error("No vehicles found in database")
            return

        # Define sample order data
        products = ["Laptop", "Phone", "Books", "Clothes", "Electronics"]
        locations = ["pallikaranai", "tambaram", "delhi", "city_center", "chennai", "mumbai"]
        urgencies = ["Low", "Medium", "High"]
        now = datetime.now(ZoneInfo("UTC"))

        for i in range(num_orders):
            # Randomize order details
            product = products[i % len(products)]
            pickup_location = locations[i % len(locations)]
            delivery_location = locations[(i + 1) % len(locations)]  # Ensure different locations
            urgency = urgencies[i % len(urgencies)]
            weight = float(i + 1) * 5  # 5, 10, 15, ...
            volume = float(i + 1) * 0.5  # 0.5, 1.0, 1.5, ...
            preferred_time = now + timedelta(hours=(i + 2))  # 2, 3, 4 hours from now
            status = "Delivered" if i % 2 == 0 else "In Transit"  # Mix statuses
            vehicle = vehicles[i % len(vehicles)]

            # Encrypt locations
            encrypted_pickup, pickup_key = encrypt_data(pickup_location)
            encrypted_delivery, delivery_key = encrypt_data(delivery_location)

            # Define route (simple A->B or B->A for simplicity)
            route = ["A", "B"] if i % 2 == 0 else ["B", "A"]
            route_json = json.dumps(route)
            distance = calculate_route_distance(route)
            fuel_consumed = distance * vehicle.fuel_rate  # Realistic fuel consumption
            cost = distance * vehicle.fuel_rate * 10 + 2  # Simplified cost
            travel_time = distance / vehicle.max_speed  # Simplified travel time
            eta = preferred_time if urgency == "High" else now + timedelta(hours=travel_time + 1)
            dispatch_time = now - timedelta(hours=i)  # Simulate past dispatch
            last_updated = now
            created_at = dispatch_time - timedelta(minutes=30)

            # Create order
            order = Order(
                partner_id=1,  # From logs (partner1)
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
                route=route_json,
                cost=cost,
                vehicle_id=vehicle.id,
                eta=eta,
                estimated_travel_time=travel_time,
                dispatch_time=dispatch_time,
                current_location=route[0] if status == "In Transit" else route[-1],
                fuel_consumed=fuel_consumed,
                last_updated=last_updated,
                created_at=created_at
            )
            db.session.add(order)
            logger.info(f"Added order {i+1}: {product} from {pickup_location} to {delivery_location}, Status: {status}")

        # Commit changes
        try:
            db.session.commit()
            logger.info(f"Successfully added {num_orders} orders")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to add orders: {str(e)}")

if __name__ == "__main__":
    add_orders(num_orders=20)  # Add 5 orders
