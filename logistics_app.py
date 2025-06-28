from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
import heapq
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64
import logging
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from werkzeug.security import generate_password_hash, check_password_hash
from math import cos, sin, radians
from io import BytesIO

app = Flask(__name__)

# Ensure the instance directory exists
instance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
os.makedirs(instance_dir, exist_ok=True)

# Use absolute path for SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(instance_dir, "logistics.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
db = SQLAlchemy(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Database path: {app.config['SQLALCHEMY_DATABASE_URI']}")

# Define location pair distances
location_pair_distances = {
    ('pallikaranai', 'delhi'): 10,
    ('pallikaranai', 'tambaram'): 5,
    ('pallikaranai', 'warehouse'): 5,
    ('pallikaranai', 'city_center'): 12,
    ('pallikaranai', 'chennai'): 0,
    ('pallikaranai', 'mumbai'): 15,
    ('tambaram', 'delhi'): 10,
    ('tambaram', 'pallikaranai'): 5,
    ('tambaram', 'warehouse'): 0,
    ('tambaram', 'city_center'): 10,
    ('tambaram', 'chennai'): 5,
    ('tambaram', 'mumbai'): 12,
    ('delhi', 'pallikaranai'): 10,
    ('delhi', 'tambaram'): 10,
    ('delhi', 'warehouse'): 10,
    ('delhi', 'city_center'): 3,
    ('delhi', 'chennai'): 10,
    ('delhi', 'mumbai'): 3,
    ('warehouse', 'pallikaranai'): 5,
    ('warehouse', 'tambaram'): 0,
    ('warehouse', 'delhi'): 10,
    ('warehouse', 'city_center'): 10,
    ('warehouse', 'chennai'): 5,
    ('warehouse', 'mumbai'): 12,
    ('city_center', 'pallikaranai'): 12,
    ('city_center', 'tambaram'): 10,
    ('city_center', 'delhi'): 3,
    ('city_center', 'warehouse'): 10,
    ('city_center', 'chennai'): 12,
    ('city_center', 'mumbai'): 0,
    ('chennai', 'pallikaranai'): 0,
    ('chennai', 'tambaram'): 5,
    ('chennai', 'delhi'): 10,
    ('chennai', 'warehouse'): 5,
    ('chennai', 'city_center'): 12,
    ('chennai', 'mumbai'): 15,
    ('mumbai', 'pallikaranai'): 15,
    ('mumbai', 'tambaram'): 12,
    ('mumbai', 'delhi'): 3,
    ('mumbai', 'warehouse'): 12,
    ('mumbai', 'chennai'): 15,
    ('mumbai', 'city_center'): 0,
}

def encrypt_data(data):
    key = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data.encode())
    return base64.b64encode(cipher.nonce + tag + ciphertext).decode(), base64.b64encode(key).decode()

def decrypt_data(encrypted_data, key):
    try:
        data = base64.b64decode(encrypted_data)
        key = base64.b64decode(key)
        nonce, tag, ciphertext = data[:16], data[16:32], data[32:]
        cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
        return cipher.decrypt_and_verify(ciphertext, tag).decode()
    except Exception as e:
        logger.error(f"Decryption error: {str(e)}")
        return None

class Partner(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    partner_id = db.Column(db.Integer, db.ForeignKey('partner.id'), nullable=False)
    product = db.Column(db.String(100), nullable=False)
    pickup_location = db.Column(db.String(200), nullable=False)
    delivery_location = db.Column(db.String(200), nullable=False)
    encrypted_pickup_location = db.Column(db.String(500), nullable=False)
    pickup_location_key = db.Column(db.String(500), nullable=False)
    encrypted_delivery_location = db.Column(db.String(500), nullable=False)
    delivery_location_key = db.Column(db.String(500), nullable=False)
    preferred_time = db.Column(db.DateTime, nullable=False)
    urgency = db.Column(db.String(20), nullable=False)
    weight = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(50), default='Pending')
    route = db.Column(db.Text)
    cost = db.Column(db.Float)
    vehicle_id = db.Column(db.Integer, db.ForeignKey('vehicle.id'))
    vehicle = db.relationship('Vehicle', backref='orders', lazy='select')
    eta = db.Column(db.DateTime)
    estimated_travel_time = db.Column(db.Float)
    dispatch_time = db.Column(db.DateTime)
    current_location = db.Column(db.String(10))
    fuel_consumed = db.Column(db.Float, default=0.0)
    last_updated = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(ZoneInfo("UTC")))

class Vehicle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(50), nullable=False)
    capacity_weight = db.Column(db.Float, nullable=False)
    capacity_volume = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), default='Available')
    fuel_rate = db.Column(db.Float, nullable=False)
    max_speed = db.Column(db.Float, nullable=False)

class Inventory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    partner_id = db.Column(db.Integer, db.ForeignKey('partner.id'), nullable=False)
    product = db.Column(db.String(100), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    location = db.Column(db.String(200), nullable=False)
    last_updated = db.Column(db.DateTime, default=lambda: datetime.now(ZoneInfo("UTC")))

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

NODE_COORDINATES = {
    'A': {'lat': 12.9345, 'lng': 80.2321, 'name': 'Pallikaranai/Chennai'},
    'B': {'lat': 12.9250, 'lng': 80.1150, 'name': 'Tambaram/Warehouse'},
    'C': {'lat': 19.0760, 'lng': 72.8777, 'name': 'Mumbai/City Center'},
    'D': {'lat': 28.7041, 'lng': 77.1025, 'name': 'Delhi'}
}

def get_traffic_multiplier(base_multiplier, preferred_time):
    hour = preferred_time.hour
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        return base_multiplier * 1.5
    return base_multiplier

def map_location_to_node(location):
    location_map = {
        'pallikaranai': 'A',
        'tambaram': 'B',
        'delhi': 'D',
        'warehouse': 'B',
        'city_center': 'C',
        'chennai': 'A',
        'mumbai': 'C'
    }
    node = location_map.get(location.lower())
    if not node:
        valid_locations = ', '.join(location_map.keys())
        logger.warning(f"Location '{location}' not found. Valid locations: {valid_locations}")
        raise ValueError(f"Invalid location '{location}'. Valid locations: {valid_locations}")
    return node

def calculate_route_distance(route):
    distance = 0
    for i in range(len(route) - 1):
        start, end = route[i], route[i + 1]
        edge = ROAD_NETWORK.get(start, {}).get(end, {})
        distance += edge.get('distance', 0)
    return distance

def allocate_vehicle(weight, volume, urgency, route):
    vehicles = Vehicle.query.filter_by(status='Available').all()
    if not vehicles:
        logger.warning("No vehicles available")
        return None, None

    route_distance = calculate_route_distance(route)
    logger.info(f"Allocating vehicle for weight: {weight}kg, volume: {volume}m³, urgency: {urgency}, distance: {route_distance}km")

    def vehicle_score(vehicle):
        score = 0
        if vehicle.capacity_weight < weight or vehicle.capacity_volume < volume:
            return float('-inf')
        if urgency == 'High':
            score += vehicle.max_speed * 2
        elif urgency == 'Low':
            score += vehicle.max_speed * 0.5
        if route_distance > 20:
            score -= vehicle.fuel_rate * 100
        else:
            score -= vehicle.fuel_rate * 50
        score += (vehicle.capacity_weight - weight) * 0.1
        score += (vehicle.capacity_volume - volume) * 0.1
        return score

    best_vehicle = max(vehicles, key=vehicle_score, default=None)
    if vehicle_score(best_vehicle) == float('-inf'):
        logger.warning("No vehicle has sufficient capacity")
        return None, None

    logger.info(f"Selected vehicle: {best_vehicle.type} (ID: {best_vehicle.id})")
    return best_vehicle, vehicle_score(best_vehicle)

def a_star_route(start, goal, preferred_time, vehicle, traffic_multiplier_override=None):
    logger.info(f"Calculating route from {start} to {goal} at {preferred_time}")
    open_set = [(0, start, [], 0)]
    closed_set = set()
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current_f, current, path, travel_time = heapq.heappop(open_set)

        if current == goal:
            logger.info(f"Route found: {path + [current]}, Cost: {g_score[current]}, Time: {travel_time}")
            return path + [current], g_score[current], travel_time

        closed_set.add(current)

        for neighbor in ROAD_NETWORK.get(current, {}):
            edge = ROAD_NETWORK[current][neighbor]
            if edge['restricted'] and vehicle.type == 'Truck':
                logger.debug(f"Skipping restricted edge {current} -> {neighbor}")
                continue

            traffic_multiplier = traffic_multiplier_override or get_traffic_multiplier(edge['traffic_multiplier'], preferred_time)
            distance = edge['distance']
            speed = min(edge['speed_limit'], vehicle.max_speed) / max(traffic_multiplier, 1.0)
            segment_time = distance / speed if speed > 0 else 0

            fuel_cost = distance * vehicle.fuel_rate
            toll_cost = edge['toll']
            segment_cost = fuel_cost * 1.0 + toll_cost + segment_time * 10.0

            tentative_g = g_score[current] + segment_cost
            tentative_travel_time = travel_time + segment_time

            if neighbor in closed_set:
                continue

            if neighbor not in [n[1] for n in open_set] or tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor, path + [current], tentative_travel_time))

    logger.error(f"No route found from {start} to {goal}")
    return None, float('inf'), 0

def heuristic(node, goal):
    return abs(ord(node) - ord(goal)) * 5

def update_vehicle_position(order):
    if order.status != 'In Transit':
        return order.status == 'Delivered', None

    vehicle = db.session.get(Vehicle, order.vehicle_id)
    route = json.loads(order.route)
    current_time = datetime.now(ZoneInfo("UTC"))
    dispatch_time = order.dispatch_time.replace(tzinfo=ZoneInfo("UTC")) if order.dispatch_time.tzinfo is None else order.dispatch_time

    elapsed_time = (current_time - dispatch_time).total_seconds() / 3600

    if elapsed_time >= order.estimated_travel_time:
        order.status = 'Delivered'
        order.current_location = route[-1]
        order.last_updated = current_time
        vehicle.status = 'Available'
        db.session.commit()
        logger.info(f"Order {order.id} delivered")
        return True, None

    progress = min(elapsed_time / order.estimated_travel_time, 1.0)
    segment_index = min(int(len(route) * progress), len(route) - 2)
    order.current_location = route[segment_index]

    distance_traveled = calculate_route_distance(route[:segment_index + 1])
    order.fuel_consumed = distance_traveled * vehicle.fuel_rate

    delay_message = None
    if random.random() < 0.1:
        delay_type = random.choice(['traffic', 'breakdown'])
        if delay_type == 'traffic':
            new_route, new_cost, new_travel_time = a_star_route(
                order.current_location, route[-1], current_time, vehicle, traffic_multiplier_override=2.0
            )
            if new_route:
                order.route = json.dumps(new_route)
                order.estimated_travel_time = new_travel_time
                order.eta = current_time + timedelta(hours=new_travel_time)
                order.cost = new_cost
                delay_message = f"Traffic delay detected. Route updated. New ETA: {order.eta.isoformat()}"
                logger.info(f"Order {order.id} rerouted due to traffic")
        else:
            order.eta += timedelta(hours=1)
            order.estimated_travel_time += 1
            delay_message = f"Vehicle breakdown detected. ETA delayed by 1 hour to {order.eta.isoformat()}"
            logger.info(f"Order {order.id} delayed due to breakdown")

    order.last_updated = current_time
    db.session.commit()
    return False, delay_message

def calculate_metrics():
    logger.info("Calculating metrics for /metrics endpoint")
    partner_id = session.get('partner_id')
    if not partner_id:
        logger.warning("No partner_id in session")
        return None, "Session expired. Please log in again."

    orders = Order.query.filter_by(partner_id=partner_id).all()
    if not orders:
        logger.info("No orders found for partner_id: %s", partner_id)
        return None, "No orders found for this partner."

    data = []
    for order in orders:
        if order.route:
            try:
                route = json.loads(order.route)
                miles_driven = calculate_route_distance(route)
            except json.JSONDecodeError:
                miles_driven = 0
        else:
            miles_driven = 0

        delivery_time = order.estimated_travel_time or 0
        status = 'on_time' if order.status == 'Delivered' and order.eta and order.eta >= order.created_at else 'delayed'
        fuel_consumption = order.fuel_consumed if order.fuel_consumed > 0 else 0.1
        data.append({
            'delivery_time': delivery_time,
            'fuel_consumption': fuel_consumption,
            'miles_driven': miles_driven,
            'status': status,
            'vehicle_type': order.vehicle.type if order.vehicle else 'Unknown',
            'created_at': order.created_at,
            'pickup_location': order.pickup_location,
            'delivery_location': order.delivery_location,
            'cost': order.cost or 0
        })

    total_delivery_time = sum(d['delivery_time'] for d in data)
    avg_delivery_time = total_delivery_time / len(data) if data else 0
    delivery_time_std = (sum((d['delivery_time'] - avg_delivery_time) ** 2 for d in data) / len(data)) ** 0.5 if data else 0
    total_miles = sum(d['miles_driven'] for d in data)
    total_fuel = sum(d['fuel_consumption'] for d in data)
    avg_mpg = total_miles / total_fuel if total_fuel > 0 else 0
    on_time_count = sum(1 for d in data if d['status'] == 'on_time')
    on_time_rate = (on_time_count / len(data)) * 100 if data else 0
    customer_scores = [random.randint(1, 5) for _ in data]
    avg_customer_score = sum(customer_scores) / len(customer_scores) if customer_scores else 0

    report = {
        "avg_delivery_time": f"{avg_delivery_time:.2f} hours",
        "delivery_time_variability": f"±{delivery_time_std:.2f} hours",
        "avg_fuel_efficiency": f"{avg_mpg:.2f} MPG",
        "on_time_rate": f"{on_time_rate:.2f}%",
        "customer_satisfaction": f"{avg_customer_score:.1f}/5"
    }

    try:
        with open('delivery_performance_report.csv', 'w') as f:
            f.write('Metric,Value\n')
            for k, v in report.items():
                f.write(f"{k},{v}\n")
        logger.info("Report saved to delivery_performance_report.csv")
    except Exception as e:
        logger.error("Failed to save report CSV: %s", str(e))

    return report, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'GET':
        return render_template('admin_login.html')
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        logger.warning("Admin login attempt with missing credentials")
        return jsonify({'status': 'error', 'message': 'Missing username or password'}), 400
    admin = Admin.query.filter_by(username=data['username']).first()
    if admin and check_password_hash(admin.password_hash, data['password']):
        session.permanent = True
        session['admin_id'] = admin.id
        logger.info("Successful admin login for admin_id: %s", admin.id)
        return jsonify({'status': 'success', 'redirect': url_for('admin_dashboard')})
    logger.warning("Failed admin login attempt for username: %s", data['username'])
    return jsonify({'status': 'error', 'message': 'Invalid admin credentials'}), 401

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'admin_id' not in session:
        logger.warning("Unauthorized admin dashboard access, redirecting to admin login")
        return redirect(url_for('admin_login'))
    partners = Partner.query.all()
    logger.info("Serving admin dashboard for admin_id: %s", session.get('admin_id'))
    return render_template('admin_dashboard.html', partners=partners)

@app.route('/dashboard')
def dashboard():
    if 'partner_id' not in session:
        logger.warning("Unauthorized dashboard access, redirecting to login")
        return redirect(url_for('login_page'))
    logger.info("Serving dashboard for partner_id: %s", session.get('partner_id'))
    return render_template('dashboard.html')

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        logger.warning("Login attempt with missing credentials")
        return jsonify({'status': 'error', 'message': 'Missing username or password'}), 400
    partner = Partner.query.filter_by(username=data['username']).first()
    if partner and check_password_hash(partner.password_hash, data['password']):
        session.permanent = True
        session['partner_id'] = partner.id
        logger.info("Successful login for partner_id: %s", partner.id)
        return jsonify({'status': 'success', 'redirect': url_for('dashboard')})
    logger.warning("Failed login attempt for username: %s", data['username'])
    return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    partner_id = session.get('partner_id')
    admin_id = session.get('admin_id')
    session.clear()
    if partner_id:
        logger.info("Logged out partner_id: %s", partner_id)
    elif admin_id:
        logger.info("Logged out admin_id: %s", admin_id)
    return jsonify({'status': 'success', 'message': 'Logged out successfully'})

@app.route('/api/orders', methods=['POST'])
def place_order():
    partner_id = session.get('partner_id')
    if not partner_id:
        session.clear()
        logger.warning("Unauthorized order attempt, session expired")
        return jsonify({'status': 'error', 'message': 'Session expired. Please log in again.'}), 401
    data = request.get_json()
    logger.info(f"Received order request: {data}")
    required_fields = ['product', 'pickup_location', 'delivery_location', 'preferred_time', 'urgency', 'weight', 'volume']
    if not data or not all(field in data for field in required_fields):
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

    product = data['product']
    pickup_location = data['pickup_location']
    delivery_location = data['delivery_location']
    urgency = data['urgency']
    try:
        weight = float(data['weight'])
        volume = float(data['volume'])
        if weight <= 0 or volume <= 0:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'Weight and volume must be positive numbers'}), 400

    try:
        preferred_time = datetime.fromisoformat(data['preferred_time'].replace('Z', '+00:00'))
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid preferred_time format. Use ISO 8601'}), 400

    now = datetime.now(ZoneInfo("UTC"))
    if preferred_time < now + timedelta(hours=2):
        return jsonify({'status': 'error', 'message': 'Delivery time must be at least 2 hours from now'}), 400

    try:
        encrypted_pickup_location, pickup_location_key = encrypt_data(pickup_location)
        encrypted_delivery_location, delivery_location_key = encrypt_data(delivery_location)
    except Exception as e:
        logger.error(f"Encryption error: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Failed to encrypt location data'}), 500

    try:
        start_node = map_location_to_node(pickup_location)
        goal_node = map_location_to_node(delivery_location)
        logger.info(f"Mapped locations: {pickup_location} -> {start_node}, {delivery_location} -> {goal_node}")
    except ValueError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

    if start_node == goal_node:
        valid_locations = ', '.join(['pallikaranai', 'tambaram', 'delhi', 'warehouse', 'city_center', 'chennai', 'mumbai'])
        return jsonify({'status': 'error', 'message': f'Pickup and delivery locations cannot be the same. Valid locations: {valid_locations}'}), 400

    temp_vehicle = Vehicle.query.first()
    if not temp_vehicle:
        return jsonify({'status': 'error', 'message': 'No vehicles available'}), 400
    route, _, a_star_travel_time = a_star_route(start_node, goal_node, preferred_time, temp_vehicle)
    if not route:
        return jsonify({'status': 'error', 'message': 'No valid route available'}), 400

    vehicle, vehicle_score = allocate_vehicle(weight, volume, urgency, route)
    if not vehicle:
        suggested_time = (preferred_time + timedelta(hours=24)).isoformat()
        return jsonify({
            'status': 'error',
            
            'suggested': suggested_time
        }), 400

    vehicle.status = 'Allocated'
    route, total_cost, a_star_travel_time = a_star_route(start_node, goal_node, preferred_time, vehicle)
    if not route:
        db.session.rollback()
        vehicle.status = 'Available'
        db.session.commit()
        return jsonify({'status': 'error', 'message': 'No valid route for selected vehicle'}), 400

    travel_time = a_star_travel_time

    eta = preferred_time if urgency == 'High' else now + timedelta(hours=travel_time + 1)
    dispatch_time = datetime.now(ZoneInfo("UTC"))
    vehicle.status = 'Dispatched'
    order = Order(
        partner_id=partner_id,
        product=product,
        pickup_location=pickup_location,
        delivery_location=delivery_location,
        encrypted_pickup_location=encrypted_pickup_location,
        pickup_location_key=pickup_location_key,
        encrypted_delivery_location=encrypted_delivery_location,
        delivery_location_key=delivery_location_key,
        preferred_time=preferred_time,
        urgency=urgency,
        weight=weight,
        volume=volume,
        status='In Transit',
        route=json.dumps(route),
        cost=total_cost,
        vehicle_id=vehicle.id,
        eta=eta,
        estimated_travel_time=travel_time,
        dispatch_time=dispatch_time,
        current_location=route[0],
        fuel_consumed=0.0,
        last_updated=dispatch_time
    )
    try:
        db.session.add(order)
        db.session.commit()
        logger.info(f"Order {order.id} dispatched. ETA: {order.eta}, Vehicle: {vehicle.type}")
        return jsonify({
            'status': 'success',
            'order_id': order.id,
            'eta': order.eta.isoformat(),
            'route': route,
            'cost': round(total_cost, 2),
            'estimated_travel_time': round(travel_time, 2),
            'vehicle_type': vehicle.type
        })
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to place order: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Failed to place order'}), 500

@app.route('/api/orders/<int:order_id>/track', methods=['GET'])
def track_order(order_id):
    partner_id = session.get('partner_id')
    if not partner_id:
        session.clear()
        logger.warning("Unauthorized track attempt, session expired")
        return jsonify({'status': 'error', 'message': 'Session expired. Please log in again.'}), 401
    order = Order.query.get(order_id)
    if not order:
        logger.info("Order %s not found", order_id)
        return jsonify({'status': 'error', 'message': 'Order not found'}), 404
    if order.partner_id != partner_id:
        logger.warning("Unauthorized access to order %s by partner_id %s", order_id, partner_id)
        return jsonify({'status': 'error', 'message': 'Unauthorized access to order'}), 403

    is_delivered, delay_message = update_vehicle_position(order)
    vehicle = db.session.get(Vehicle, order.vehicle_id) if order.vehicle_id else None
    
    current_location = order.current_location or (json.loads(order.route)[-1] if order.route else 'N/A')
    coordinates = NODE_COORDINATES.get(current_location, {'lat': 0, 'lng': 0, 'name': 'Unknown'})
    
    logger.info("Tracking order %s: status=%s, location=%s", order_id, order.status, current_location)
    return jsonify({
        'status': order.status,
        'current_location': coordinates['name'],
        'coordinates': {'lat': coordinates['lat'], 'lng': coordinates['lng']},
        'fuel_consumed': round(order.fuel_consumed, 2) if order.fuel_consumed is not None else 0.00,
        'eta': order.eta.isoformat() if order.eta else None,
        'vehicle_type': vehicle.type if vehicle else 'N/A',
        'estimated_travel_time': round(order.estimated_travel_time, 2) if order.estimated_travel_time else 0,
        'delay_message': delay_message
    })

@app.route('/api/orders/<int:order_id>/complete', methods=['POST'])
def complete_order(order_id):
    partner_id = session.get('partner_id')
    if not partner_id:
        session.clear()
        logger.warning("Unauthorized complete attempt, session expired")
        return jsonify({'status': 'error', 'message': 'Session expired. Please log in again.'}), 401
    order = Order.query.get_or_404(order_id)
    if order.partner_id != partner_id:
        logger.warning("Unauthorized complete attempt for order %s by partner_id %s", order_id, partner_id)
        return jsonify({'status': 'error', 'message': 'Unauthorized access to order'}), 403
    if order.status == 'Delivered':
        logger.warning("Order %s already delivered", order_id)
        return jsonify({'status': 'error', 'message': 'Order already delivered'}), 400
    order.status = 'Delivered'
    vehicle = db.session.get(Vehicle, order.vehicle_id)
    vehicle.status = 'Available'
    order.current_location = json.loads(order.route)[-1]
    order.last_updated = datetime.now(ZoneInfo("UTC"))
    try:
        db.session.commit()
        logger.info(f"Order {order_id} completed")
        return jsonify({'status': 'success', 'message': 'Delivery completed'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to complete order {order_id}: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Failed to complete order'}), 500

@app.route('/api/orders/history', methods=['GET'])
def order_history():
    partner_id = session.get('partner_id')
    if not partner_id:
        session.clear()
        logger.warning("Unauthorized history attempt, session expired")
        return jsonify({'status': 'error', 'message': 'Session expired. Please log in again.'}), 401
    orders = Order.query.filter_by(partner_id=partner_id).all()
    logger.info("Fetched order history for partner_id: %s, %d orders", partner_id, len(orders))
    return jsonify([{
        'id': o.id,
        'product': o.product,
        'pickup_location': o.pickup_location,
        'delivery_location': o.delivery_location,
        'status': o.status,
        'eta': o.eta.isoformat() if o.eta else None,
        'cost': round(o.cost, 2) if o.cost else 0,
        'estimated_travel_time': round(o.estimated_travel_time, 2) if o.estimated_travel_time else 0,
        'vehicle_type': db.session.get(Vehicle, o.vehicle_id).type if o.vehicle_id else 'N/A'
    } for o in orders])

@app.route('/api/admin/partners', methods=['GET'])
def get_partners():
    if 'admin_id' not in session:
        logger.warning("Unauthorized partners access attempt")
        return jsonify({'status': 'error', 'message': 'Unauthorized access'}), 401
    partners = Partner.query.all()
    logger.info("Fetched %d partners for admin_id: %s", len(partners), session.get('admin_id'))
    return jsonify([{
        'id': p.id,
        'username': p.username
    } for p in partners])

@app.route('/api/admin/partners/<int:partner_id>/orders', methods=['GET'])
def get_partner_orders(partner_id):
    if 'admin_id' not in session:
        logger.warning("Unauthorized partner orders access attempt")
        return jsonify({'status': 'error', 'message': 'Unauthorized access'}), 401
    partner = Partner.query.get(partner_id)
    if not partner:
        logger.info("Partner %s not found", partner_id)
        return jsonify({'status': 'error', 'message': 'Partner not found'}), 404
    orders = Order.query.filter_by(partner_id=partner_id).all()
    logger.info("Fetched %d orders for partner_id: %s", len(orders), partner_id)
    return jsonify([{
        'id': o.id,
        'product': o.product,
        'pickup_location': o.pickup_location,
        'delivery_location': o.delivery_location,
        'status': o.status,
        'eta': o.eta.isoformat() if o.eta else None,
        'cost': round(o.cost, 2) if o.cost else 0,
        'estimated_travel_time': round(o.estimated_travel_time, 2) if o.estimated_travel_time else 0,
        'vehicle_type': db.session.get(Vehicle, o.vehicle_id).type if o.vehicle_id else 'N/A'
    } for o in orders])

@app.route('/api/admin/orders/<int:order_id>', methods=['PUT', 'DELETE'])
def manage_order(order_id):
    if 'admin_id' not in session:
        logger.warning("Unauthorized order management attempt")
        return jsonify({'status': 'error', 'message': 'Unauthorized access'}), 401
    order = Order.query.get(order_id)
    if not order:
        logger.info("Order %s not found", order_id)
        return jsonify({'status': 'error', 'message': 'Order not found'}), 404

    if request.method == 'PUT':
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400

        if 'product' in data:
            order.product = data['product']
        if 'status' in data:
            if data['status'] in ['Pending', 'In Transit', 'Delivered']:
                order.status = data['status']
                if data['status'] == 'Delivered' and order.vehicle_id:
                    vehicle = db.session.get(Vehicle, order.vehicle_id)
                    vehicle.status = 'Available'
        if 'pickup_location' in data:
            try:
                encrypted_pickup_location, pickup_location_key = encrypt_data(data['pickup_location'])
                order.pickup_location = data['pickup_location']
                order.encrypted_pickup_location = encrypted_pickup_location
                order.pickup_location_key = pickup_location_key
            except Exception as e:
                logger.error(f"Encryption error: {str(e)}")
                return jsonify({'status': 'error', 'message': 'Failed to encrypt pickup location'}), 500
        if 'delivery_location' in data:
            try:
                encrypted_delivery_location, delivery_location_key = encrypt_data(data['delivery_location'])
                order.delivery_location = data['delivery_location']
                order.encrypted_delivery_location = encrypted_delivery_location
                order.delivery_location_key = delivery_location_key
            except Exception as e:
                logger.error(f"Encryption error: {str(e)}")
                return jsonify({'status': 'error', 'message': 'Failed to encrypt delivery location'}), 500
        if 'urgency' in data and data['urgency'] in ['High', 'Medium', 'Low']:
            order.urgency = data['urgency']
        if 'weight' in data:
            try:
                weight = float(data['weight'])
                if weight > 0:
                    order.weight = weight
                else:
                    raise ValueError
            except (ValueError, TypeError):
                return jsonify({'status': 'error', 'message': 'Weight must be a positive number'}), 400
        if 'volume' in data:
            try:
                volume = float(data['volume'])
                if volume > 0:
                    order.volume = volume
                else:
                    raise ValueError
            except (ValueError, TypeError):
                return jsonify({'status': 'error', 'message': 'Volume must be a positive number'}), 400
        if 'preferred_time' in data:
            try:
                preferred_time = datetime.fromisoformat(data['preferred_time'].replace('Z', '+00:00'))
                order.preferred_time = preferred_time
            except ValueError:
                return jsonify({'status': 'error', 'message': 'Invalid preferred_time format. Use ISO 8601'}), 400

        order.last_updated = datetime.now(ZoneInfo("UTC"))
        try:
            db.session.commit()
            logger.info(f"Order {order_id} updated by admin")
            return jsonify({'status': 'success', 'message': 'Order updated successfully'})
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to update order {order_id}: {str(e)}")
            return jsonify({'status': 'error', 'message': 'Failed to update order'}), 500

    elif request.method == 'DELETE':
        try:
            if order.vehicle_id:
                vehicle = db.session.get(Vehicle, order.vehicle_id)
                vehicle.status = 'Available'
            db.session.delete(order)
            db.session.commit()
            logger.info(f"Order {order_id} deleted by admin")
            return jsonify({'status': 'success', 'message': 'Order deleted successfully'})
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to delete order {order_id}: {str(e)}")
            return jsonify({'status': 'error', 'message': 'Failed to delete order'}), 500

@app.route('/api/routes', methods=['GET'])
def routes():
    partner_id = session.get('partner_id')
    if not partner_id:
        session.clear()
        logger.warning("Unauthorized routes attempt, session expired")
        return jsonify({'status': 'error', 'message': 'Session expired. Please log in again.'}), 401

    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    vehicle_type = request.args.get('vehicle_type', 'all')

    query = Order.query.filter_by(partner_id=partner_id)
    if start_date and end_date:
        try:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            query = query.filter(Order.created_at.between(start, end))
        except ValueError:
            return jsonify({'status': 'error', 'message': 'Invalid date format'}), 400
    if vehicle_type != 'all':
        query = query.join(Vehicle).filter(Vehicle.type == vehicle_type)

    orders = query.all()
    route_map = {}
    for order in orders:
        if not order.route:
            continue
        try:
            route_nodes = json.loads(order.route)
            if len(route_nodes) < 2:
                continue
            key = f"{route_nodes[0]}-{route_nodes[-1]}-{order.vehicle.type if order.vehicle else 'Unknown'}"
            if key not in route_map:
                start_coords = NODE_COORDINATES.get(route_nodes[0], {'name': 'Unknown', 'lat': 0})
                end_coords = NODE_COORDINATES.get(route_nodes[-1], {'name': 'Unknown', 'lat': 0})
                distance = calculate_route_distance(route_nodes)
                route_map[key] = {
                    'id': len(route_map) + 1,
                    'start_location': start_coords.get('name', 'Unknown'),
                    'end_location': end_coords.get('name', 'Unknown'),
                    'distance': round(distance, 2),
                    'estimated_time': order.estimated_travel_time or 0,
                    'vehicle_type': order.vehicle.type if order.vehicle else 'Unknown',
                    'order_count': 0
                }
            route_map[key]['order_count'] += 1
        except json.JSONDecodeError:
            logger.warning(f"Invalid route JSON for order {order.id}")
            continue

    routes = list(route_map.values())
    logger.info(f"Fetched {len(routes)} routes for partner_id: {partner_id}")
    return jsonify({'status': 'success', 'routes': routes})

@app.route('/api/inventory', methods=['GET'])
def inventory():
    partner_id = session.get('partner_id')
    if not partner_id:
        session.clear()
        logger.warning("Unauthorized inventory attempt, session expired")
        return jsonify({'status': 'error', 'message': 'Session expired. Please log in again.'}), 401

    items = Inventory.query.filter_by(partner_id=partner_id).all()
    inventory_list = [{
        'id': item.id,
        'product': item.product,
        'quantity': item.quantity,
        'location': item.location,
        'last_updated': item.last_updated.isoformat() if item.last_updated else None
    } for item in items]
    logger.info(f"Fetched {len(inventory_list)} inventory items for partner_id: {partner_id}")
    return jsonify({'status': 'success', 'inventory': inventory_list})

@app.route('/api/inventory/add', methods=['POST'])
def add_inventory():
    partner_id = session.get('partner_id')
    if not partner_id:
        session.clear()
        logger.warning("Unauthorized inventory add attempt, session expired")
        return jsonify({'status': 'error', 'message': 'Session expired. Please log in again.'}), 401

    data = request.get_json()
    required_fields = ['product', 'quantity', 'location']
    if not data or not all(field in data for field in required_fields):
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

    try:
        quantity = int(data['quantity'])
        if quantity <= 0:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'Quantity must be a positive integer'}), 400

    valid_locations = ['pallikaranai', 'tambaram', 'delhi', 'warehouse', 'city_center', 'chennai', 'mumbai']
    if data['location'].lower() not in valid_locations:
        return jsonify({'status': 'error', 'message': f"Invalid location. Valid locations: {', '.join(valid_locations)}"}), 400

    item = Inventory.query.filter_by(partner_id=partner_id, product=data['product'], location=data['location']).first()
    try:
        if item:
            item.quantity += quantity
            item.last_updated = datetime.now(ZoneInfo("UTC"))
        else:
            item = Inventory(
                partner_id=partner_id,
                product=data['product'],
                quantity=quantity,
                location=data['location'],
                last_updated=datetime.now(ZoneInfo("UTC"))
            )
            db.session.add(item)
        db.session.commit()
        logger.info(f"Added/Updated inventory item: {data['product']} at {data['location']} for partner_id: {partner_id}")
        return jsonify({'status': 'success', 'message': f"Added/Updated {data['product']} in inventory"})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to add/update inventory: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Failed to add/update inventory'}), 500

@app.route('/api/analytics', methods=['GET'])
def analytics():
    partner_id = session.get('partner_id')
    if not partner_id:
        session.clear()
        logger.warning("Unauthorized analytics attempt, session expired")
        return jsonify({'status': 'error', 'message': 'Session expired. Please log in again.'}), 401

    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    vehicle_type = request.args.get('vehicle_type', 'all')

    query = Order.query.filter_by(partner_id=partner_id)
    if start_date and end_date:
        try:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            query = query.filter(Order.created_at.between(start, end))
        except ValueError:
            return jsonify({'status': 'error', 'message': 'Invalid date format'}), 400
    if vehicle_type != 'all':
        query = query.join(Vehicle).filter(Vehicle.type == vehicle_type)

    orders = query.all()
    total_orders = len(orders)
    if total_orders == 0:
        return jsonify({
            'status': 'success',
            'total_orders': 0,
            'on_time_rate': 0,
            'fuel_cost': 0,
            'avg_cost': 0,
            'avg_travel_time': 0,
            'status_distribution': {'Pending': 0, 'In Transit': 0, 'Delivered': 0},
            'daily_orders': [],
            'fuel_by_vehicle': [],
            'top_pickup_locations': [],
            'top_delivery_locations': [],
            'cost_per_km': 0,
            'fuel_efficiency_by_vehicle': {}
        })

    on_time = sum(1 for o in orders if o.status == 'Delivered' and o.eta and o.eta >= o.created_at)
    fuel_cost = sum(o.fuel_consumed for o in orders if o.fuel_consumed)
    total_cost = sum(o.cost for o in orders if o.cost)
    total_travel_time = sum(o.estimated_travel_time for o in orders if o.estimated_travel_time)
    total_distance = sum(calculate_route_distance(json.loads(o.route)) if o.route else 0 for o in orders)

    status_counts = {'Pending': 0, 'In Transit': 0, 'Delivered': 0}
    daily_orders = {}
    fuel_by_vehicle = {}
    distance_by_vehicle = {}
    pickup_locations = {}
    delivery_locations = {}

    for order in orders:
        status_counts[order.status] = status_counts.get(order.status, 0) + 1
        date_str = order.created_at.strftime('%Y-%m-%d')
        daily_orders[date_str] = daily_orders.get(date_str, 0) + 1
        vehicle_type = order.vehicle.type if order.vehicle else 'Unknown'
        fuel = float(order.fuel_consumed or 0.0)
        if fuel < 0:
            logger.warning(f"Negative fuel_consumed ({fuel}) for order {order.id}, setting to 0")
            fuel = 0.0
        fuel_by_vehicle[vehicle_type] = fuel_by_vehicle.get(vehicle_type, 0) + fuel
        distance = calculate_route_distance(json.loads(order.route)) if order.route else 0
        distance_by_vehicle[vehicle_type] = distance_by_vehicle.get(vehicle_type, 0) + distance
        pickup_locations[order.pickup_location] = pickup_locations.get(order.pickup_location, 0) + 1
        delivery_locations[order.delivery_location] = delivery_locations.get(order.delivery_location, 0) + 1

    daily_orders_list = [{'date': k, 'count': v} for k, v in sorted(daily_orders.items())]
    fuel_by_vehicle_list = [{'vehicle_type': k, 'fuel': round(v, 2)} for k, v in fuel_by_vehicle.items() if v > 0]
    status_distribution = {k: (v / total_orders * 100) for k, v in status_counts.items()}
    top_pickup_locations = sorted(pickup_locations.items(), key=lambda x: x[1], reverse=True)[:3]
    top_delivery_locations = sorted(delivery_locations.items(), key=lambda x: x[1], reverse=True)[:3]
    cost_per_km = (total_cost / total_distance) if total_distance > 0 else 0
    fuel_efficiency_by_vehicle = {k: round(distance_by_vehicle.get(k, 0) / fuel_by_vehicle.get(k, 1), 2) for k in fuel_by_vehicle}

    logger.info("Analytics for partner_id: %s, total_orders=%d", partner_id, total_orders)
    return jsonify({
        'status': 'success',
        'total_orders': total_orders,
        'on_time_rate': (on_time / total_orders * 100) if total_orders > 0 else 0,
        'fuel_cost': round(fuel_cost, 2),
        'avg_cost': round(total_cost / total_orders, 2) if total_orders > 0 else 0,
        'avg_travel_time': round(total_travel_time / total_orders, 2) if total_orders > 0 else 0,
        'status_distribution': status_distribution,
        'daily_orders': daily_orders_list,
        'fuel_by_vehicle': fuel_by_vehicle_list,
        'top_pickup_locations': [{'location': k, 'count': v} for k, v in top_pickup_locations],
        'top_delivery_locations': [{'location': k, 'count': v} for k, v in top_delivery_locations],
        'cost_per_km': round(cost_per_km, 2),
        'fuel_efficiency_by_vehicle': fuel_efficiency_by_vehicle
    })

@app.route('/api/analytics/charts', methods=['GET'])
def analytics_charts():
    partner_id = session.get('partner_id')
    if not partner_id:
        session.clear()
        logger.warning("Unauthorized analytics charts attempt, session expired")
        return jsonify({'status': 'error', 'message': 'Session expired. Please log in again.'}), 401

    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    vehicle_type = request.args.get('vehicle_type', 'all')

    query = Order.query.filter_by(partner_id=partner_id)
    if start_date and end_date:
        try:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            query = query.filter(Order.created_at.between(start, end))
        except ValueError:
            return jsonify({'status': 'error', 'message': 'Invalid date format'}), 400
    if vehicle_type != 'all':
        query = query.join(Vehicle).filter(Vehicle.type == vehicle_type)

    orders = query.all()
    if not orders:
        return jsonify({
            'status': 'success',
            'status_distribution': {'Pending': 0, 'In Transit': 0, 'Delivered': 0},
            'daily_orders': [],
            'fuel_by_vehicle': [],
            'on_time_rate': {'on_time': 0, 'delayed': 0},
            'chart': '/charts/on_time_rate.png'
        })

    status_counts = {'Pending': 0, 'In Transit': 0, 'Delivered': 0}
    daily_orders = {}
    fuel_by_vehicle = {}
    on_time = 0
    delayed = 0

    for order in orders:
        status_counts[order.status] = status_counts.get(order.status, 0) + 1
        date_str = order.created_at.strftime('%Y-%m-%d')
        daily_orders[date_str] = daily_orders.get(date_str, 0) + 1
        if order.vehicle:
            vehicle_type = order.vehicle.type
            fuel = float(order.fuel_consumed or 0.0)
            if fuel < 0:
                logger.warning(f"Negative fuel_consumed ({fuel}) for order {order.id}, setting to 0")
                fuel = 0.0
            fuel_by_vehicle[vehicle_type] = fuel_by_vehicle.get(vehicle_type, 0.0) + fuel
        if order.status == 'Delivered' and order.eta and order.eta >= order.created_at:
            on_time += 1
        else:
            delayed += 1

    daily_orders_list = [{'date': k, 'count': v} for k, v in sorted(daily_orders.items())]
    fuel_by_vehicle_list = [{'vehicle_type': k, 'fuel': round(v, 2)} for k, v in fuel_by_vehicle.items() if v > 0]

    try:
        labels = ['On Time', 'Delayed']
        sizes = [on_time, delayed]
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('On-Time Delivery Rate')
        chart_path = 'static/charts/on_time_rate.png'
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        plt.savefig(chart_path)
        plt.close()
        logger.info(f"Generated chart at {chart_path}")
    except Exception as e:
        logger.error(f"Failed to save on_time_rate.png in analytics_charts: {str(e)}")

    logger.info("Fetched chart data for partner_id: %s", partner_id)
    return jsonify({
        'status': 'success',
        'status_distribution': status_counts,
        'daily_orders': daily_orders_list,
        'fuel_by_vehicle': fuel_by_vehicle_list,
        'on_time_rate': {'on_time': on_time, 'delayed': delayed},
        'chart': '/charts/on_time_rate.png'
    })

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    partner_id = session.get('partner_id')
    if not partner_id:
        session.clear()
        logger.warning("Unauthorized chatbot attempt, session expired")
        return jsonify({'status': 'error', 'message': 'Session expired. Please log in again.'}), 401
    
    data = request.get_json()
    option = data.get('option') if data else None
    
    if not option:
        return jsonify({
            'status': 'success',
            'message': 'Hello! How can I assist you with your logistics data?',
            'options': [
                {'id'== {'id': 'total_orders', 'text': 'View total orders'},
                {'id': 'on_time_rate', 'text': 'Check on-time delivery rate'},
                {'id': 'recent_orders', 'text': 'View recent order statuses'},
                {'id': 'avg_delivery_time', 'text': 'Check average delivery time'},
                {'id': 'fuel_efficiency', 'text': 'View fuel efficiency by vehicle type'}
                }
        ]})
    
    orders = Order.query.filter_by(partner_id=partner_id).all()
    response = ''
    if option == 'total_orders':
        total_orders = len(orders)
        response = f"You have a total of {total_orders} orders."
    elif option == 'on_time_rate':
        on_time = sum(1 for o in orders if o.status == 'Delivered' and o.eta and o.eta >= o.created_at)
        on_time_rate = (on_time / len(orders) * 100) if orders else 0
        response = f"Your on-time delivery rate is {round(on_time_rate, 2)}%."
    elif option == 'recent_orders':
        recent_orders = Order.query.filter_by(partner_id=partner_id).order_by(Order.created_at.desc()).limit(5).all()
        if recent_orders:
            response = "Here are your 5 most recent orders:\n" + "\n".join(
                [f"Order #{o.id} - Product: {o.product}, Status: {o.status}, Date: {o.created_at.strftime('%Y-%m-%d')}" for o in recent_orders]
            )
        else:
            response = "No recent orders found."
    elif option == 'avg_delivery_time':
        total_travel_time = sum(o.estimated_travel_time for o in orders if o.estimated_travel_time)
        avg_delivery_time = (total_travel_time / len(orders)) if orders else 0
        response = f"Your average delivery time is {round(avg_delivery_time, 2)} hours."
    elif option == 'fuel_efficiency':
        fuel_by_vehicle = {}
        distance_by_vehicle = {}
        for order in orders:
            vehicle_type = order.vehicle.type if order.vehicle else 'Unknown'
            fuel = order.fuel_consumed or 0.1
            distance = calculate_route_distance(json.loads(order.route)) if order.route else 0
            fuel_by_vehicle[vehicle_type] = fuel_by_vehicle.get(vehicle_type, 0) + fuel
            distance_by_vehicle[vehicle_type] = distance_by_vehicle.get(vehicle_type, 0) + distance
        fuel_efficiency = {k: round(distance_by_vehicle.get(k, 0) / fuel_by_vehicle.get(k, 1), 2) for k in fuel_by_vehicle}
        if fuel_efficiency:
            response = "Fuel efficiency by vehicle type (miles per gallon):\n" + "\n".join(
                [f"{k}: {v} MPG" for k, v in fuel_efficiency.items()]
            )
        else:
            response = "No fuel efficiency data available."
    else:
        response = "Invalid option selected. Please choose again."
    
    return jsonify({
        'status': 'success',
        'message': response,
        'options': [
            {'id': 'total_orders', 'text': 'View total orders'},
            {'id': 'on_time_rate', 'text': 'Check on-time delivery rate'},
            {'id': 'recent_orders', 'text': 'View recent order statuses'},
            {'id': 'avg_delivery_time', 'text': 'Check average delivery time'},
            {'id': 'fuel_efficiency', 'text': 'View fuel efficiency by vehicle type'}
        ]
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    report, error = calculate_metrics()
    if error:
        logger.error(f"Metrics error: {error}")
        return jsonify({'status': 'error', 'message': error}), 400
    logger.info("Serving metrics for partner_id: %s", session.get('partner_id'))
    return jsonify({'status': 'success', 'report': report})

@app.route('/download/report')
def download_report():
    try:
        logger.info("Serving delivery_performance_report.csv")
        return send_file('delivery_performance_report.csv', as_attachment=True)
    except FileNotFoundError:
        logger.error("delivery_performance_report.csv not found")
        return jsonify({'status': 'error', 'message': 'Report not found'}), 404

@app.route('/charts/<filename>')
def serve_chart(filename):
    chart_path = os.path.join('static/charts', filename)
    if os.path.exists(chart_path):
        logger.info(f"Serving chart: {chart_path}")
        return send_file(chart_path, mimetype='image/png')
    logger.warning(f"Chart file {chart_path} not found")
    return jsonify({'status': 'error', 'message': 'Chart not found'}), 404

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not Partner.query.first():
            db.session.add(Vehicle(type='Van', capacity_weight=1000, capacity_volume=10, fuel_rate=0.1, max_speed=80))
            db.session.add(Vehicle(type='Truck', capacity_weight=5000, capacity_volume=30, fuel_rate=0.15, max_speed=60))
            db.session.add(Vehicle(type='Bike', capacity_weight=50, capacity_volume=2, fuel_rate=0.05, max_speed=100))
            db.session.add(Partner(
                username='partner1',
                password_hash=generate_password_hash('pass123')
            ))
            db.session.add(Admin(
                username='root',
                password_hash=generate_password_hash('toor')
            ))
            db.session.commit()
            logger.info("Initialized database with default vehicles, partner, and admin")
        else:
            logger.info("Partner table already populated")
    app.run(debug=True, use_reloader=True)