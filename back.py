from logistics_app import app, db, Partner
from werkzeug.security import generate_password_hash

with app.app_context():
    partners = Partner.query.all()
    for partner in partners:
        partner.password_hash = generate_password_hash("pass123")
        print(f"Reset password for {partner.username}")
    try:
        db.session.commit()
        print("All passwords reset to 'pass123'")
    except Exception as e:
        db.session.rollback()
        print(f"Error resetting passwords: {str(e)}")