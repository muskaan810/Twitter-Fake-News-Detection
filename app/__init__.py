from flask import Flask

def create_app():
    app = Flask(__name__)
    app.secret_key = 'your_secret_key'  # Set your secret key for session management

    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app