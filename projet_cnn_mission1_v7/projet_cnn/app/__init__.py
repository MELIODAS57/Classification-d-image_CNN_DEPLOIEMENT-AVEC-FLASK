"""
app/__init__.py
---------------
Factory Flask — crée et configure l'application.
"""

from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 Mo max par upload

    from .routes import bp
    app.register_blueprint(bp)

    return app
