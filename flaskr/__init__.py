import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
import io
from PIL import Image, ImageDraw, ImageFont
import time
import json
from flaskr.db import get_db

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY = 'dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config == None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    from . import db
    db.init_app(app)

    @app.route('/', methods=['POST'])
    def update_image():
        db = get_db()
        image = [request.data]
        db.execute('UPDATE store SET image=? WHERE id=1', image).fetchone()
        db.commit()
        return '', 200

    @app.route('/camera_status', methods=['GET'])
    def camera_status():
        db = get_db()
        camera_status = db.execute('SELECT image_capture FROM camera1 where id=1').fetchone()[0]
        print(camera_status)
        return str(camera_status), 200

    from . import image_capture
    app.register_blueprint(image_capture.bp)

    return app

def save_img(file_name):
    pass
