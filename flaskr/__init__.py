import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
from test_fruit_model import fruit_test
import io
from PIL import Image, ImageDraw, ImageFont
import time
import json
from random import randint

def net(img):
    return img

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

    from flaskr.db import get_db
    @app.route('/', methods=['POST'])
    def updateImage():
        db = get_db()
        image = [request.data]
        db.execute('UPDATE store SET image=? WHERE id=1', image).fetchone()
        db.commit()
        return '', 200

    @app.route('/test.jpeg')
    def getImage():
        db = get_db()
        image = db.execute('SELECT image FROM store where id=1').fetchone()[0]
        return image, 200

    @app.route('/static.jpeg')
    def getStaticImage():
        db = get_db()
        image = db.execute('SELECT image FROM store where id=2').fetchone()[0]
        return image, 200

    @app.route('/static_image')
    def static_image():
        return "<img src='/static.jpeg'>"

    @app.route('/camera_status', methods=['GET'])
    def camera_status():
        db = get_db()
        camera_status = db.execute('SELECT image_capture FROM camera1 where id=1').fetchone()[0]
        print(camera_status)
        return str(camera_status), 200

    @app.route('/test', methods=['GET','POST'])
    def test_image():
        with open('/Users/georgesnomicos/fridge_net/class_idx.txt') as f:
            items_dict = json.load(f)
        items = list(items_dict.keys())

        if request.method == "POST":
            button_state = request.form['button']
            if button_state  == 'Forward':

                # Text for img file label (collection of data)
                item_name = request.form['text']
                img_idx = str(items_dict[item_name])
                while len(img_idx) < 3:
                    img_idx = '0' + img_idx
                img_name = img_idx + '_' + item_name + '_' + str(randint(0,1000))

                # Changing camera state to image capturing in database
                db = get_db()
                camera_status = db.execute('UPDATE camera1 SET image_capture=1')
                db.commit()

                # Wait for camera status to change
                time.sleep(3)
                
                # Then get image from database 
                db = get_db()
                image = db.execute('SELECT image FROM store where id=1').fetchone()[0]
                prob, item_name = fruit_test(image)
                stream = io.BytesIO(image)
                img = Image.open(stream)

                # Collecting images:
                img.save('/Users/georgesnomicos/fridge_net/fruits-360_dataset/pi_data/' + img_name + '.jpeg')

                d = ImageDraw.Draw(img)
                font = ImageFont.truetype('/Users/georgesnomicos/Library/Fonts/arial_narrow_7.ttf',50)

                d.text((250,500), item_name[0], fill=(255,0,0),font=font)
                d.text((250,400), "%s "%(round(prob.item(),2)), fill=(255,0,0),font=font)

                img.save('img.jpeg')
                with open('img.jpeg','rb') as f:
                    image = f.read()

                db.execute('UPDATE store SET image=? WHERE id=2', [image])
                db.commit()

                # Change capturing mode
                db = get_db()
                camera_status = db.execute('UPDATE camera1 SET image_capture=0')
                db.commit()

            return redirect(url_for('static_image'))
        return render_template('img/img.html', items=items)

    return app

def save_img(file_name):
    pass
