from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
import json
from random import randint
from flaskr.db import get_db
import time
from test_fruit_model import fruit_test
import io
from PIL import Image, ImageDraw, ImageFont

bp = Blueprint('image_capture',__name__,url_prefix='/image_capture')

@bp.route('/static_image')
def captured_image():
    return "<img src='/image_capture/static.jpeg'>"

@bp.route('/live_feed', methods=['GET','POST'])
def test_image():
    with open('/Users/georgesnomicos/fridge_net/class_idx.txt') as f:
        items_dict = json.load(f)
    items = list(items_dict.keys())

    if request.method == "POST":
        print(request.form)
        if request.form.get('image') == 'Img':
            db = get_db()
            camera_status = db.execute('UPDATE camera1 SET image_capture=1')
            db.commit()
            return render_template('image_capture/img.html', items=items)
        button_state = request.form.get('button')
        if button_state  == 'Forward':

            # Text for img file label (collection of data)
            item_name = request.form['text']
            img_idx = str(items_dict[item_name])
            while len(img_idx) < 3:
                img_idx = '0' + img_idx
            img_name = img_idx + '_' + item_name + '_' + str(randint(0,1000))

            # Changing camera state to image capturing in database


            # Wait for camera status to change
            
            # Then get image from database 
            db = get_db()
            image = db.execute('SELECT image FROM store where id=1').fetchone()[0]
            # prob, item_name = fruit_test(image)
            prob, item_name = 0.9, "Other"
            stream = io.BytesIO(image)
            img = Image.open(stream)

            # Collecting images:
            img.save('/Users/georgesnomicos/fridge_net/fruits-360_dataset/pi_data/' + img_name + '.jpeg')

            d = ImageDraw.Draw(img)
            font = ImageFont.truetype('/Users/georgesnomicos/Library/Fonts/arial_narrow_7.ttf',50)

            d.text((250,500), item_name[0], fill=(255,0,0),font=font)
            # d.text((250,400), "%s "%(round(prob.item(),2)), fill=(255,0,0),font=font)

            img.save('img.jpeg')
            with open('img.jpeg','rb') as f:
                image = f.read()

            db.execute('UPDATE store SET image=? WHERE id=2', [image])
            db.commit()

            # Change capturing mode
            db = get_db()
            camera_status = db.execute('UPDATE camera1 SET image_capture=1')
            db.commit()

            return redirect(url_for('image_capture.captured_image'))
    
    return render_template('image_capture/img.html', items=items)

@bp.route('/test.jpeg')
def get_image():
    db = get_db()
    image = db.execute('SELECT image FROM store where id=1').fetchone()[0]
    return image, 200

@bp.route('/static.jpeg')
def get_static_image():
    db = get_db()
    image = db.execute('SELECT image FROM store where id=2').fetchone()[0]
    return image, 200

