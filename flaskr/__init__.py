import os

from flask import Flask, request, render_template, redirect, url_for

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
        db.execute('UPDATE store SET image=? WHERE id=1', image)
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

    @app.route('/test', methods=['GET','POST'])
    def test_image():
        if request.method == "POST":
            button_state = request.form['button']
            if button_state  == 'Forward':
                db = get_db()
                image = db.execute('SELECT image FROM store where id=1').fetchone()[0]

                image = net(image)

                db.execute('UPDATE store SET image=? WHERE id=2', [image])
                db.commit()

            return redirect(url_for('static_image'))

        return render_template('img/img.html')

    return app
