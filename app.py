import imghdr
import os

import numpy as np
from PIL import Image
from flask import Flask, request
from keras.models import load_model
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './upload'
app = Flask(__name__)
app.config['UPLOAD_PATH'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']


model = load_model('models/my_best_model.epoch36-loss0.87.hdf5')
model.summary()


def normalize_pic(image_path):
    image = Image.open(image_path)
    image = image.resize((64, 64), Image.ANTIALIAS)
    image = np.array(image)
    image = np.expand_dims(image, 0)
    return image


def model_predict(img_path, model):
    image = normalize_pic(img_path)
    if image is None:
        return [0, 0, 0, 0]
    predict = model.predict(image)
    return predict


def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')


@app.route('/', methods=['POST'])
def upload_files():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            print(app.config)
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))

            image = normalize_pic(os.path.join(app.config['UPLOAD_PATH'], filename))
            g, m, pp, pl = model.predict(image)[0]

            return f'[{g}, {m}, {pp}, {pl}]'


if __name__ == '__main__':
    app.run(debug=True)
