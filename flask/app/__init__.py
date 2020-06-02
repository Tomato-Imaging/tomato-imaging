from flask import Flask

from os.path import join, dirname, realpath

UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static/uploads/')
PROCESSED_FOLDER = join(dirname(realpath(__file__)), 'static/processed/')

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

from app import main