from flask import Flask, redirect, url_for, render_template, request, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = 'static/image_uploads'

app = Flask(__name__)
app.secret_key = "RahasiaTau"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    filename = session.get('upload_img', None)
    return render_template("home.html", filename=filename)

@app.route('/', methods=['POST'])
def submit_file():
    if 'file' not in request.files:
        flash('No File Part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No File Selected For Uploading')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        session['upload_img'] = filename 
        flash('File Successfully Uploaded')
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)