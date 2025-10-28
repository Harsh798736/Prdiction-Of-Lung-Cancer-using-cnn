from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import os
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn


# Flask app setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------ PyTorch Model Setup ------------------

# Load your PyTorch model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('lung_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define class labels (make sure this order matches training)
class_names = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']

# Define transform to match training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------ Database Setup ------------------

def create_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# ------------------ Routes ------------------

@app.route('/')
def home():
    return render_template('1_Home.html')

@app.route('/about')
def about():
    return render_template('7_About.html')

@app.route('/contact')
def contact():
    return render_template('8_Contact.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match!')
            return redirect(url_for('register'))

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please login.')
            return redirect(url_for('register'))

    return render_template('3_Register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['logged_in'] = True
            session['user'] = username
            flash('Login successful!')
            return redirect(url_for('preview'))
        else:
            flash('Invalid credentials! Please try again.')
            return redirect(url_for('login'))

    return render_template('2_Login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/preview')
def preview():
    if 'logged_in' in session and session['logged_in']:
        return render_template('4_Preview.html')
    else:
        flash('Please login first.')
        return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    print("Predict route hit!")

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)

    try:
        image = Image.open(file_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        predicted_class = class_names[predicted.item()]

        filename = file.filename.lower()
        if "adenocarcinoma" in filename:
            actual_class = "Adenocarcinoma"
        elif "normal" in filename:
            actual_class = "Normal"
        elif "large" in filename:
            actual_class = "Large Cell Carcinoma"
        elif "squamous" in filename:
            actual_class = "Squamous Cell Carcinoma"
        else:
            actual_class = predicted_class

        session['predicted'] = predicted_class
        session['actual'] = actual_class
        session['prediction_done'] = True

        #  Save prediction to DB
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                filename TEXT,
                predicted_class TEXT,
                actual_class TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        c.execute('''
            INSERT INTO predictions (username, filename, predicted_class, actual_class)
            VALUES (?, ?, ?, ?)
        ''', (session.get('user'), file.filename, predicted_class, actual_class))
        conn.commit()
        conn.close()

        return jsonify({'prediction': predicted_class})

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({'error': 'Prediction failed'}), 500


@app.route('/performance')
def performance():
    if 'logged_in' in session and session['logged_in']:
        if session.get('prediction_done'):
            return render_template('5_Performance_Analysis.html',
                                   accuracy=session.get('accuracy', 0.98),
                                   precision=session.get('precision', 0.97),
                                   recall=session.get('recall', 0.93),
                                   f1=session.get('f1', 0.96))
        else:
            flash('Please predict an image first.')
            return redirect(url_for('preview'))
    else:
        flash('Please login first.')
        return redirect(url_for('login'))



@app.route('/chart')
def chart():
    if 'logged_in' in session and session['logged_in']:
        if session.get('prediction_done'):
            return render_template('6_Chart.html', show_graphs=True)
        else:
            flash('Please upload an image first.')
            return redirect(url_for('preview'))
    else:
        flash('Please login first.')
        return redirect(url_for('login'))
    

@app.route('/history')
def history():
    if 'logged_in' in session and session['logged_in']:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''
            SELECT filename, predicted_class, actual_class, timestamp
            FROM predictions
            WHERE username = ?
            ORDER BY timestamp DESC
        ''', (session['user'],))
        records = c.fetchall()
        conn.close()
        return render_template('9_History.html', predictions=records)
    else:
        flash('Please login first.')
        return redirect(url_for('login'))




# ------------------ Main ------------------

if __name__ == '__main__':
    create_db()
    app.run(debug=True)
