# testmodel.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'h5', 'model'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def evaluate_model(model, test_data, test_labels):
    loss, accuracy = model.evaluate(test_data, test_labels)
    return accuracy

@app.route('/test', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        model = load_model(file_path)

        # Dummy test data and labels for example purposes
        test_data = np.random.random((100, 28, 28, 1))
        test_labels = np.random.randint(10, size=(100,))

        accuracy = evaluate_model(model, test_data, test_labels)

        return jsonify({'success': True, 'accuracy': accuracy})

    return jsonify({'success': False, 'message': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)
