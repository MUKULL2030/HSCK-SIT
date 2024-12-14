from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
import PyPDF2
import docx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask('__name__', static_folder='static')

nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained ML model
model = joblib.load('resume_match_model.pkl')

# Define functions to parse and extract resume data (similar to previous examples)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'main.html') #ADD YOUR HTML FILE HERE
@app.route('/analyze', methods=['POST'])
def analyze_file():
    try:
        file = request.files['file']
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        if validate_file_format(file_path):
            resume_data = parse_resume(file_path)
            features = prepare_features(resume_data)
            match = model.predict(features)
            return jsonify({
                'resume_data': resume_data,
                'match': int(match[0])
            })
        else:
            return jsonify({'error': 'Invalid file format'}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred while processing the file.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
