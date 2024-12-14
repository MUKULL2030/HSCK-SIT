from flask import Flask, request, render_template, jsonify
import docx

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files:
        return jsonify(success=False, error="No file uploaded.")
    
    file = request.files['resume']
    
    if file.filename == '':
        return jsonify(success=False, error="No selected file.")

    try:
        # Assuming the file is a .docx
        doc = docx.Document(file)
        text = " ".join([para.text for para in doc.paragraphs])
        # Perform analysis on text
        analysis_result = f"Resume contains {len(text.split())} words."
        print(analysis_result)
        return jsonify(success=True, message=analysis_result)
    except Exception as e:
        print(str(e))
        return jsonify(success=False, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
