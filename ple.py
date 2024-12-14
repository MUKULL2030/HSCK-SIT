import json
import PyPDF2
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')

# Function to extract text from a PDF resume
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Function to analyze text using BERT
def analyze_text_with_bert(text):
    # Load a pre-trained BERT model for sentiment analysis or text classification
    nlp = pipeline("sentiment-analysis")  # You can change this to a specific task if needed

    # Analyze the text
    sentiment_results = nlp(text)

    # Simple keyword-based analysis for strengths and weaknesses
    strengths_keywords = ['leadership', 'team player', 'communication', 'problem-solving', 'adaptability']
    weaknesses_keywords = ['perfectionism', 'impatience', 'lack of experience', 'overthinking']

    strengths = []
    weaknesses = []

    # Tokenize the text for analysis
    tokens = word_tokenize(text.lower())

    # Identify strengths
    for keyword in strengths_keywords:
        if keyword in tokens:
            strengths.append(keyword)

    # Identify weaknesses
    for keyword in weaknesses_keywords:
        if keyword in tokens:
            weaknesses.append(keyword)

    # Generate insights and suggestions based on sentiment analysis
    insights = f"Overall sentiment: {sentiment_results[0]['label']} with score {sentiment_results[0]['score']:.2f}"
    suggestions = "Consider focusing on the identified strengths and addressing the weaknesses."

    return strengths, weaknesses, insights, suggestions

# Main function to handle the resume processing
def process_resume(file_path):
    # Step 1: Extract text from the resume
    extracted_text = extract_text_from_pdf(file_path)

    # Step 2: Analyze the extracted text using BERT
    strengths, weaknesses, insights, suggestions = analyze_text_with_bert(extracted_text)

    # Step 3: Create a JSON object with the results
    result_json = json.dumps({
        'strengths': strengths,
        'weaknesses': weaknesses,
        'insights': insights,
        'suggestions': suggestions
    }, indent=4)

    return result_json

# Example usage
if __name__ == "__main__":
    resume_file_path = r"C:\Users\Mukul Prasad\Desktop\SIT HACK_CULT\MAIN\resuu.pdf"  # Replace with your resume file path
    result = process_resume(resume_file_path)
    print(result)