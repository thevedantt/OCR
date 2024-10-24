import os
import pytesseract
from flask import Flask, request, render_template
from PyPDF2 import PdfReader
from PIL import Image
import io

# If tesseract is not in your PATH, specify the full path to tesseract.exe here
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if needed

app = Flask(__name__)

# Route to handle the main index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the OCR functionality
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']

    if file.filename == '':
        return "No selected file"
    
    if file and file.filename.endswith('.pdf'):
        # Handle PDF file
        text = extract_text_from_pdf(file)
    else:
        # Handle image file
        text = extract_text_from_image(file)

    return f"<h1>Extracted Text:</h1><pre>{text}</pre>"

# Function to extract text from an image
def extract_text_from_image(image_file):
    img = Image.open(image_file)  # Open the image
    text = pytesseract.image_to_string(img)  # Perform OCR
    return text

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)  # Open the PDF file
    text = ""
    
    # Extract text from each page
    for page in reader.pages:
        text += page.extract_text()
    
    return text

if __name__ == '__main__':
    app.run(debug=True)
