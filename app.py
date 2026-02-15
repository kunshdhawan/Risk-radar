from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from malware_analysis import MalwareTester 

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_path = 'enhanced_malware_model.pkl'
encoder_path = 'encoder.joblib'
analyzer = MalwareTester(model_path)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("main.html", error="No file selected")

        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(file_path)

        try:
            result = analyzer.analyze_file(file_path)
            return render_template("main.html", result=result)
        except Exception as e:
            return render_template("main.html", error=f"Error analyzing file: {str(e)}")

    return render_template("main.html")

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")

if __name__ == '__main__':
    app.run(debug=True)
