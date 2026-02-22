from flask import Flask, render_template, request
import os
import easyocr
import cv2

from ocr_test import (
    preprocess_image,
    split_theory_math,
    evaluate_theory,
    evaluate_math
)

app = Flask(__name__)

# -------------------------
# Config
# -------------------------
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------
# Initialize OCR reader ONCE (VERY IMPORTANT)
# -------------------------
print("⏳ Loading EasyOCR model (first time may be slow)...")
reader = easyocr.Reader(['en'], gpu=False)
print("✅ EasyOCR loaded")

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        # 1️⃣ Validate file
        if "file" not in request.files:
            return "❌ No file part found"

        file = request.files["file"]

        if file.filename == "":
            return "❌ No file selected"

        # 2️⃣ Save file
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        print(f"📁 File saved: {filepath}")

        try:
            # 3️⃣ Preprocess image
            processed_img = preprocess_image(filepath)

            # 4️⃣ OCR
            results = reader.readtext(processed_img)
            theory_answers, math_answers = split_theory_math(results)

            # -------------------------
            # Teacher Answer Key (STATIC for now)
            # Later: make this uploadable
            # -------------------------
            expected_theory_keywords = [
                "skewness",
                "kurtosis",
                "calculate",
                "probability"
            ]

            expected_math_values = [
                31.5,
                0.48
            ]

            # 5️⃣ Evaluation
            theory_score, theory_max = evaluate_theory(
                theory_answers,
                expected_theory_keywords
            )

            math_score, math_max = evaluate_math(
                math_answers,
                expected_math_values
            )

            total_score = theory_score + math_score

            # 6️⃣ Render result
            return render_template(
                "result.html",
                filename=file.filename,
                theory_answers=theory_answers,
                math_answers=math_answers,
                theory_score=theory_score,
                theory_max=theory_max,
                math_score=math_score,
                math_max=math_max,
                total_score=total_score
            )

        except Exception as e:
            print("❌ Error:", e)
            return f"<h3>Error occurred:</h3><pre>{str(e)}</pre>"

    # GET request
    return render_template("index.html")


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    app.run(
        debug=True,
        threaded=True   # prevents browser hanging
    )
