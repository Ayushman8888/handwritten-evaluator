import cv2
import easyocr
import numpy as np
import re
from pymongo import MongoClient

# ---------------------------
# OCR → THEORY / MATH SPLIT
# ---------------------------
def split_theory_math(ocr_results):
    theory, math = [], []

    for item in ocr_results:
        text = item[1].strip()
        letters = len(re.findall(r'[A-Za-z]', text))
        numbers = len(re.findall(r'[0-9=+\-*/^%]', text))

        if letters > numbers:
            theory.append(text)
        else:
            math.append(text)

    return theory, math


# ---------------------------
# IMAGE PREPROCESSING
# ---------------------------

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh

# ---------------------------
# THEORY EVALUATION (KEYWORDS)
# ---------------------------
def evaluate_theory(theory_answers, expected_keywords):
    score = 0

    for keyword in expected_keywords:
        for ans in theory_answers:
            if keyword.lower() in ans.lower():
                score += 1
                break

    return score, len(expected_keywords)


# ---------------------------
# MATH EVALUATION
# ---------------------------
def extract_number(text):
    match = re.search(r'-?\d+(\.\d+)?', text)
    return float(match.group()) if match else None


def evaluate_math(math_answers, expected_values, tolerance=0.5):
    score = 0
    limit = min(len(math_answers), len(expected_values))

    for i in range(limit):
        num = extract_number(math_answers[i])
        if num is None:
            continue

        if abs(num - expected_values[i]) <= tolerance:
            score += 1

    return score, limit


# ---------------------------
# TEACHER ANSWER KEY
# ---------------------------
expected_theory_keywords = [
    "skewness",
    "kurtosis",
    "calculate"
]

expected_math_values = [
    31.5,
    0.48
]


# ---------------------------
# MAIN PIPELINE
# ---------------------------
image_path = "test_data/Screenshot 2025-12-14 095507.png"

processed_img = preprocess_image(image_path)
cv2.imwrite("test_data/processed_image.png", processed_img)
print("✅ Processed image saved")

reader = easyocr.Reader(['en'])
results = reader.readtext(processed_img)

theory_answers, math_answers = split_theory_math(results)

print("\n📘 THEORY ANSWERS:")
for t in theory_answers:
    print("-", t)

print("\n📐 MATH ANSWERS:")
for m in math_answers:
    print("-", m)

theory_score, theory_max = evaluate_theory(
    theory_answers, expected_theory_keywords
)

math_score, math_max = evaluate_math(
    math_answers, expected_math_values
)

print("\n📘 THEORY SCORE:", theory_score, "/", theory_max)
print("📐 MATH SCORE:", math_score, "/", math_max)
print("\n🎯 TOTAL SCORE:", theory_score + math_score, "/", theory_max + math_max)


# ---------------------------
# SAVE TO MONGODB
# ---------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["handwritten_evaluator"]
collection = db["results"]

collection.insert_one({
    "student": "Test Student",
    "theory_answers": theory_answers,
    "math_answers": math_answers,
    "theory_score": theory_score,
    "math_score": math_score,
    "total_score": theory_score + math_score
})

print("✅ Result saved to MongoDB")
