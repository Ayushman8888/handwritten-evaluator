import pytesseract
import cv2
import numpy as np
from PIL import Image


def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove noise
    gray = cv2.medianBlur(gray, 3)

    # Apply threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return thresh


def extract_text(image_path):
    processed_image = preprocess_image(image_path)

    # Convert OpenCV image to PIL format
    pil_image = Image.fromarray(processed_image)

    # Extract text
    text = pytesseract.image_to_string(pil_image)

    return text.strip()
