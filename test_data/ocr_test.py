import cv2
import easyocr

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Read image
image = cv2.imread("test_data/Screenshot 2025-12-14 095507.png")

# Perform OCR
results = reader.readtext(image)

# Print extracted text
print("Extracted Text:\n")
for detection in results:
    print(detection[1])

