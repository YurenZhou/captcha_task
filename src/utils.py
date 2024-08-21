import cv2

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_bi = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # separate the captcha to characters
    contours, _ = cv2.findContours(image_bi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # Sort left to right
    segmented = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char_img = image_bi[y:y+h, x:x+w]
        char_img = cv2.resize(char_img, (10, 10))  # Resize for consistency
        segmented.append(char_img)

    return segmented