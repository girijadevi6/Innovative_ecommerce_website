import cv2
import numpy as np
import pyzxing

def detect_barcode(image_path):
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use OpenCV's built-in barcode detector
    barcode_detector = cv2.barcode_BarcodeDetector()
    retval, decoded_info, decoded_type, points = barcode_detector.detectAndDecode(gray)

    if retval:
        for i, barcode in enumerate(decoded_info):
            print(f"Barcode {i+1}: {barcode}")
            if "EXP" in barcode or "EXPIRY" in barcode:
                print(f"Expiry Date Found: {barcode.split()[-1]}")
        
        # Draw bounding box
        for point in points:
            if point is not None:
                pts = np.int32(point).reshape(-1, 1, 2)
                cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Show Image
        cv2.imshow("Barcode Detected", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No barcode detected.")

# Provide the image path containing a barcode
detect_barcode("barcode_image.jpg")
