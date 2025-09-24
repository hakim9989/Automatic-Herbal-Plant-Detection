from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Configuration ---
# Plant categories for specific identification 
categories = ["Betel", "Guava", "Lemon", "Mint", "Neem", "Tulsi"]
# Image size for models and processing (must match your ResNet model's input)
img_size = 224

# Confidence threshold for the binary 'is_plant' model.
# If the binary model's confidence for 'plant' is below this, it's considered 'Not a Plant'.
CONFIDENCE_THRESHOLD_IS_PLANT = 0.5 

# --- TensorFlow Model Loading ---
# Load the main multi-class plant identification model (your trained ResNet50)
try:
    plant_identifier_model = tf.keras.models.load_model("Trained_ResNet_model.keras")
    print("Main plant identification model (ResNet50) loaded successfully.")
except Exception as e:
    print(f"Error: Failed to load the main plant identification model: {e}")
    plant_identifier_model = None # Set to None if loading fails

# Load the binary "is plant or not" model (if you trained one)
# This model would have been trained on a dataset with 'plant' and 'non_plant' classes.
try:
    is_plant_model = tf.keras.models.load_model("is_plant_model.keras")
    print("Binary 'is_plant' model loaded successfully.")
except Exception as e:
    print(f"Warning: Binary 'is_plant' model not found or failed to load: {e}. "
          "The API will proceed with specific plant identification without this initial filter.")
    is_plant_model = None # Set to None if loading fails


# In-memory stats (won't persist across server restarts)
scan_stats = {
    "total_scans": 0,
    "recognized_plants": {plant: 0 for plant in categories}
}

# --- Helper Functions ---

# Function to check if an image is a plant using the binary classifier
def is_plant_image(img_array, threshold=CONFIDENCE_THRESHOLD_IS_PLANT):
    """
    Uses the binary classifier model to determine if the image contains a plant.
    Assumes the binary model outputs a single probability for the 'plant' class.
    """
    if is_plant_model is None:
        print("Warning: Binary 'is_plant' model not loaded. Skipping initial plant check.")
        return True # Skip check if model isn't loaded

    # Ensure image is 3-channel (color) for the binary model
    if len(img_array.shape) == 2: # If grayscale, convert to BGR (3 channels)
        img_array_processed = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4: # If RGBA, convert to BGR
        img_array_processed = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else: # Already BGR (3 channels)
        img_array_processed = img_array

    img = cv2.resize(img_array_processed, (img_size, img_size))
    img = img / 255.0 # Normalize pixel values to [0, 1] as per training
    img = np.expand_dims(img, axis=0) # Add batch dimension

    # Get prediction from the binary model
    # Assuming the binary model outputs a single probability (sigmoid activation)
    prediction = is_plant_model.predict(img)[0][0] # Get the scalar probability
    print(f"Binary 'is_plant' check confidence: {prediction:.4f} (Threshold: {threshold})")

    # The 'plant' class is usually mapped to 1 by ImageDataGenerator if it comes alphabetically last
    # in the 'binary_plant_detection_dataset' 'plant' vs 'non_plant' folders.
    # If 'plant' is class 0, you might need 'prediction < (1 - threshold)' or adjust threshold logic.
    return prediction >= threshold

# Function to detect specific plant type using the ResNet50 model
def detect_specific_plant(img_array):
    """
    Preprocesses an image and uses the main multi-class ResNet50 model to predict the plant type.
    """
    if plant_identifier_model is None:
        raise Exception("Main plant identification model is not loaded.")

    # Ensure image is 3-channel (color) as ResNet50 expects it
    if len(img_array.shape) == 2: # If grayscale, convert to BGR (3 channels)
        img_array_processed = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4: # If RGBA, convert to BGR
        img_array_processed = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else: # Already BGR (3 channels)
        img_array_processed = img_array

    img = cv2.resize(img_array_processed, (img_size, img_size))
    img = img / 255.0 # Normalize pixel values to [0, 1] as per training
    img = np.expand_dims(img, axis=0) # Add batch dimension

    prediction = plant_identifier_model.predict(img)
    predicted_class_idx = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return categories[predicted_class_idx], confidence

# Placeholder for plant details
def get_plant_details(plant_name):
    """
    Fetches static details for a given plant name.
    """
    details_map = {
        "Betel": {"scientific_name": "Piper betle", "description": "A vine of the family Piperaceae, whose leaves are commonly chewed, native to Southeast Asia. It is an evergreen, dioecious vine, with glossy heart-shaped leaves and white catkins. Betel plants are cultivated for their leaves which are most commonly used as flavoring for chewing areca nut in so-called betel quid (often confusingly referred to as betel nut, which is toxic and is associated with a wide range of serious health conditions.", "medicinal_uses": "Used in traditional medicine for its antiseptic and anti-inflammatory properties."},
        "Guava": {"scientific_name": "Psidium guajava", "description": "A common tropical fruit cultivated for its sweet fruit.Guava fruits, usually 4 to 12 centimetres long, are round or oval depending on the species The outer skin may be rough, often with a bitter taste, or soft and sweet. Varying between species, the skin can be any thickness, is usually green before maturity, but may be yellow, maroon, or green when ripe,", "medicinal_uses": "Leaves used for treating diarrhea, dysentery, and diabetes."},
        "Lemon": {"scientific_name": "Citrus limon", "description": "A species of small evergreen tree in the flowering plant family Rutaceae.The lemon tree produces a pointed oval yellow fruit. Botanically this is a hesperidium, a modified berry with a tough, leathery rind.Its origins are uncertain, but some evidence suggests lemons originated during the 1st millennium BC in what is now northeastern India.", "medicinal_uses": "Rich in Vitamin C, used for colds, sore throats, and as an antioxidant."},
        "Mint": {"scientific_name": "Mentha", "description": "A genus of plants in the family Lamiaceae.Mints are aromatic, almost exclusively perennial herbs. They have wide-spreading underground and overground stolons and erect, square branched stems. Mints will grow 10–120 cm (4–47 in) tall and can spread over an indeterminate area. Due to their tendency to spread unchecked, some mints are considered invasive.", "medicinal_uses": "Used for digestive issues, headache relief, and as a decongestant."},
        "Neem": {"scientific_name": "Azadirachta indica", "description": "A fast-growing tree in the mahogany family Meliaceae.It is one of the two species in the genus Azadirachta. It is native to the Indian subcontinent and to parts of Southeast Asia, but is naturalized and grown around the world in tropical and subtropical areas. Its fruits and seeds are the source of neem oil. Nim is a Hindustani noun derived from Sanskrit nimba ", "medicinal_uses": "Known for antifungal, antibacterial, and anti-inflammatory properties; used in skincare and pest control."},
        "Tulsi": {"scientific_name": "Ocimum tenuiflorum", "description": "Also known as Holy Basil, an aromatic perennial plant.It is widely cultivated throughout the Southeast Asian tropics. It is native to tropical and subtropical regions of Asia, Australia and the western Pacific.", "medicinal_uses": "An adaptogen, used to reduce stress, support immunity, and aid digestion."}
    }
    return details_map.get(plant_name, {
        "scientific_name": "Unknown Plant",
        "description": "Details not available for this plant type.",
        "medicinal_uses": "N/A"
    })

# --- API Endpoints ---

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint for plant detection. First checks if the image is a plant (if binary model is loaded),
    then proceeds with specific plant classification.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        # Decode image in color as both models expect 3 channels
        img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_color is None:
            return jsonify({"error": "Invalid image file. Could not decode."}), 400

        # --- Step 1: Optional Binary Check (if is_plant_model is loaded) ---
        if is_plant_model is not None:
            if not is_plant_image(img_color):
                return jsonify({
                    "plant_type": "Not a Plant",
                    "confidence": 0.0, # No confidence in specific plant type if not a plant
                    "details": {"name": "Not a Plant", "scientific_name": "N/A", "description": "The uploaded image does not appear to be a plant.", "medicinal_uses": "N/A"},
                    "message": "The uploaded image does not appear to be a plant. Please upload a clear photo of a plant."
                }), 200 # Return 200 OK, but with a specific message

        # --- Step 2: Proceed with specific plant detection using ResNet50 ---
        predicted_plant_type, confidence_score = detect_specific_plant(img_color)

        plant_details = get_plant_details(predicted_plant_type)
        plant_details["name"] = predicted_plant_type # Ensure the name is included in details

        # Update in-memory statistics (optional, won't persist)
        scan_stats["total_scans"] += 1
        if predicted_plant_type in scan_stats["recognized_plants"]:
            scan_stats["recognized_plants"][predicted_plant_type] += 1
        else:
            scan_stats["recognized_plants"]["Unknown"] = scan_stats["recognized_plants"].get("Unknown", 0) + 1
            print(f"Warning: Model predicted an unknown plant type: {predicted_plant_type}")


        return jsonify({
            "plant_type": predicted_plant_type,
            "confidence": round(confidence_score, 2),
            "details": plant_details,
            "message": "Plant identified successfully."
        }), 200

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500


# --- Main Application Run ---
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
