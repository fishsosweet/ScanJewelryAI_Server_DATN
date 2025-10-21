# Import Cloudinary API for listing resources
from cloudinary.api import resources
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from services.cloudinary_service import upload_to_cloudinary
from services.ollama_service import extract_keywords
from services.cloudinary_service import search_images_by_keywords
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client import QdrantClient
import os
import logging
import time  # For retry logic
import base64
from flask_cors import CORS
import io
import requests
# Initialize CORS
from dotenv import load_dotenv
load_dotenv()
# Disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
QD_URL = os.getenv('QD_URL')
QD_API_KEY = os.getenv('QD_API_KEY')
qdrant_client = QdrantClient(
    url=QD_URL,
    api_key=QD_API_KEY,
)


def qdrant_request_with_retries(func, *args, retries=3, backoff_multiplier=2, **kwargs):
    """Custom retry logic for Qdrant requests."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                wait_time = backoff_multiplier ** attempt
                logging.warning(
                    f"Retrying Qdrant request in {wait_time} seconds... (Attempt {attempt + 1})")
                time.sleep(wait_time)
            else:
                logging.error(
                    f"Qdrant request failed after {retries} attempts: {e}")
                raise


# Initialize Qdrant client

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Load pre-trained ResNet50 model for feature extraction
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Function to extract features from an image


def extract_features_from_pil_image(img):
    try:
        img = img.convert("RGB").resize((224, 224))
        image_array = img_to_array(img)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        features = model.predict(image_array)
        return features.flatten()
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None


def extract_features(image_path):
    try:
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        features = model.predict(image_array)
        return features.flatten()
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None


# Create a collection in Qdrant
collection_name = "image_search"
if not qdrant_request_with_retries(qdrant_client.collection_exists, collection_name=collection_name):
    qdrant_request_with_retries(
        qdrant_client.create_collection,
        collection_name=collection_name,
        vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
    )


def clean_qdrant_collection():
    """Delete and recreate the Qdrant collection."""
    try:
        if qdrant_request_with_retries(qdrant_client.collection_exists, collection_name=collection_name):
            logging.info(f"Deleting existing collection: {collection_name}")
            qdrant_request_with_retries(
                qdrant_client.delete_collection, collection_name=collection_name)
        logging.info(f"Creating collection: {collection_name}")
        qdrant_request_with_retries(
            qdrant_client.create_collection,
            collection_name=collection_name,
            vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
        )
    except Exception as e:
        logging.error(f"Error cleaning Qdrant collection: {e}")


def process_images_folder():
    """Fetch and process images from the 'TrangSuc' folder in Cloudinary."""
    folder_name = "TrangSuc"
    try:
        # Fetch the list of images from the Cloudinary folder
        response = resources(
            type="upload", prefix=folder_name, max_results=1100)
        images = response.get("resources", [])

        if not images:
            logging.warning(
                f"No images found in Cloudinary folder '{folder_name}'.")
            return

        clean_qdrant_collection()  # Clean the collection before processing

        for idx, image in enumerate(images):
            image_url = image.get("secure_url")
            if not image_url:
                logging.error(f"Image at index {idx} has no URL.")
                continue

            # Download the image and save it locally (optional)
            response = requests.get(image_url)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))

            # Extract features
            vector = extract_features_from_pil_image(img)
            if vector is not None:  # Only upsert if feature extraction was successful
                try:
                    qdrant_request_with_retries(
                        qdrant_client.upsert,
                        collection_name=collection_name,
                        points=[{
                            "id": idx,
                            "vector": vector,
                            "payload": {
                                "filename": image.get("public_id"),
                                "image_url": image_url
                            }
                        }],
                    )
                    logging.info(
                        f"Inserted image '{image.get('public_id')}' into Qdrant with URL: {image_url}")
                except Exception as e:
                    logging.error(
                        f"Error inserting image '{image.get('public_id')}' into Qdrant: {e}")
    except Exception as e:
        logging.error(
            f"Error processing images from Cloudinary folder '{folder_name}': {e}")


def image_to_base64(image_path):
    """Convert an image to a Base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error converting image to Base64: {e}")
        return None

##############################################
# chạy cái này dễ quan sát trong terminal
# process_images_folder()
##############################################


@app.route("/", methods=["GET"])
def hello():
    return jsonify({"message": "Hello, World!"})


@app.route("/process-images", methods=["POST"])
def process_images():
    """Endpoint to trigger processing of the images folder."""
    try:
        process_images_folder()
        return jsonify({"message": "Images processed successfully."})
    except Exception as e:
        logging.error(f"Error processing images: {e}")
        return jsonify({"error": "Failed to process images."}), 500


@app.route("/search", methods=["POST"])
def search():
    """Search endpoint to find similar images."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    # query_image = request.files["image"]
    # query_image.save(query_path)
    query_image = request.files["image"]
    img = Image.open(io.BytesIO(query_image.read()))
    query_vector = extract_features_from_pil_image(img)
    if query_vector is None:
        return jsonify({"error": "Failed to extract features from the query image."}), 400

    try:
        search_result = qdrant_request_with_retries(
            qdrant_client.search,
            collection_name=collection_name,
            query_vector=query_vector,
            limit=15,
        )
        results = [
            {
                "id": hit.id,
                "score": hit.score,
                "filename": hit.payload["filename"],
                "image_url": hit.payload["image_url"]
            }
            for hit in search_result
        ]
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error during search: {e}")
        return jsonify({"error": "Search operation failed."}), 500

@app.route('/searchAI', methods=['POST']) 
def search_images():
    data = request.json
    prompt = data.get('query', '')

    if not prompt:
        return jsonify({"error": "Thiếu từ khóa"}), 400

    # 1. Trích xuất từ khóa từ câu mô tả
    keywords = extract_keywords(prompt)
    print("Từ khóa:", keywords)

    # 2. Tìm ảnh trong Cloudinary theo từ khóa
    results = search_images_by_keywords(keywords)

    return jsonify({"keywords": keywords, "results": results})



if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
