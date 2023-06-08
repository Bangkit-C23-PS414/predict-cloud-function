from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from google.cloud import storage
from google.cloud import firestore
import numpy as np
import time
import tensorflow as tf
import tensorflow_hub as hub
import os
import gc
import uuid
import functions_framework

# Model info
folder = "/tmp/"
model = None
class_names = ['Healthy', 'Miner', 'Phoma', 'Rust']

# Model Bucket details
PROJECT_ID = "cosmic-anthem-386408"
GCS_MODEL_BUCKET_NAME = "c23-ps414-statics"
GCS_MODEL_FILE = "models/model.h5"

# Initialise client
client = storage.Client(PROJECT_ID)
db = firestore.Client(PROJECT_ID)

@functions_framework.cloud_event
def predict(event):
    # TODO: Remove this when deploy
    event = event.data

    # Download file from GCS
    file_path = folder + str(uuid.uuid4())
    bucket = client.get_bucket(event["bucket"])
    blob = bucket.get_blob(event["name"])
    blob.download_to_filename(file_path)
    
    # Use the global model variable 
    global model
    if not model:
        download_model_file()
        model = tf.keras.models.load_model(folder + "model.h5", custom_objects = {"KerasLayer" : hub.KerasLayer})

    # Transform image
    original_image = Image.open(file_path)
    image = transform_image(original_image)

    # Predict image
    start_time = time.time() * 1000
    prediction = model.predict(image)
    end_time = time.time() * 1000

    # Get prediction class
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    inference_time = end_time - start_time

    # Update data
    doc_id = os.path.basename(event["name"])
    doc_ref = db.collection('images').document(doc_id)
    doc_ref.update({
        'label': predicted_class,
        'confidence': float(confidence),
        'inferenceTime': round(inference_time),
        'detectedAt': round(end_time),
        'isDetected': True
    })

    # Clean up
    os.remove(file_path)
    gc.collect()


def transform_image(img):
    img = img_to_array(img)
    img = img.astype(np.float64) / 255
    imgs = tf.image.resize(img, [224, 224])
    imgs = np.expand_dims(imgs, axis=0)
    return imgs


def download_model_file():
    # Create a bucket object for our bucket
    bucket = client.get_bucket(GCS_MODEL_BUCKET_NAME)

    # Create a blob object from the filepath
    blob = bucket.blob(GCS_MODEL_FILE)

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Download the file to a destination
    blob.download_to_filename(folder + "model.h5")
