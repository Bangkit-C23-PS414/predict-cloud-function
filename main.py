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

# Model info
folder = "/tmp/"
model = None
class_names = ['Healthy', 'Miner', 'Phoma', 'Rust']

# Initialise client
client = storage.Client("cosmic-anthem-386408")
db = firestore.Client("cosmic-anthem-386408")

def predict(event, context):
    print(f"Processing: {context.resource}")

    # Get resource
    id = event["value"]["fields"]["filename"]["stringValue"]
    path_parts = context.resource.split('/documents/')[1].split('/')
    collection_path = path_parts[0]
    document_path = '/'.join(path_parts[1:])
    affectedDoc = db.collection(collection_path).document(document_path)

    # Download file from GCS
    file_path = folder + id
    bucket = client.get_bucket("cs23-ps414-images-bkt")
    blob = bucket.get_blob(f"images/{id}")
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
    affectedDoc.update({
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
    bucket = client.get_bucket("c23-ps414-statics")

    # Create a blob object from the filepath
    blob = bucket.blob("models/model.h5")

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Download the file to a destination
    blob.download_to_filename(folder + "model.h5")