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
import json

# Model info
folder = "/tmp/"
model = None
class_names = ['Healthy', 'Miner', 'Phoma', 'Rust']

# Initialise client
client = storage.Client("cosmic-anthem-386408")
db = firestore.Client("cosmic-anthem-386408")

def predict(event, context):
    print(f"Processing: {context.resource}")
    logs = dict()

    # Get resource
    id = event["value"]["fields"]["filename"]["stringValue"]
    path_parts = context.resource.split('/documents/')[1].split('/')
    collection_path = path_parts[0]
    document_path = '/'.join(path_parts[1:])
    affectedDoc = db.collection(collection_path).document(document_path)

    # Download file from GCS
    start_time = time.time()
    file_path = folder + id
    bucket = client.get_bucket("cs23-ps414-images-bkt")
    blob = bucket.get_blob(f"images/{id}")
    blob.download_to_filename(file_path)
    end_time = time.time()
    logs["download-image"] = end_time - start_time

    # Use the global model variable 
    global model
    if not model:
        # Download model
        start_time = time.time()
        download_model_file()
        end_time = time.time()
        logs["download-model"] = end_time - start_time

        # Load model
        start_time = time.time()
        model = tf.keras.models.load_model(
            folder + "model.h5",
            custom_objects={"KerasLayer": hub.KerasLayer})
        end_time = time.time()
        logs["load-model"] = end_time - start_time

        # Warm up
        start_time = time.time()
        tensor_zeros = tf.zeros([1, 224, 224, 3], tf.float64)
        model.predict(tensor_zeros, verbose=0)
        end_time = time.time()
        logs["warm-up-model"] = end_time - start_time

    # Transform image
    start_time = time.time()
    original_image = Image.open(file_path)
    image = transform_image(original_image)
    end_time = time.time()
    logs["transform-image"] = end_time - start_time

    # Predict image
    start_time = time.time()
    prediction = model.predict(image, verbose=0)
    end_time = time.time()
    inference_time = end_time - start_time
    logs["inference-time"] = end_time - start_time

    # Get prediction class
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    logs["label"] = predicted_class
    logs["confidence"] = float(confidence)

    # Update data
    affectedDoc.update({
        'label': predicted_class,
        'confidence': float(confidence),
        'inferenceTime': round(inference_time * 1000),
        'detectedAt': round(end_time * 1000),
        'isDetected': True
    })

    # Clean up
    os.remove(file_path)
    gc.collect()
    print(f"Logs: {context.resource} {json.dumps(logs)}")


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
