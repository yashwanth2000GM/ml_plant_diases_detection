from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the saved Keras model
MODEL_PATH = "yashwanth3(200e)_plantmodel.keras"  # Change to your actual model file
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize FastAPI app
app = FastAPI()

# Function to preprocess the image
def preprocess_image(image_file):
    image = Image.open(io.BytesIO(image_file))  # Open image
    image = image.resize((128, 128))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize (convert to [0,1] range)
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 128, 128, 3)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read file and preprocess the image
    image_bytes = await file.read()
    input_array = preprocess_image(image_bytes)

    # Get model prediction
    prediction = model.predict(input_array).tolist()
    
    return {"prediction": prediction}