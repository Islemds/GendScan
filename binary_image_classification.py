from build_model import build_model_EfficientNetB0, build_model_EfficientNetB4, build_model_EfficientNetV2B3, build_model_EfficientNetV2L, build_model_InceptionV3
import tensorflow as tf
import numpy as np
from PIL import Image

# Define label names
label_names = ['Fake', 'Pristine']

# Function to preprocess a single image (expects a NumPy array)
def preprocess_image(img, img_size=(224, 224)):
    # Ensure the image is a tensor
    img_resized = tf.image.resize(img, img_size)  # Resize the image
    img_array = np.expand_dims(img_resized, axis=0)  # Expand dimensions for batch (single image batch)
    return img_array

# Example list of model names
model_names = [
    'EfficientNetB0',
    'EfficientNetB4',
    'EfficientNetV2B3',
    'EfficientNetV2L',
    'InceptionV3'
]

# Function to load and return a model with pretrained weights
def load_model(model_function, num_classes=2):
    model = model_function(num_classes)
    model.load_weights(f"{model.name}_weights.h5")  # Adjust this path based on where you store the weights
    return model

# Example list of model loading functions
model_functions = [
    lambda num_classes: load_model(build_model_EfficientNetB0, num_classes),
    lambda num_classes: load_model(build_model_EfficientNetB4, num_classes),
    lambda num_classes: load_model(build_model_EfficientNetV2B3, num_classes),
    lambda num_classes: load_model(build_model_EfficientNetV2L, num_classes),
    lambda num_classes: load_model(build_model_InceptionV3, num_classes)
]

# Function to predict on a single image using a loaded model
def predict_on_image(model, img):
    # Preprocess image
    input_image = preprocess_image(img)
    
    # Predict on the image
    predictions = model.predict(input_image)
    return predictions

# Function to aggregate predictions from multiple models
def aggregate_predictions(predictions):
    # Perform voting or calculate average probabilities
    num_models = len(predictions)
    final_predictions = {
        'models': [],
        'predicted_labels': [],
        'probabilities': []
    }
    
    # Aggregate predictions
    for i in range(num_models):
        model_name = model_names[i]
        model_pred = predictions[i]
        predicted_class = np.argmax(model_pred, axis=1)[0]
        probability = np.max(model_pred, axis=1)[0]
        
        final_predictions['models'].append(model_name)
        final_predictions['predicted_labels'].append(label_names[predicted_class])
        final_predictions['probabilities'].append(probability)
    
    # Calculate final predicted label based on voting or average probability
    labels, counts = np.unique(final_predictions['predicted_labels'], return_counts=True)
    majority_label = labels[np.argmax(counts)]
    majority_index = final_predictions['predicted_labels'].index(majority_label)
    final_probability = np.mean(final_predictions['probabilities'])
    
    final_prediction = {
        # 'models': final_predictions['models'],
        'predicted_label': majority_label,
        'predicted_probability': final_probability
    }
    
    return final_prediction