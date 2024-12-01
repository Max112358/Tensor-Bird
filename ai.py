import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

def create_flappy_model(model_path='saved_model'):
    try:
        if os.path.exists(model_path):
            print("Loading existing model...")
            return keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading existing model: {e}")
        print("Creating new model instead...")
    
    print("Creating model...")
    model = keras.Sequential([
        # Input layer - exactly 5 features
        keras.layers.InputLayer(input_shape=(5,)),
        
        # Expand to learn feature combinations
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        
        # Contract to distill features
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        
        # Output layer
        keras.layers.Dense(units=2, activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def save_model(model, model_path='saved_model'):
    """Save the model to disk with error handling"""
    if model is None:
        print("Error: Cannot save None model")
        return False
        
    try:
        model.save(model_path)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

