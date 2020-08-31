"""# Model Implementations

## Densenet
"""
from tensorflow.keras.applications.densenet import DenseNet121
def get_densenet():
  return DenseNet121(include_top=True, 
                      weights='imagenet',
                      input_shape=(224, 224, 3), 
                      classes=2)