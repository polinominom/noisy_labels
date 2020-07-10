"""# Model Implementations

## Densenet
"""
from tensorflow.keras.applications.densenet import DenseNet121
def get_densenet():
  return DenseNet121(include_top=True, 
                      weights=None,
                      input_shape=(256, 256, 3), 
                      classes=2)