"""# Model Implementations

## Densenet
"""
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.models import Model
def get_densenet(num_classes):
  base_model = DenseNet121(include_top=False, weights='imagenet')
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  predictions = Dense(num_classes, activation='sigmoid')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
  return model