# Noisy Labels

## File & folder descriptions
### chexpert_downloader.py

- This script will download the pre-adjusted chexpert data which are saved to a google drive. 
- The file ids of the adjusted data is in the file namely "file_ids.csv". This file and it's content is necessary for the downloader.

### chexpert_noisfy.py

- Main objective of this file is to generate semantic noisy labels for the chexpert dataset.
- Consisted of three neural network models
    - Densenet121
    - Resnet50
    - VGG16
- Simply trains these each networks under very few percent of the real data, then tries to find the uncertain decisions towards the unseen data betwen networks.

### adjusted_data and labels folders

- Contains chunks of labels and adjusted images that are going to be used by noise generation and general training

### history and models folders
- Consisted of trained model and the result of the training(history)
