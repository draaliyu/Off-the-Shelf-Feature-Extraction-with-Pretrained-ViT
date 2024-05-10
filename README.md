# Off the Shelf Feature Extraction with Pretrained ViT
This project consists of a series of scripts designed for loading data, processing images, extracting features, training a model, and evaluating the model for an image classification task.

## Directory Structure
- ├── load_data.py
- ├── processing.py
- ├── feature_extraction.py
- ├── train_model.py
- ├── evaluate_model.py
- ├── main.py
- └── README.md

## Scripts
### 1. load_data.py
This script is responsible for loading and preprocessing image data.

##### Functions:
- load_image(image_path): Loads an image, converts it to RGB, resizes it, and converts it to a numpy array.
- load_data(dataset_path): Loads images and labels from the dataset directory, splits the data into training and validation sets, and returns them along with class mappings.

### 2. processing.py
This script contains functions for further preprocessing the image data.

##### Functions:
- normalize_images(images): Normalizes the image pixel values.
- augment_images(images): Applies data augmentation techniques to the images.


### 3. feature_extraction.py
This script is used for extracting features from images using a pre-trained model.

##### Functions:
- extract_features(images, model): Extracts features from the images using the specified model.

### 4. train_model.py
This script handles the training of the image classification model.

##### Functions:
- train_model(X_train, y_train, model): Trains the model using the provided training data.

### 5. evaluate_model.py
This script evaluates the performance of the trained model on the validation set.

##### Functions:
- evaluate_model(X_val, y_val, model): Evaluates the model using the provided validation data.

### 6. main.py
This is the main script that orchestrates the entire image classification pipeline from data loading to model evaluation.

## Dependencies
Ensure you have the following dependencies installed:
- numpy
- Pillow
- scikit-learn
- keras
- vit-keras
- matplotlib
- pandas
- seaborn
- joblib
- PIL

## Example Directory Structure for Dataset
### dataset_path/
    class1/
        image1.jpg
        image2.jpg
        ...
    class2/
        image1.jpg
        image2.jpg
        ...
    ...

Each subdirectory within dataset_path corresponds to a different class and contains the images belonging to that class.

## Notes
The load_data.py script uses an 80-20 split for training and validation data.

