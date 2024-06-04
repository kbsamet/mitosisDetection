import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images_and_masks(data_dir):
    images = []
    masks = []
    
    for patient_folder in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, patient_folder)):
            continue
        patient_path = os.path.join(data_dir, patient_folder)
        
        for file in os.listdir(patient_path):
            if file.endswith('.bmp'):
                # Load image
                image_path = os.path.join(patient_path, file)
                image = cv2.imread(image_path)
                images.append(image)
                
                # Create corresponding mask
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                csv_file = file.replace('.bmp', '.csv')
                csv_path = os.path.join(patient_path, csv_file)
                
                if os.path.exists(csv_path):
                    try:
                        with open(csv_path, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                values = line.strip().split(',')
                                # Ensure even number of values (pairs of coordinates)
                                if len(values) % 2 != 0:
                                    print(f"Warning: Skipping invalid row in {csv_path}: {line}")
                                    continue
                                # Convert pairs of coordinates to integers and set mask
                                for i in range(0, len(values), 2):
                                    try:
                                        row_idx = int(values[i])
                                        col_idx = int(values[i+1])
                                        mask[row_idx, col_idx] = 1
                                    except ValueError:
                                        print(f"Warning: Skipping invalid coordinate pair in {csv_path}: ({values[i]}, {values[i+1]})")
                    
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")
                
                masks.append(mask)
    
    return np.array(images), np.array(masks)
def preprocess_and_augment(images, masks, target_size=(256, 256), augment=True):
    # Resize images and masks to the target size
    images_resized = np.array([cv2.resize(image, target_size) for image in images])
    masks_resized = np.array([cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST) for mask in masks])
    
    # Normalize images
    images_resized = images_resized / 255.0
    
    # Add a channel dimension to masks
    masks_resized = np.expand_dims(masks_resized, axis=-1)
    
    if augment:
        data_gen_args = dict(
            rotation_range=90,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )

        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)

        image_datagen.fit(images_resized, augment=True)
        mask_datagen.fit(masks_resized, augment=True)

        image_generator = image_datagen.flow(images_resized, batch_size=32, seed=1)
        mask_generator = mask_datagen.flow(masks_resized, batch_size=32, seed=1)

        train_generator = data_generator(image_generator, mask_generator)
        return train_generator
    else:
        return images_resized, masks_resized

    
def data_generator(image_generator, mask_generator):
    while True:
        image_batch = next(image_generator)
        mask_batch = next(mask_generator)
        yield image_batch, mask_batch


def cross_validation(images, masks, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for train_index, val_index in kf.split(images):
        X_train, X_val = images[train_index], images[val_index]
        y_train, y_val = masks[train_index], masks[val_index]

        train_generator = preprocess_and_augment(X_train, y_train)
        X_val, y_val = preprocess_and_augment(X_val, y_val, augment=False)

        fold_results.append((train_generator, (X_val, y_val)))
    
    return fold_results
