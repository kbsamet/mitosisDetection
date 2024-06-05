import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras import backend as K

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = K.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(fl)
    return focal_loss_fixed

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

def extract_patches(image, patch_size=32, stride=8):
    patches = []
    coordinates = []
    if len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            coordinates.append((y, x))
    return np.array(patches), coordinates


def reconstruct_mask(pred_patches, coordinates, image_shape, patch_size=32, stride=8):
    mask = np.zeros((image_shape[0], image_shape[1]))
    count = np.zeros((image_shape[0], image_shape[1]))

    for i, (y, x) in enumerate(coordinates):
        mask[y:y + patch_size, x:x + patch_size] += pred_patches[i].squeeze()
        count[y:y + patch_size, x:x + patch_size] += 1
    
    mask = mask / count
    return mask


def preprocess_and_augment(images, masks, patch_size=32, stride=8):
    patch_images = []
    patch_masks = []

    for image, mask in zip(images, masks):
        patches, _ = extract_patches(image, patch_size, stride)
        mask_patches, _ = extract_patches(mask, patch_size, stride)
        patch_images.extend(patches)
        patch_masks.extend(mask_patches)
    
    patch_images = np.array(patch_images)
    patch_masks = np.array(patch_masks)

    # Normalize images
    patch_images = patch_images / 255.0
    
    # Add a channel dimension to masks
    patch_masks = np.expand_dims(patch_masks, axis=-1)

    return patch_images, patch_masks

def cross_validation(images, masks, n_splits=5, patch_size=32, stride=8):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for train_index, val_index in kf.split(images):
        X_train, X_val = images[train_index], images[val_index]
        y_train, y_val = masks[train_index], masks[val_index]

        X_train_patches, y_train_patches = preprocess_and_augment(X_train, y_train, patch_size, stride)
        X_val_patches, y_val_patches = preprocess_and_augment(X_val, y_val, patch_size, stride)

        fold_results.append((X_train_patches, y_train_patches, X_val, y_val, X_val_patches, y_val_patches))
    
    return fold_results

def unet_model(input_size=(32, 32, 3)):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Decoder
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Concatenate()([up6, conv4])
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Concatenate()([up7, conv3])
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Concatenate()([up8, conv2])
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Concatenate()([up9, conv1])
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy', 'Precision', 'Recall'])
    
    return model

def calculate_metrics(y_true, y_pred):
    # Flatten the arrays to calculate the metrics at the pixel level
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    return precision, recall, f1


def main():
    data_dir = 'scanner_A'  # Your data directory
    input_size = (32, 32, 3)  # Define the input size for the model

    # Load images and masks
    images, masks = load_images_and_masks(data_dir)
    print(f"Loaded {len(images)} images and {len(masks)} masks.")

    # Check class distribution
    total_pixels = np.prod(masks.shape)
    num_mitosis_pixels = np.sum(masks)
    num_non_mitosis_pixels = total_pixels - num_mitosis_pixels
    print(f"Mitosis pixels: {num_mitosis_pixels}, Non-mitosis pixels: {num_non_mitosis_pixels}")

    # Perform 5-fold cross-validation
    fold_results = cross_validation(images, masks, patch_size=32, stride=8)

    # Example of using the fold results for training and validation
    for fold_num, (X_train_patches, y_train_patches, X_val, y_val, X_val_patches, y_val_patches) in enumerate(fold_results):
        print(f"Fold {fold_num + 1}")

        # Define and compile the model
        model = unet_model(input_size=input_size)

        # Train the model using the training generator and validate it on the validation data
        model.fit(X_train_patches, y_train_patches,
                  validation_data=(X_val_patches, y_val_patches),
                  epochs=10,  # Adjust the number of epochs as needed
                  batch_size=3)  # Adjust batch size as needed

        # Predict on validation patches
        val_patch_preds = model.predict(X_val_patches)
        val_patch_preds = (val_patch_preds > 0.5).astype(np.uint8)  # Convert predictions to binary masks

        # Reconstruct the full validation mask
        val_mask_pred = reconstruct_mask(val_patch_preds, [c for _, c in extract_patches(X_val[0], patch_size=32, stride=8)], X_val[0].shape, patch_size=32, stride=8)
        
        precision, recall, f1 = calculate_metrics(y_val.flatten(), val_mask_pred.flatten())
        print(f"Fold {fold_num + 1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    print("Training and evaluation completed.")
    model.save('model.h5')
if __name__ == "__main__":
    main()