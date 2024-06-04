import numpy as np
import tensorflow as tf
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

def data_generator(image_generator, mask_generator):
    while True:
        image_batch = next(image_generator)
        mask_batch = next(mask_generator)
        yield image_batch, mask_batch

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
        return train_generator, len(images_resized) // 32
    else:
        return images_resized, masks_resized

def cross_validation(images, masks, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for train_index, val_index in kf.split(images):
        X_train, X_val = images[train_index], images[val_index]
        y_train, y_val = masks[train_index], masks[val_index]

        train_generator, steps_per_epoch = preprocess_and_augment(X_train, y_train)
        X_val, y_val = preprocess_and_augment(X_val, y_val, augment=False)

        fold_results.append((train_generator, (X_val, y_val), steps_per_epoch))
    
    return fold_results

def unet_model(input_size=(256, 256, 3)):
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
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])
    
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
    input_size = (256, 256, 3)  # Define the input size for the model

    # Load images and masks
    images, masks = load_images_and_masks(data_dir)
    print(f"Loaded {len(images)} images and {len(masks)} masks.")

    # Check class distribution
    total_pixels = np.prod(masks.shape)
    num_mitosis_pixels = np.sum(masks)
    num_non_mitosis_pixels = total_pixels - num_mitosis_pixels
    print(f"Mitosis pixels: {num_mitosis_pixels}, Non-mitosis pixels: {num_non_mitosis_pixels}")

    # Perform 5-fold cross-validation
    fold_results = cross_validation(images, masks)

    # Example of using the fold results for training and validation
    for fold_num, (train_gen, (X_val, y_val), steps_per_epoch) in enumerate(fold_results):
        print(f"Fold {fold_num + 1}")

        # Here you can define and compile your model
        model = unet_model(input_size=input_size)

        # Create TensorFlow dataset from the generator
        train_dataset = tf.data.Dataset.from_generator(
            lambda: train_gen,
            output_signature=(
                tf.TensorSpec(shape=(None, input_size[0], input_size[1], input_size[2]), dtype=tf.float32),
                tf.TensorSpec(shape=(None, input_size[0], input_size[1], 1), dtype=tf.float32)
            )
        )

        # Train the model using the training generator and validate it on the validation data
        model.fit(train_dataset,
                  validation_data=(X_val, y_val),
                  epochs=10,  # Adjust the number of epochs as needed
                  steps_per_epoch=2)  # Adjust steps per epoch as needed

        # Evaluate the model on the validation data
        val_preds = model.predict(X_val)
        val_preds = (val_preds > 0.5).astype(np.uint8)  # Convert predictions to binary masks

        precision, recall, f1 = calculate_metrics(y_val.flatten(), val_preds.flatten())
        print(f"Fold {fold_num + 1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    print("Training and evaluation completed.")

if __name__ == "__main__":
    main()
