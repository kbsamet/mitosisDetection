from model import unet_model, calculate_metrics
from preprocess import load_images_and_masks,cross_validation
import numpy as np
import tensorflow as tf

def main():
    data_dir = 'scanner_A'  # Your data directory
    input_size = (256, 256, 3)  # Define the input size for the model

    # Load images and masks
    images, masks = load_images_and_masks(data_dir)
    print(f"Loaded {len(images)} images and {len(masks)} masks.")

    # Perform 5-fold cross-validation
    fold_results = cross_validation(images, masks)

    # Example of using the fold results for training and validation
    for fold_num, (train_gen, (X_val, y_val)) in enumerate(fold_results):
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
                  steps_per_epoch=len(X_val) // 3)  # Adjust batch size as needed

        # Evaluate the model on the validation data
        val_preds = model.predict(X_val)
        val_preds = (val_preds > 0.5).astype(np.uint8)  # Convert predictions to binary masks

        precision, recall, f1 = calculate_metrics(y_val.flatten(), val_preds.flatten())
        print(f"Fold {fold_num + 1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    print("Training and evaluation completed.")
    model.save('unet_model.keras')

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
