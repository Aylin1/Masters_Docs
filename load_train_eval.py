import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import rasterio
import tensorflow as tf


def load_raster_data(file_path):
    with rasterio.open(file_path) as src:
        data = src.read()
    return np.moveaxis(data, 0, -1)  # Move channel axis to the end

def load_data(matches, resize_shape=(256, 256)):
    X, Y = [], []
    for stacked_tif, ground_truth_tif in matches.items():
        # Load stacked TIF and ground truth mask
        stacked = load_raster_data(stacked_tif)
        ground_truth = load_raster_data(ground_truth_tif)[..., 0]  # Use the first channel
        
        # Resize to consistent shape
        stacked = tf.image.resize(stacked, resize_shape).numpy()
        ground_truth = tf.image.resize(ground_truth[..., np.newaxis], resize_shape).numpy()
        
        X.append(stacked)
        Y.append(ground_truth)
    
    return np.array(X), np.array(Y)


def visualize_prediction(model, X, Y):
    """
    Visualize predictions for the segmentation task.

    Parameters:
    - model: Trained segmentation model.
    - X: Input data (stacked raster).
    - Y: Ground truth masks.
    """
    pred = model.predict(X)
    for i in range(len(X)):
        plt.figure(figsize=(15, 5))
        
        # Plot the input (first three bands as RGB)
        plt.subplot(1, 3, 1)
        plt.title("Input (First 3 Bands)")
        plt.imshow(X[i, :, :, :3])  # Display the first 3 bands as an RGB image
        
        # Plot the ground truth
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(Y[i].squeeze(), cmap='gray')  # Ground truth mask
        
        # Plot the prediction
        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(pred[i].squeeze(), cmap='gray')  # Predicted mask
        
        plt.show()

def evaluate_predictions(model, X, Y, threshold=0.5):
    """
    Evaluate predictions of the segmentation model using IoU, Dice, Accuracy, Precision, and Recall.
    
    Parameters:
    - model: Trained segmentation model.
    - X: Input images (stacked raster).
    - Y: Ground truth masks.
    - threshold: Threshold to binarize predicted mask (default = 0.5).
    
    Returns:
    - metrics: Dictionary containing evaluation metrics (IoU, Dice, Accuracy, Precision, Recall).
    """
    metrics = {'IoU': [], 'Dice': [], 'Accuracy': [], 'Precision': [], 'Recall': []}
    
    # Generate predictions for X
    predictions = model.predict(X)
    predictions = (predictions > threshold).astype(np.uint8)  # Binarize predictions
    
    for i in range(len(X)):
        y_true = Y[i].squeeze().flatten()
        y_pred = predictions[i].squeeze().flatten()
        
        iou = jaccard_score(y_true, y_pred)
        dice = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        
        metrics['IoU'].append(iou)
        metrics['Dice'].append(dice)
        metrics['Accuracy'].append(accuracy)
        metrics['Precision'].append(precision)
        metrics['Recall'].append(recall)
    
    # Calculate the mean of all metrics
    mean_metrics = {key: np.mean(values) for key, values in metrics.items()}
    print("\n==== Model Evaluation Metrics ====")
    for metric, value in mean_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics, mean_metrics


def train_and_evaluate(X_train, Y_train, X_val, Y_val, build_model_fn, batch_size=8, epochs=25, learning_rate=0.001, visualize=True):
    """
    Train and evaluate a model using full-resolution training and validation data.

    Args:
        X_train (np.ndarray): Full-resolution training images.
        Y_train (np.ndarray): Full-resolution training masks.
        X_val (np.ndarray): Full-resolution validation images.
        Y_val (np.ndarray): Full-resolution validation masks.
        build_model_fn (function): Function to build the model.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        visualize (bool): Whether to visualize predictions. Default is True.

    Returns:
        dict: Dictionary containing metrics for the trained model.
    """
    # Build and compile the model
    input_shape = X_train.shape[1:]
    model = build_model_fn(input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        epochs=epochs
    )

    # Plot training results
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.legend()
    plt.show()

    # Evaluate the model
    print("Evaluating model...")
    if visualize:
        visualize_prediction(model, X_val, Y_val)
    metrics, mean_metrics = evaluate_predictions(model, X_val, Y_val, threshold=0.5)

    return {
        "metrics": metrics,
        "mean_metrics": mean_metrics,
    }

