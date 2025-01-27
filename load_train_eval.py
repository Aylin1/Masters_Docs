import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import tensorflow as tf



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
        plt.imshow(X[i, :, :, :3])  # Display the first 3 bands as an RGB image
        plt.axis('off') 
        
        # Plot the ground truth
        plt.subplot(1, 3, 2)
        plt.imshow(Y[i].squeeze(), cmap='gray')  # Ground truth mask
        plt.axis('off')  
        
        # Plot the prediction
        plt.subplot(1, 3, 3)
        plt.imshow(pred[i].squeeze(), cmap='gray')  # Predicted mask
        plt.axis('off') 
        
        plt.tight_layout()
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


def train_and_evaluate(X_train, Y_train, X_val, Y_val, build_model_fn, batch_size=8, epochs=25, learning_rate=0.001, threshold=0.5, visualize_pred=True):
    """
    Train a segmentation model, evaluate its performance, and return results including metrics, history, and predictions.

    Args:
        X_train (np.ndarray): Full-resolution training images.
        Y_train (np.ndarray): Full-resolution training masks.
        X_val (np.ndarray): Full-resolution validation images.
        Y_val (np.ndarray): Full-resolution validation masks.
        build_model_fn (function): Function to build the model.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        threshold (float): Threshold to binarize predicted mask. Default is 0.5.

    Returns:
        dict: Dictionary containing evaluation metrics, training history, and predicted images.
    """
    # Build and compile the model
    input_shape = X_train.shape[1:]
    model = build_model_fn(input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model with validation split
    print("Training the model...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        epochs=epochs,
    )

    # Evaluate predictions and calculate metrics
    print("\nEvaluating model performance...")
    metrics = {'IoU': [], 'Dice': [], 'Accuracy': [], 'Precision': [], 'Recall': []}
    predictions = model.predict(X_val)
    predictions_binarized = (predictions > threshold).astype(np.uint8)  # Binarize predictions

    if visualize_pred:
        visualize_prediction(model, X_val, Y_val)

    for i in range(len(X_val)):
        y_true = Y_val[i].squeeze().flatten()
        y_pred = predictions_binarized[i].squeeze().flatten()

        iou = jaccard_score(y_true, y_pred)
        metrics['IoU'].append(iou)
        metrics['Dice'].append(f1_score(y_true, y_pred))
        metrics['Accuracy'].append(accuracy_score(y_true, y_pred))
        metrics['Precision'].append(precision_score(y_true, y_pred, zero_division=1))
        metrics['Recall'].append(recall_score(y_true, y_pred, zero_division=1))

    # Calculate mean, min, and max IoU metrics
    mean_metrics = {key: np.mean(values) for key, values in metrics.items()}
    mean_metrics['Min IoU'] = np.min(metrics['IoU'])
    mean_metrics['Max IoU'] = np.max(metrics['IoU'])

    print("\n==== Model Evaluation Metrics ====")
    for metric, value in mean_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Visualize predictions (if applicable)
    predicted_images = []
    for i in range(min(len(X_val), 5)):  # Limit visualization to first 5 validation samples
        predicted_images.append({
            "input_image": X_val[i],
            "ground_truth": Y_val[i],
            "predicted_mask": predictions[i],
            "predicted_mask_binarized": predictions_binarized[i]
        })

    # Return results as a dictionary
    return {
        "metrics": metrics,              # List of metrics for each validation sample
        "mean_metrics": mean_metrics,    # Mean, min, and max IoU metrics across the validation dataset
        "history": history.history,      # Training history (loss, accuracy, etc.)
        "predicted_images": predicted_images  # Sample predicted images for visualization
    }


# Function to plot the comparison of different augmentation histories
def plot_comparison_history(history_dict):
    """
    Plot the comparison of loss and accuracy for different augmentations.

    Args:
        history_dict (dict): Dictionary containing loss and accuracy histories for each augmentation.
    """
    # Plot the losses for different augmentations
    plt.figure(figsize=(12, 6))

    # Losses
    plt.subplot(1, 2, 1)
    for i, loss in enumerate(history_dict['loss']):
        plt.plot(loss, label=f"Augmentation {i+1} Training Loss")
    for i, val_loss in enumerate(history_dict['val_loss']):
        plt.plot(val_loss, label=f"Augmentation {i+1} Validation Loss", linestyle='--')
    plt.title('Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracies
    plt.subplot(1, 2, 2)
    for i, accuracy in enumerate(history_dict['accuracy']):
        plt.plot(accuracy, label=f"Augmentation {i+1} Training Accuracy")
    for i, val_accuracy in enumerate(history_dict['val_accuracy']):
        plt.plot(val_accuracy, label=f"Augmentation {i+1} Validation Accuracy", linestyle='--')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
