import matplotlib.pyplot as plt
import time
import numpy as np

def plot(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.show()

def FPS(model, val_gen):
    num_samples = len(val_gen)  
    total_time = 0.0


    start_time = time.time()
    predictions = model.predict(val_gen, steps=num_samples)
    end_time = time.time()
    total_time += end_time - start_time

    average_time_per_inference = total_time / num_samples
    fps = 1 / average_time_per_inference

    print("Average inference time per frame: {:.4f} seconds".format(average_time_per_inference))
    print("Frames per second (FPS): {:.2f}".format(fps))

def visualize(model ,lgr_gen):


    num_samples_to_plot = 10
    predictions = model.predict(lgr_gen, steps=num_samples_to_plot)

    for i in range(num_samples_to_plot):
        batch_images, batch_masks = lgr_gen[i]
        
    
        ground_truth_mask = batch_masks[0] 
        
        predicted_mask = predictions[i]
        
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(ground_truth_mask, cmap="gray", vmin=0, vmax=1)
        plt.title("Ground Truth Mask")
        plt.axis("off")
        
        # Display the predicted mask
        plt.subplot(1, 2, 2)
        plt.imshow(predicted_mask, cmap="gray", vmin=0, vmax=1)
        plt.title("Predicted Mask")
        plt.axis("off")
        
        plt.show()
