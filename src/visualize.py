import os
import matplotlib.pyplot as plt
import numpy as np


def plot_loss(train_loss, valid_loss, output_path):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'loss.png'))


def plot_evaluation_metrics(eval_results, output_path):
    # plot accuracy
    plt.figure(figsize=(10, 7))
    plt.plot(eval_results['accuracy'], color='blue', label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(output_path, 'Accuracy.png'))

    # plot precision
    plt.figure(figsize=(10, 7))
    plt.plot(eval_results['micro/precision'], color='orange', label='micro/precision')
    plt.plot(eval_results['macro/precision'], color='red', label='macro/precision')
    plt.plot(eval_results['samples/precision'], color='yellow', label='samples/precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'Precision.png'))

    # plot recall
    plt.figure(figsize=(10, 7))
    plt.plot(eval_results['micro/recall'], color='orange', label='micro/recall')
    plt.plot(eval_results['macro/recall'], color='red', label='macro/recall')
    plt.plot(eval_results['samples/recall'], color='yellow', label='samples/recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'recall.png'))

    # plot f1
    plt.figure(figsize=(10, 7))
    plt.plot(eval_results['micro/f1'], color='orange', label='micro/f1')
    plt.plot(eval_results['macro/f1'], color='red', label='macro/f1')
    plt.plot(eval_results['samples/f1'], color='yellow', label='samples/f1')
    plt.xlabel('Epochs')
    plt.ylabel('F1_score')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'F1.png'))


def plot_inference(image, string_predicted, string_actual, counter, save_path):
    image = image.squeeze(0)
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
    plt.savefig(os.path.join(save_path, f"{counter}.jpg"))
