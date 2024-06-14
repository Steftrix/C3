import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_loader import get_data 
from model_manager import ModelManager

def main():
    _, _, test_data, _, _, label2index, index2label, test_x, test_y = get_data()
    model = load_model('./SavedModels/ResNet50V2_DataAug.h5')
    def preprocess_image(path):
        image = Image.open(path)
        image = image.resize((128, 128)) 
        image = np.array(image) / 255.0
        return image

    test_x_display = [preprocess_image(path) for path in test_x]
    test_x_display = np.asarray(test_x_display)
    test_predictions = model.predict(test_x_display)

    true_predict = 0
    false_predict = 0
    fig = plt.figure(figsize=(20, 16))
    for i, file in enumerate(test_x_display[:50]):
        axs = fig.add_subplot(10, 5, i + 1)
        axs.set_aspect('equal')
        predicted_breed = index2label[test_predictions.argmax(axis=1)[i]][10:]
        true_breed = test_y[i][10:]
        if true_breed == predicted_breed:
            axs.set_title('Predicție: ' + predicted_breed + '\n' + 'Adevărat: ' + true_breed, fontsize=8, color='green')
            true_predict += 1
        else:
            axs.set_title('Predicție: ' + predicted_breed + '\n' + 'Adevărat: ' + true_breed, fontsize=8, color='red')
            false_predict += 1
        plt.imshow(test_x_display[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    print('# de predicții corecte: ', true_predict)
    print('# de predicții greșite: ', false_predict)

    model_manager = ModelManager()
    model_manager.display_current_model_metrics('ResNet50V2_DataAug')

if __name__ == "__main__":
    main()
