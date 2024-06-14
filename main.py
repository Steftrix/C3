import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses, layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50V2
from data_loader import get_data
from model_manager import ModelManager

def adjust_brightness(image, label):
    return tf.image.adjust_brightness(image, 0.01), label

def adjust_contrast(image, label):
    return tf.image.adjust_contrast(image, 1.2), label

def build_resnet_model(model_name='ResNet50V2', print_summary=True):
    resnet50_v2 = ResNet50V2(include_top=False, input_shape=(128, 128, 3))

    for layer in resnet50_v2.layers:
        layer.trainable = False

    model_input = resnet50_v2.input
    hidden = resnet50_v2.output
    hidden = layers.Flatten()(hidden)
    output = layers.Dense(units=120, activation='softmax')(hidden)
    model = Model(inputs=model_input, outputs=output, name=model_name)

    if print_summary:
        print(model.summary())

    return model

def main():
    train_data, validation_data, test_data, steps_per_epoch, validation_steps, label2index, index2label, test_x, test_y = get_data()
    
    model_manager = ModelManager()

    batch_size = 32
    epochs = 50

    earlystopping = EarlyStopping(monitor='val_accuracy', patience=10)

    checkpoint_filepath = './Checkpoints/checkpoint_ResNet50V2DataAug.weights.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        verbose=1,
        mode='max',
        save_best_only=True)

    model = build_resnet_model(model_name='ResNet50V2_DataAug')

    optimizer = optimizers.Adam()
    loss = losses.categorical_crossentropy

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    start_time = time.time()
    training_results = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=[earlystopping, model_checkpoint_callback],
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    execution_time = (time.time() - start_time) / 60.0
    print("Training execution time (mins)", execution_time)

    model_manager.evaluate_save_model(model, training_results, test_data, execution_time, optimizer.learning_rate.numpy(), batch_size, epochs, optimizer)

if __name__ == "__main__":
    main()
