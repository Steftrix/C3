import json
import os
import numpy as np
import decimal
import matplotlib.pyplot as plt
import pandas as pd

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)

class ModelManager:
    def __init__(self):
        pass

    def save_model(self, model, path="./SavedModels", model_name="model01"):
        filename = "./SavedModels/"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        model.save(os.path.join(path, model_name+".h5"))
        model.save_weights(os.path.join(path, model_name+"_weights.h5"))
        model_json = model.to_json()
        with open(os.path.join(path, model_name+".json"), "w") as json_file:
            json_file.write(model_json)

    def get_model_size(self, path="./SavedModels", model_name="model01"):
        model_size = os.stat(os.path.join(path, model_name+".h5")).st_size
        return model_size

    def save_model_metrics(self, model_name="model_1", metrics={}):
        if os.path.exists("./SavedModels/model_metrics.json"):
            with open("./SavedModels/model_metrics.json") as json_file:
                model_metrics = json.load(json_file)
        else:
            model_metrics = {}
        
        model_metrics[model_name] = metrics
        
        with open("./SavedModels/model_metrics.json", 'w') as json_file:
            json_file.write(json.dumps(model_metrics, cls=JsonEncoder))

    def evaluate_save_model(self, model, training_results, test_data, execution_time, learning_rate, batch_size, epochs, optimizer, momentum=None, save=True):
        model_train_history = training_results.history
        num_epochs = len(model_train_history["loss"])

        fig = plt.figure(figsize=(15,5))
        axs = fig.add_subplot(1,2,1)
        axs.set_title('Loss')
        for metric in ["loss", "val_loss"]:
            axs.plot(np.arange(0, num_epochs), model_train_history[metric], label=metric)
        axs.legend()

        axs = fig.add_subplot(1,2,2)
        axs.set_title('Accuracy')
        for metric in ["accuracy", "val_accuracy"]:
            axs.plot(np.arange(0, num_epochs), model_train_history[metric], label=metric)
        axs.legend()

        plt.show()
        
        evaluation_results = model.evaluate(test_data)
        print('Evaluation results: [loss, accuracy]', evaluation_results)
        
        if save:
            self.save_model(model, model_name=model.name)
            model_size = self.get_model_size(model_name=model.name)

            with open(os.path.join("./SavedModels", model.name+"_train_history.json"), "w") as json_file:
                json_file.write(json.dumps(model_train_history, cls=JsonEncoder))

            trainable_parameters = model.count_params()

            metrics = {
                "trainable_parameters": trainable_parameters,
                "execution_time": execution_time,
                "loss": evaluation_results[0],
                "accuracy": evaluation_results[1],
                "model_size": model_size,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                'momentum': momentum,
                "epochs": epochs,
                "optimizer": type(optimizer).__name__
            }
            self.save_model_metrics(model_name=model.name, metrics=metrics)

    def compare_model_metrics(self):
        view_metrics = pd.read_json("./SavedModels/model_metrics.json")
        view_metrics = view_metrics.T
        view_metrics['accuracy'] = view_metrics['accuracy']*100
        view_metrics['accuracy'] = view_metrics['accuracy'].map('{:,.2f}%'.format)
        view_metrics['trainable_parameters'] = view_metrics['trainable_parameters'].map('{:,.0f}'.format)
        view_metrics['execution_time'] = view_metrics['execution_time'].map('{:,.2f} mins'.format)
        view_metrics['model_size'] = view_metrics['model_size']/1000000
        view_metrics['model_size'] = view_metrics['model_size'].map('{:,.0f} MB'.format)
        print('Number of models:', view_metrics.shape[0])
        print(view_metrics.sort_values(by=['accuracy'], ascending=False).head(10))

    def display_current_model_metrics(self, model_name):
        if os.path.exists("./SavedModels/model_metrics.json"):
            with open("./SavedModels/model_metrics.json") as json_file:
                model_metrics = json.load(json_file)
            
            if model_name in model_metrics:
                metrics = model_metrics[model_name]
                view_metrics = pd.DataFrame([metrics])

                view_metrics['accuracy'] = view_metrics['accuracy'] * 100
                view_metrics['accuracy'] = view_metrics['accuracy'].map('{:,.2f}%'.format)
                view_metrics['trainable_parameters'] = view_metrics['trainable_parameters'].map('{:,.0f}'.format)
                view_metrics['execution_time'] = view_metrics['execution_time'].map('{:,.2f} mins'.format)
                view_metrics['model_size'] = view_metrics['model_size'] / 1000000
                view_metrics['model_size'] = view_metrics['model_size'].map('{:,.0f} MB'.format)

                print(view_metrics)
            else:
                print(f"Model {model_name} nu a fost găsit în model_metrics.json.")
        else:
            print("Fisierul model_metrics.json nu există.")
