import torch
import numpy as np
import torchvision.transforms as tforms
import torchvision.datasets as dtsets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


class Evaluation():
    def __init__(self, data_dir, interpolate, model) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.data_dir = data_dir
        self.interpolate = interpolate

        self.model = model

        self.eval_results = {
            'confusion_matrix': None,
            'accuracy': None,
            'f1_score': None
        }

    def load_data(self):
        transforms = tforms.Compose([
            tforms.ToTensor(),
            # tforms.Normalize(mean=(0.5,), std=(0.5,)),
            # tforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            tforms.Lambda(lambd=lambda X: X / 255)
        ])
        if self.interpolate:
            transforms = tforms.Compose([
                tforms.Resize((self.interpolate, self.interpolate)),
                transforms
            ])

        test_set = dtsets.FashionMNIST(
            root=self.data_dir, download=True, train=False, transform=transforms)

        return test_set

    def get_test_loader(self):
        dataset = self.load_data()
        test_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=1, shuffle=False)
        return test_loader

    def evaluate(self):
        test_loader = self.get_test_loader()
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.model(X)
                _, pred = torch.max(y_hat, 1)
                y_true.append(y.data.item())
                y_pred.append(pred.data.item())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        self.eval_results['confusion_matrix'] = confusion_matrix(
            y_true, y_pred)
        self.eval_results['accuracy'] = accuracy_score(y_true, y_pred)
        self.eval_results['f1_score'] = f1_score(
            y_true, y_pred, average='macro')

        return self.eval_results

    def plot_results(self):
        print('Accuray: {0}'.format(self.eval_results['accuracy']))
        print('f1-score: {0}'.format(self.eval_results['f1_score']))
        sns.heatmap(self.eval_results['confusion_matrix'],
                    annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()
