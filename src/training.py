import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tforms
import torchvision.datasets as dtsets
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary


class Training():
    def __init__(self, data_dir, train_ratio, interpolate, ckpt_dir,
                 from_ckpt, model_ptt, model_params,
                 optimizer_ptt, optimizer_params,
                 scheduler_ptt, scheduler_params):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.interpolate = interpolate
        self.ckpt_dir = ckpt_dir

        checkpoint = None
        if from_ckpt:
            ckpt_path = os.path.join(self.ckpt_dir, 'checkpoint.pth')
            checkpoint = torch.load(ckpt_path)

        self.model_params = model_params
        self.model = self.get_model(model_ptt=model_ptt,
                                    model_params=model_params,
                                    checkpoint=checkpoint).to(self.device)
        self.optimizer = self.get_optimizer(optimizer_ptt=optimizer_ptt,
                                            optimizer_params=optimizer_params,
                                            checkpoint=checkpoint)
        if scheduler_ptt and scheduler_params:
            self.scheduler = self.get_scheduler(scheduler_ptt=scheduler_ptt,
                                                scheduler_params=scheduler_params,
                                                checkpoint=checkpoint)
        else:
            self.scheduler = None

        self.train_history = {
            'train_loss': [],
            'val_acc': []
        }

    def get_model(self, model_ptt, model_params, checkpoint=None):
        model = model_ptt(**model_params)
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def get_model_summary(self):
        input_shape = (self.model_params['img_channel'],
                       self.model_params['img_size'],
                       self.model_params['img_size'])
        return summary(self.model, input_shape)

    def get_optimizer(self, optimizer_ptt, optimizer_params, checkpoint=None):
        optimizer = optimizer_ptt(params=self.model.parameters(),
                                  **optimizer_params)
        if checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return optimizer

    def get_scheduler(self, scheduler_ptt, scheduler_params, checkpoint=None):
        scheduler = scheduler_ptt(optimizer=self.optimizer, **scheduler_params)
        if checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return scheduler

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

        train_set = dtsets.FashionMNIST(
            root=self.data_dir, download=True, train=True, transform=transforms)

        train_size = int(len(train_set) * self.train_ratio)
        val_size = len(train_set) - train_size
        train_set, val_set = torch.utils.data.random_split(
            train_set, [train_size, val_size])

        return train_set, val_set

    def get_data_loader(self, batch_sizes):
        train_set, val_set = self.load_data()
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=batch_sizes[0],
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                 batch_size=batch_sizes[1],
                                                 shuffle=False)
        return train_loader, val_loader

    def train(self, epochs, save_every, batch_sizes):
        criterion = nn.CrossEntropyLoss()
        train_loader, val_loader = self.get_data_loader(
            batch_sizes=batch_sizes)

        for epoch in tqdm(range(epochs), desc='Epochs', total=epochs,
                          leave=False, unit='epoch'):
            batch_loss = 0
            count = 0
            for X, y in train_loader:
                self.model.train()
                self.optimizer.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.model(X)
                loss = criterion(y_hat, y)
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                batch_loss += loss.data.item()
                count += 1
            self.train_history['train_loss'].append(batch_loss / count)

            self.model.eval()
            self.train_history['val_acc'].append(
                self.evaluate(val_loader=val_loader))

            if (epoch + 1) % save_every == 0:
                self.save()

        return self.train_history

    def evaluate(self, val_loader):
        correct = 0
        total = 0
        for X, y in val_loader:
            X, y = X.to(self.device), y.to(self.device)
            y_hat = self.model(X)
            _, pred = torch.max(y_hat, 1)
            correct += (pred == y).sum().item()
            total += len(X)
        return correct / total

    def save(self):
        ckpt_path = os.path.join(self.ckpt_dir, 'checkpoint.pth')
        if self.scheduler:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            }, ckpt_path)
        else:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': None
            }, ckpt_path)

    def plot_results(self):
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.plot(self.train_history['train_loss'], color=color)
        ax1.set_xlabel('epoch', color=color)
        ax1.set_ylabel('total loss', color=color)
        ax1.tick_params(axis='y', color=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('accuracy', color=color)
        ax2.plot(self.train_history['val_acc'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()

        plt.show()
