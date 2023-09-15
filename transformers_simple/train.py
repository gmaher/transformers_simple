import torch
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, dataset, loss_fn=torch.nn.MSELoss(), batch_size=8,
    learning_rate=1e-3, epochs=1, log_interval=20, workers=0,
    device=torch.device('cpu')):
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = epochs
        self.log_interval = log_interval
        self.workers=workers
        self.device = device

    def run(self):
        train_loader = DataLoader(
            self.dataset,
            sampler=torch.utils.data.RandomSampler(self.dataset, replacement=True),
            shuffle=False,
            pin_memory=True,
            batch_size=self.batch_size
        )
        opt = torch.optim.Adam(self.model.parameters(),lr=self.lr)

        self.model.train()

        for e in range(self.epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                opt.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                opt.step()
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        e, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

        self.model.eval()
