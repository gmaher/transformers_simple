import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model, dataset, loss_fn=torch.nn.MSELoss(), batch_size=8,
    learning_rate=1e-3, epochs=1, max_iters=None, log_interval=20, workers=0, grad_norm_clip=1.0,
    device=torch.device('cpu'), val_dataset=None, val_interval=None):
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = epochs
        self.max_iters = max_iters
        self.log_interval = log_interval
        self.workers=workers
        self.device = device
        self.grad_norm_clip = grad_norm_clip
        self.val_dataset = val_dataset
        self.val_interval = val_interval
        self.loss_history = []
        self.val_loss_history = []
        self.val_loss_iters = []

    def evaluate_val_loss(self):
        val_loader = DataLoader(
            self.val_dataset,
            sampler=torch.utils.data.RandomSampler(self.val_dataset, replacement=False),
            shuffle=False,
            pin_memory=True,
            batch_size=self.batch_size
        )

        val_loss = 0
        N = len(val_loader.dataset)/self.batch_size
        for batch_idx, (data, target) in tqdm(enumerate(val_loader)):
            output = self.model(data)
            loss = self.loss_fn(output,target)

            val_loss += loss.item()*1.0/N

        return val_loss

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                loss.backward()
                opt.step()
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        e, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))


                    self.loss_history.append(loss.item())

                if self.val_dataset is not None and self.val_interval is not None and\
                batch_idx%self.val_interval == 0:
                    val_loss = self.evaluate_val_loss()
                    self.val_loss_history.append(val_loss)
                    count = e*len(train_loader.dataset) + batch_idx*self.batch_size
                    self.val_loss_iters.append(count)

                if self.max_iters is not None and batch_idx >= self.max_iters:
                    break

        self.model.eval()
