import torch
from pathlib import Path
from tp2_module import Model
from tp2_dataset import MyDataSet
from sklearn.datasets import fetch_openml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


ITERATION = 50
DATA_PATH = '../data/'

save_path = Path("model.pch")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiClassModel(torch.nn.Module):
    def  __init__(self):
        super(MultiClassModel, self).__init__()
        self.linear1 = torch.nn.Linear(784, 200)
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(200, 64)
        self.linear3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        x = self.linear3(x)
        return x

class State:
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0, 0


if save_path.is_file():
    with save_path.open('rb') as fp:
        state = torch.load(fp)

else:
    model = MultiClassModel()
    model = model.to(device)
    optim = torch.optim.Adam(params=model.parameters())
    state = State(model, optim)

x, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False, data_home=DATA_PATH)
minst_dataset = MyDataSet(x, y)
batch = DataLoader(minst_dataset, shuffle=True, batch_size=32)
loss = torch.nn.CrossEntropyLoss()
writer = SummaryWriter(log_dir='./runs/checkpoint')

for epoch in range(state.epoch, ITERATION):
    total_loss = 0.0
    print(f'processing {epoch} epoch')
    for x_train, y_train in batch:
        state.optim.zero_grad()
        x_train = x_train.to(device).to(torch.float32)
        y_train = torch.tensor(list(map(lambda z: int(z), y_train))).to(device)

        yhat = state.model(x_train)

        l = loss(yhat, y_train)

        total_loss += l

        state.optim.step()
        state.iteration += 1

    writer.add_scalar('Loss/train', total_loss, epoch)

    with save_path.open('wb') as fp:
        state.epoch += 1
        torch.save(state, fp)



