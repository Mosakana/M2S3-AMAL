import torch
from torch.utils.tensorboard import SummaryWriter

from TME1.src.tp1 import MSE, Linear, Context



def SGD_optimizer(x, w, b, y, epsilon):
    optim = torch.optim.SGD(params=[w, b], lr=epsilon)

    writer = SummaryWriter(log_dir='./runs/sdg')
    for n_iter in range(100):
        loss = MSE.apply(Linear.apply(x, w, b), y)

        # `loss` doit correspondre au coût MSE calculé à cette itération
        # on peut visualiser avec
        # tensorboard --logdir runs/
        writer.add_scalar('Loss/train', loss, n_iter)

        # Sortie directe
        print(f"Itérations {n_iter}: loss {loss}")

        loss.backward()

        optim.step()
        optim.zero_grad()

def Adam_optimizer(x, w, b, y, epsilon):
    optim = torch.optim.Adam(params=[w, b], lr=epsilon)

    writer = SummaryWriter(log_dir='./runs/adam')
    for n_iter in range(100):
        loss = MSE.apply(Linear.apply(x, w, b), y)

        # `loss` doit correspondre au coût MSE calculé à cette itération
        # on peut visualiser avec
        # tensorboard --logdir runs/
        writer.add_scalar('Loss/train', loss, n_iter)

        # Sortie directe
        print(f"Itérations {n_iter}: loss {loss}")

        loss.backward()

        optim.step()
        optim.zero_grad()

def naive_module(x, y, epsilon, device):
    m1 = torch.nn.Linear(13, 3, device=device)
    tanh = torch.nn.Tanh()
    mse = torch.nn.MSELoss()

    optim = torch.optim.Adam(params=[m1.weight, m1.bias], lr=epsilon)

    writer = SummaryWriter(log_dir='./runs/naive_module')

    for n_iter in range(100):
        loss = mse(tanh(m1(x)), y)
        writer.add_scalar('Loss/train', loss, n_iter)
        print(f"Itérations {n_iter}: loss {loss}")
        loss.backward()
        optim.step()
        optim.zero_grad()

def sequential_module(x, y, epsilon, device):
    model = torch.nn.Sequential(
        torch.nn.Linear(13, 3, device=device),
        torch.nn.Tanh()
    )

    mse = torch.nn.MSELoss()

    optim = torch.optim.Adam(params=model.parameters(), lr=epsilon)

    writer = SummaryWriter(log_dir='./runs/squential_module')

    for n_iter in range(100):
        loss = mse(model(x), y)
        print(f"Itérations {n_iter}: loss {loss}")
        writer.add_scalar('Loss/train', loss, n_iter)

        loss.backward()

        optim.step()
        optim.zero_grad()

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(13, 3)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        return self.tanh(self.linear(x))

def class_module(x, y, epsilon, device):
    model = Model()
    mse = torch.nn.MSELoss()

    optim = torch.optim.Adam(params=model.parameters(), lr=epsilon)

    writer = SummaryWriter(log_dir='./runs/class_module')

    for n_iter in range(100):
        loss = mse(model(x), y)
        print(f"Itérations {n_iter}: loss {loss}")
        writer.add_scalar('Loss/train', loss, n_iter)

        loss.backward()

        optim.step()
        optim.zero_grad()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Les données supervisées
    x = torch.nn.Parameter(torch.randn(50, 13, requires_grad=True, device=device))
    y = torch.randn(50, 3)

    # Les paramètres du modèle à optimiser
    w = torch.nn.Parameter(torch.randn(13, 3, requires_grad=True, device=device))
    b = torch.nn.Parameter(torch.randn(3, requires_grad=True, device=device))

    epsilon = 5e-5

    class_module(x, y, epsilon, device)


