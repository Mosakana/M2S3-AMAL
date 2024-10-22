import torch
from torch.utils.tensorboard import SummaryWriter

from tp1 import MSE, Linear, Context


# Les données supervisées
x = torch.randn(50, 13, requires_grad=True)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

epsilon = 5e-5

writer = SummaryWriter(log_dir='./runs/artificial_data')
for n_iter in range(1000):
    output = Linear.apply(x, w, b)
    loss = MSE.apply(output, y)

    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    loss.backward()

    with torch.no_grad():
        w -= epsilon * w.grad
        b -= epsilon * b.grad
        w.grad.zero_()
        b.grad.zero_()


