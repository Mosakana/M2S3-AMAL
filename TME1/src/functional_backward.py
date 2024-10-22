import torch
import matplotlib.pyplot as plt

def descent_grad(x, w, y, epoch, lr):
    l = []
    for i in range(epoch):
        output = x.mm(w)
        loss = ((y - output) ** 2).sum() / 2
        l.append(loss)

        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            w.grad.zero_()

    return l

x = torch.randn(100, 30, requires_grad=True)
w = torch.randn(30, 30, requires_grad=True)
y = torch.randn(100, 30)


l = descent_grad(x, w, y, 100, 5e-5)

plt.plot(list(range(100)), torch.tensor(l))
plt.show()

