from typing import Any

import torch
from torch.autograd import Function

class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
        self.needs_input_grad = []
    def save_for_backward(self, *args):
        self._saved_tensors = args
        self.needs_input_grad = [input.requires_grad for input in args]
    @property
    def saved_tensors(self):
        return self._saved_tensors

class Linear(Function):
    @staticmethod
    def forward(ctx: Context, x: torch.tensor, w: torch.tensor, b: torch.tensor) -> Any:
        ctx.save_for_backward(x, w, b)

        output = x.mm(w)

        if b is not None:
            output += b.unsqueeze(0)

        return output

    @staticmethod
    def backward(ctx: Context, grad_output: torch.tensor) -> tuple:
        x, w, b = ctx.saved_tensors
        grad_x = grad_w = grad_b = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output.mm(w.t())

        if ctx.needs_input_grad[1]:
            grad_w = x.t().mm(grad_output)

        if ctx.needs_input_grad[2] and b is not None:
            grad_b = grad_output.sum(dim=0)

        return grad_x, grad_w, grad_b

class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)

        return ((y - yhat).norm() ** 2) / yhat.shape[0]

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors

        grad_yhat = grad_y = None

        if ctx.needs_input_grad[0]:
            grad_yhat = -2 * (y - yhat) / y.shape[0]
            grad_yhat *= grad_output

        if ctx.needs_input_grad[1]:
            grad_y = 2 * (y - yhat) / yhat.shape[0]
            grad_y *= grad_output

        return grad_yhat, grad_y

#  TODO:  Implémenter la fonction Linear(X, W, b)sur le même modèle que MSE

## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

