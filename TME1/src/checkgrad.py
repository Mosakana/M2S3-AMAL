import unittest

import torch
from tp1 import mse, linear

class CheckGrad(unittest.TestCase):
    def test_mse(self):
        yhat = torch.randn(10, 5, requires_grad=True, dtype=torch.float64)
        y = torch.randn(10, 5, requires_grad=True, dtype=torch.float64)

        check_mse = torch.autograd.gradcheck(mse, (yhat, y))
        self.assertEqual(True, check_mse)

    def test_linear(self):
        x = torch.randn(20, 12, requires_grad=True, dtype=torch.float64)
        w = torch.randn(12, 7, requires_grad=True, dtype=torch.float64)
        b = torch.randn(7, requires_grad=True, dtype=torch.float64)

        check_linear = torch.autograd.gradcheck(linear, (x, w, b))
        self.assertEqual(True, check_linear)

if __name__ == '__main__':
    unittest.main()
