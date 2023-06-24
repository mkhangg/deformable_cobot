# import torch library
import torch

# create tensor without requires_grad = true
x = torch.tensor(3.0, requires_grad = True)

# create tensors with requires_grad = true
w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(5.0, requires_grad = True)

# print the tensors
print("x:", x)
print("w:", w)
print("b:", b)

# define a function y for the above tensors
y = w*x + b
print("y:", y)

# Compute gradients by calling backward function for y
y.backward()

# Access and print the gradients w.r.t x, w, and b
dx = x.grad
dw = w.grad
db = b.grad
print("x.grad :", dx)
print("w.grad :", dw)
print("b.grad :", db)