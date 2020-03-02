import torch

x = torch.ones(2,2,requires_grad=True)
print(x.grad_fn)
#print(x)
y = x + 2
print(y.grad_fn)
print('---')
z = y * y * 3
out = z.mean()

print(z)
print(out)

print('---')
a = torch.randn(2,2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a**2).sum()
print(b.grad_fn)

print('---')
out.backward()
print(x.grad)

print('---')
x = torch.randn(3, requires_grad=True)

y = x * 2
print(y)
print(y.data)
print(y.data.norm())
while y.data.norm() < 1000:
    y = y * 2

print(y)
print(y.norm())
print(y.data.norm())


scores = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(scores)

print(x.grad)
print('---')
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)