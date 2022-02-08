import torch
print(torch.__version__)

x = torch.empty(2, 3)
print(x)

x = torch.rand(2, 2)
print(x)
print(x.dtype)
print(x.size())

x = torch.tensor([1, 2, 3])
print(x)


x = torch.rand(2, 2)
y = torch.rand(2, 2)

z = torch.add(x, y)
print(z)
y.add_(x)
print(y)

z = x - y
z = torch.sub(x, y)

z = x * y
z = torch.mul(x, y)

z = x / y
z = torch.div(x, y)


x = torch.rand(3, 5)
print(x[:, 1])
print(x[1, :])
print(x[1][1].item())

x = torch.tensor([1., 2., 3.], requires_grad=True)
y = torch.tensor([5., 6., 7.], requires_grad=True)
z = torch.sum(x*y)

z.backward()
print("for x:", x.grad)
print("for y:", y.grad)
