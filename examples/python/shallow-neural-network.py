#!/usr/bin/env python3
import le

x = le.tensor([[1.0, 2.0, 1.0, 2.0],
               [2.0, 2.0, 1.0, 1.0]])

y = le.tensor([[0.0, 1.0, 1.0, 0.0]])

print('Train set:')
print('x =\n', x)
print('y =\n', y)

nn = le.Sequential()
nn.add(le.DenseLayer('D1', 2, 2))
nn.add(le.ActivationLayer('A1', le.Activation.SIGMOID))
nn.add(le.DenseLayer('D2', 2, 1))
nn.add(le.ActivationLayer('A2', le.Activation.SIGMOID))
nn.setLoss(le.Loss.LOGISTIC)

print('Training Neural Network')
optimizer = le.BGD(nn, x, y, 3.0)
for i in range(1000):
    optimizer.step()
    if i % 100 == 0:
        print('Iteration', i)
        h = nn.predict(x)
        print('Training Error =', format(le.logistic_loss(h, y), '.3f'))

h = nn.predict(x)
print('Predicted value =\n', h)
