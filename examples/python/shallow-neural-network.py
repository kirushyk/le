#!/usr/bin/env python3
import le

x = le.tensor([[1.0, 2.0, 1.0, 2.0],
               [2.0, 2.0, 1.0, 1.0]])

y = le.tensor([[0.0, 1.0, 1.0, 0.0]])

print('Train set: ')
print('x =\n', x)
print('y =\n', y)

nn = le.Sequential()
nn.add(le.DenseLayer('D1', 2, 2))
nn.add(le.ActivationLayer('A1', le.Activation.SIGMOID))
nn.add(le.DenseLayer('D2', 2, 1))
nn.add(le.ActivationLayer('A2', le.Activation.SIGMOID))
nn.setLoss(le.Loss.LOGISTIC)
h = nn.predict(x)
print('Predicted value =\n', h)
