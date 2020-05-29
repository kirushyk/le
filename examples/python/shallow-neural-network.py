#!/usr/bin/env python3
import le

x = le.tensor([[1.0, 2.0, 1.0, 2.0],
               [2.0, 2.0, 1.0, 1.0]])

y = le.tensor([[0.0, 1.0, 1.0, 0.0]])

print("Train set: ")
print("x =\n", x)
print("y =\n", y)

nn = le.Sequential();
h = nn.predict(x)
print("Predicted value =\n", h)
