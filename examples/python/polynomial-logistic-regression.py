#!/usr/bin/env python3
import le

x = le.tensor([[1.0, 2.0, 3.0, 4.0],
               [4.0, 3.0, 2.0, 1.0]])

y = le.tensor([[0.0, 0.0, 1.0, 1.0]])

print('Train set: ')
print('x =\n', x)
print('y =\n', y)

classifier = le.LogisticClassifier()
classifier.train(x, y)
h = classifier.predict(x)
print('Predicted value =\n', h)
