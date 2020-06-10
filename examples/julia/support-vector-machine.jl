x = Tensor([[1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0]])

y = Tensor([[-1.0, -1.0, 1.0, 1.0]])

print("Train set: ")
print("x =\n", x)
print("y =\n", y)

svm = le.SVM();
svm.train(x, y)
h = svm.predict(x)
print("Predicted value =\n", h)
