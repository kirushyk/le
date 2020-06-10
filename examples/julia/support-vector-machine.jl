x = Tensor([[1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0]])

y = Tensor([[-1.0, -1.0, 1.0, 1.0]])

println("Train set: ")
println("x =\n", x)
println("y =\n", y)

svm = SVM()
svm.train(x, y)
h = svm.predict(x)
println("Predicted value =\n", h)
