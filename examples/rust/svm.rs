// Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
// Released under the MIT license. See LICENSE file in the project root for full license information.

use le::Tensor;

fn main() {
    let x = Tensor::new([[1.0, 2.0, 3.0, 4.0],
                         [4.0, 3.0, 2.0, 1.0]]);

    let y = Tensor::new([[-1.0, -1.0, 1.0, 1.0]]);
    
    print!("Train set:");
    print!("x = {}", x);
    print!("x = {}", y);

    let svm = SVM::new();
    let options = SVM::TrainingOptions {
        kernel: 0.9,
        c: 1.0
    }
    svm.train(x, y, options);
    
    let h = svm.predict(x);
    print!("Predicted value = {}", h);
    
    return 0;
}
