struct SVM {

}

struct SVMTrainingOptions {
    kernel: f32,
    c: f32
}

impl SVM {
    pub fn new() -> SVM {

    }
    pub fn train(&mut self, x: &Tensor, y: &Tensor, options: SVMTrainingOptions) {

    }
}

impl Model for SVM {
    pub fn predict(&mut self, x: &Tensor) -> Tensor {

    }
}

