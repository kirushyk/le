pub trait Model {
    pub fn predict(&mut self, x: &Tensor) -> Tensor;
}
