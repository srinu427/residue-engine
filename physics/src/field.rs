pub struct Gravity {
  const_value: f32,
}

impl Gravity {
  pub fn new(const_value: f32) -> Self {
    Gravity { const_value }
  }

  pub fn value_at(&self, pos: glam::Vec3) -> f32 {
    self.const_value
  }
}
