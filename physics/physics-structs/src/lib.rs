pub mod primitives;

#[derive(Debug, Copy, Clone)]
pub enum Mass {
  Infinite,
  Finite(f32),
}

#[derive(Debug, Copy, Clone)]
pub enum MomentOfInertia {
  Infinite,
  Finite(glam::Mat3),
}
