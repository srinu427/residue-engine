use geometry::Direction;

#[derive(Debug, Copy, Clone)]
pub enum SingleBodyForce {
  ConstantForce { value: Direction },
  ConstantAcceleration { value: Direction },
}

#[derive(Debug, Copy, Clone)]
pub enum CouplingForce {
  ConstantForce { value: f32, min_distance: f32, max_distance: f32 },
  ConstantAcceleration { value: f32, min_distance: f32, max_distance: f32 },
  Spring { pull_constant: f32, push_constant: f32, length: f32 },
  InverseSquare { constant: f32, min_distance: f32, max_distance: f32 },
}
