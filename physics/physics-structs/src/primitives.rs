use geometry::{glam, Orientation};
use polygon_face::PolygonFace;
use sphere::Sphere;

pub mod polygon_face;
pub mod sphere;

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

#[derive(Debug, Clone)]
pub enum RigidBodyType {
  PolygonFace(PolygonFace),
  Sphere(Sphere),
}

impl RigidBodyType {
  pub fn oriented(&self, orientation: Orientation) -> Self {
    match self {
      Self::PolygonFace(polygon_face) => Self::PolygonFace(polygon_face.oriented(orientation)),
      Self::Sphere(sphere) => Self::Sphere(sphere.oriented(orientation)),
    }
  }
}

#[derive(Debug, Clone)]
pub struct RigidBody {
  bodies: Vec<RigidBodyType>,
  collision_mask: u32,
  mass: Mass,
  velocity: glam::Vec3,
  acceleration: glam::Vec3,
  moment_of_inertia: MomentOfInertia,
  angular_velocity: glam::Vec3,
  angular_acceleration: glam::Vec3,
  orientation: Orientation,
}

impl RigidBody {
  pub fn new(
    bodies: Vec<RigidBodyType>,
    collision_mask: u32,
    mass: Mass,
    velocity: glam::Vec3,
    acceleration: glam::Vec3,
    moment_of_inertia: MomentOfInertia,
    angular_velocity: glam::Vec3,
    angular_acceleration: glam::Vec3,
    orientation: Orientation,
  ) -> Self {
    Self {
      bodies,
      collision_mask,
      mass,
      velocity,
      acceleration,
      moment_of_inertia,
      angular_velocity,
      angular_acceleration,
      orientation,
    }
  }
}
