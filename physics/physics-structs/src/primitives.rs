use geometry::{glam, Direction, LineSegment, Orientation, Point};
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
pub struct RigidBodyPhysicsProps {
  mass: Mass,
  velocity: glam::Vec3,
  acceleration: glam::Vec3,
  orientation: Orientation,
}

#[derive(Debug, Clone)]
pub struct RigidPoint {
  raw: Point,
  thickness: f32,
}

impl RigidPoint {
  pub fn min_dist_from_r_point(&self, r_point: &RigidPoint) -> f32 {
    (self.raw.as_vec3() - r_point.raw.as_vec3()).length() - self.thickness - r_point.thickness
  }
}

#[derive(Debug, Clone)]
pub struct RigidLineSegment {
  raw: LineSegment,
  thickness: f32,
}

#[derive(Debug, Clone)]
pub struct RigidCircle {
  center: Point,
  normal: Direction,
  radius: f32,
  thickness: f32,
}

#[derive(Debug, Clone)]
pub struct RigidRectangle {
  center: Point,
  u: Direction,
  v: Direction,
  n: Direction,
  u_len: f32,
  v_len: f32,
  thickness: f32,
}

#[derive(Debug, Clone)]
pub struct RigidCuboid {
  center: Point,
  u: Direction,
  v: Direction,
  w: Direction,
  u_len: f32,
  v_len: f32,
  w_len: f32,
  thickness: f32,
}
