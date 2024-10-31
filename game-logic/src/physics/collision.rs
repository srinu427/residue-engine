#[derive(Debug, Clone, Copy)]
pub struct Circle {
  center: glam::Vec3,
  normal: glam::Vec3,
  radius: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Rectangle {
  center: glam::Vec3,
  normal: glam::Vec3,
  tangent: glam::Vec3,
  bitangent: glam::Vec3,
}

#[derive(Debug, Clone, Copy)]
pub enum CollisonShape2D {
  Circle(Circle),
  Rectangle(Rectangle),
}

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
  center: glam::Vec3,
  radius: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum CollisonShape3D {
  Sphere(glam::Vec3, f32),
  Prism(CollisonShape2D, glam::Vec3),
}
