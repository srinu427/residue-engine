use glam::Vec4Swizzles;
pub use glam;

pub fn vec4_from_vec3(v: glam::Vec3, w: f32) -> glam::Vec4 {
  glam::Vec4::new(v.x, v.y, v.z, w)
}

#[derive(Debug, Copy, Clone)]
pub struct Point {
  pos: glam::Vec3,
}

impl Point {
  pub fn from_vec3(pos: glam::Vec3) -> Self {
    Self { pos }
  }

  pub fn from_vec4(v: glam::Vec4) -> Self {
    Self { pos: v.xyz() }
  }

  pub fn as_vec3(&self) -> glam::Vec3 {
    self.pos
  }

  pub fn as_vec4(&self) -> glam::Vec4 {
    vec4_from_vec3(self.pos, 1.0)
  }

  pub fn transform(&self, transform: glam::Mat4) -> Self {
    Self::from_vec4(transform * self.as_vec4())
  }
}

#[derive(Debug, Copy, Clone)]
pub struct Direction {
  dir: glam::Vec3,
}

impl Direction {
  pub fn from_vec3(dir: glam::Vec3) -> Self {
    Self { dir }
  }

  pub fn from_vec4(dir: glam::Vec4) -> Self {
    Self { dir: dir.xyz() }
  }

  pub fn from_points(p1: Point, p2: Point) -> Self {
    Self { dir: p1.pos - p2.pos }
  }

  pub fn as_vec3(&self) -> glam::Vec3 {
    self.dir
  }

  pub fn as_vec4(&self) -> glam::Vec4 {
    vec4_from_vec3(self.dir, 0.0)
  }

  pub fn transform(&self, transform: glam::Mat4) -> Self {
    Self::from_vec4(transform * self.as_vec4())
  }

  pub fn normalize(&self) -> Self {
    Self::from_vec3(self.as_vec3().normalize())
  }

  pub fn cross(&self, other: Self) -> Self {
    Self::from_vec3(self.dir.cross(other.dir))
  }

  pub fn opposite(&self) -> Self {
    Self::from_vec3(-self.dir)
  }

  pub fn is_zero(&self) -> bool {
    self.dir.length_squared() == 0.0
  }
}

#[derive(Debug, Copy, Clone)]
pub struct Plane {
  dir: Direction,
  point: Point,
}

impl Plane {
  pub fn new(dir: Direction, point: Point) -> Self {
    Self { dir, point }
  }

  pub fn get_plane_eq(&self) -> glam::Vec4 {
    vec4_from_vec3(self.dir.as_vec3(), -self.dir.as_vec3().dot(self.point.as_vec3()))
  }

  pub fn get_direction(&self) -> Direction {
    self.dir
  }

  pub fn get_point(&self) -> Point {
    self.point
  }

  pub fn transform(&self, transform: glam::Mat4) -> Self {
    Self { dir: self.dir.transform(transform), point: self.point.transform(transform) }
  }

  pub fn dist_from_point(&self, point: &Point) -> f32 {
    self.get_plane_eq().dot(point.as_vec4())
  }
}