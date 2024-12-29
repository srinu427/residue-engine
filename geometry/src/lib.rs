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

  pub fn displace(&self, displacement: glam::Vec3) -> Self {
    Self::from_vec3(self.pos + displacement)
  }

  pub fn average_of(points: &[Self]) -> Self {
    if points.len() == 0 {
      return Point::from_vec3(glam::Vec3::ZERO);
    }
    let point_count_f32 = points.len() as f32;
    let mut sum_point = glam::Vec3::ZERO;
    for point in points {
      sum_point = sum_point + point.as_vec3();
    }
    Point::from_vec3(sum_point / point_count_f32)
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
pub struct LineSegment {
  start: glam::Vec3,
  end: glam::Vec3,
}

impl LineSegment {
  pub fn from_vec3s(start: glam::Vec3, end: glam::Vec3) -> Self {
    Self { start, end }
  }

  pub fn from_points(start: Point, end: Point) -> Self {
    Self { start: start.as_vec3(), end: end.as_vec3() }
  }

  pub fn get_direction(&self) -> Direction {
    Direction::from_vec3(self.end - self.start)
  }

  pub fn get_start(&self) -> Point {
    Point::from_vec3(self.start)
  }

  pub fn get_end(&self) -> Point {
    Point::from_vec3(self.end)
  }

  pub fn transform(&self, transform: glam::Mat4) -> Self {
    Self::from_points(self.get_start().transform(transform), self.get_end().transform(transform))
  }

  pub fn displace(&self, displacement: glam::Vec3) -> Self {
    Self::from_vec3s(self.start + displacement, self.end + displacement)
  }
}

#[derive(Debug, Copy, Clone)]
pub struct Plane {
  dir: Direction,
  point: Point,
}

impl Plane {
  pub fn new(dir: Direction, point: Point) -> Self {
    Self { dir: dir.normalize(), point }
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

  pub fn displace(&self, displacement: glam::Vec3) -> Self {
    Self::new(self.dir, Point::from_vec3(self.point.as_vec3() + displacement))
  }

  pub fn opposite(&self) -> Self {
    Self {dir: self.dir.opposite(), point: self.point}
  }

  pub fn dist_from_point(&self, point: &Point) -> f32 {
    self.get_plane_eq().dot(point.as_vec4())
  }

  pub fn project_direction(&self, point: &Point) -> Direction {
    let dist = self.dist_from_point(point);
    Direction::from_vec3(-self.dir.as_vec3() * dist)
  }

  pub fn project_point(&self, point: &Point) -> Point {
    Point::from_vec3(point.as_vec3() + self.project_direction(point).as_vec3())
  }
}

#[derive(Debug, Copy, Clone)]
pub struct Orientation {
  pub position: glam::Vec3,
  pub rotation: glam::Mat4,
}
impl Orientation {
  pub fn new(position: glam::Vec3, rotation: glam::Mat4) -> Self {
    Self { position, rotation }
  }

  pub fn relative_to(&self, other: Self) -> Self {
    Self::new(self.position - other.position, self.rotation.transpose() * self.rotation)
  }

  pub fn get_full_transform(&self) -> glam::Mat4 {
    glam::Mat4::from_translation(self.position) * self.rotation
  }

  pub fn inverse(&self) -> Self {
    Self::new(-self.position, self.rotation.transpose())
  }

  pub fn add(&self, other: Self) -> Self {
    Self::new(self.position + other.position, other.rotation * self.rotation)
  }
}