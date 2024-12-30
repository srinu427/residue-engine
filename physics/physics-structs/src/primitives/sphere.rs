use geometry::{glam, Orientation, Point};

static INV_ROOT_3: f32 = 0.577350269189625764508;
static ROOT_3: f32 = 1.732050807568877293527;

static REGULAR_TETRAHEDRON_VERTS: [[Point; 3]; 4] = [
  [
    Point::from_vec3(glam::Vec3::new(-1.0, -1.0, -1.0)),
    Point::from_vec3(glam::Vec3::new(1.0, 1.0, -1.0)),
    Point::from_vec3(glam::Vec3::new(-1.0, 1.0, 1.0)),
  ],
  [
    Point::from_vec3(glam::Vec3::new(-1.0, 1.0, 1.0)),
    Point::from_vec3(glam::Vec3::new(1.0, 1.0, -1.0)),
    Point::from_vec3(glam::Vec3::new(1.0, -1.0, 1.0)),
  ],
  [
    Point::from_vec3(glam::Vec3::new(1.0, 1.0, -1.0)),
    Point::from_vec3(glam::Vec3::new(-1.0, -1.0, -1.0)),
    Point::from_vec3(glam::Vec3::new(1.0, -1.0, 1.0)),
  ],
  [
    Point::from_vec3(glam::Vec3::new(-1.0, -1.0, -1.0)),
    Point::from_vec3(glam::Vec3::new(-1.0, 1.0, 1.0)),
    Point::from_vec3(glam::Vec3::new(1.0, -1.0, 1.0)),
  ],
];

fn subdivide_sphere_triangles(triangles: Vec<[Point; 3]>) -> Vec<[Point; 3]> {
  let mut new_sphere_triangles = vec![];
  for triangle in triangles {
    let midpoint = triangle[0].as_vec3() + triangle[1].as_vec3() + triangle[2].as_vec3();
    let midpoint = midpoint.normalize() * ROOT_3;
    let midpoint = Point::from_vec3(midpoint);
    let mut new_triangles = Vec::from([
      [triangle[0], triangle[1], midpoint],
      [triangle[1], triangle[2], midpoint],
      [triangle[2], triangle[0], midpoint],
    ]);
    new_sphere_triangles.append(&mut new_triangles);
  }
  new_sphere_triangles
}

#[derive(Debug, Clone)]
pub struct Sphere {
  pub radius: f32,
  pub center: Point,
}

impl Sphere {
  pub fn new(radius: f32, center: Point) -> Self {
    Self { radius, center }
  }

  pub fn to_triangles(&self, subdivision: usize) -> Vec<[Point; 3]> {
    let mut triangles = REGULAR_TETRAHEDRON_VERTS.to_vec();
    for _ in 0..subdivision {
      triangles = subdivide_sphere_triangles(triangles);
    }
    let translation_mat = glam::Mat4::from_translation(self.center.as_vec3());
    let radius_by_root_3 = self.radius * INV_ROOT_3;
    let scale_mat =
      glam::Mat4::from_scale(glam::Vec3::new(radius_by_root_3, radius_by_root_3, radius_by_root_3));
    let transformation_mat = translation_mat * scale_mat;
    for triangle in triangles.iter_mut() {
      *triangle = [
        triangle[0].transform(transformation_mat),
        triangle[1].transform(transformation_mat),
        triangle[2].transform(transformation_mat),
      ];
    }
    triangles
  }

  pub fn oriented(&self, orientation: Orientation) -> Self {
    Self {
      radius: self.radius,
      center: Point::from_vec3(self.center.as_vec3() + orientation.position),
    }
  }
}
