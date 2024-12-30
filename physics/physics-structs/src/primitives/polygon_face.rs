use geometry::{glam, Direction, LineSegment, Orientation, Plane, Point};

#[derive(Debug, Clone)]
pub struct PolygonFace {
  verts: Vec<Point>,
  face: Plane,
  edges: Vec<LineSegment>,
  bound_planes: Vec<Plane>,
}

impl PolygonFace {
  pub fn new(verts: Vec<Point>) -> PolygonFace {
    let edges = (0..verts.len() - 1)
      .map(|i| LineSegment::from_points(verts[i], verts[i + 1]))
      .collect::<Vec<_>>();
    let normal = edges[0].get_direction().cross(edges[1].get_direction());
    let face = Plane::new(normal, verts[0]);
    let bound_planes = edges
      .iter()
      .map(|e| {
        let bound_normal = normal.cross(e.get_direction());
        Plane::new(bound_normal, e.get_start())
      })
      .collect::<Vec<_>>();
    PolygonFace { verts, face, edges, bound_planes }
  }

  pub fn get_verts(&self) -> &Vec<Point> {
    &self.verts
  }

  pub fn get_face(&self) -> Plane {
    self.face
  }

  pub fn get_bound_planes(&self) -> &Vec<Plane> {
    &self.bound_planes
  }

  pub fn get_edges(&self) -> &Vec<LineSegment> {
    &self.edges
  }

  pub fn transformed(&self, transform: glam::Mat4) -> Self {
    let t_verts = self.verts.iter().map(|v| v.transform(transform)).collect::<Vec<_>>();
    let t_face = self.face.transform(transform);
    let t_edges = self.edges.iter().map(|e| e.transform(transform)).collect::<Vec<_>>();
    let t_bound_planes =
      self.bound_planes.iter().map(|e| e.transform(transform)).collect::<Vec<_>>();

    Self { verts: t_verts, face: t_face, edges: t_edges, bound_planes: t_bound_planes }
  }

  pub fn oriented(&self, orientation: Orientation) -> Self {
    self.transformed(orientation.get_full_transform())
  }

  pub fn new_rectangle(center: Point, tangent: Direction, bitangent: Direction) -> Self {
    let h_tangent = tangent.as_vec3() / 2.0;
    let h_bitangent = bitangent.as_vec3() / 2.0;

    let vertices = [
      center.as_vec3() + h_tangent + h_bitangent,
      center.as_vec3() - h_tangent + h_bitangent,
      center.as_vec3() - h_tangent - h_bitangent,
      center.as_vec3() + h_tangent - h_bitangent,
    ]
    .iter()
    .map(|x| Point::from_vec3(*x))
    .collect::<Vec<_>>();
    PolygonFace::new(vertices)
  }

  pub fn new_cuboid(
    center: Point,
    tangent: Direction,
    bitangent: Direction,
    depth: f32,
  ) -> Vec<Self> {
    let n_tangent = tangent.normalize();
    let n_bitangent = bitangent.normalize();
    let normal = n_tangent.cross(n_bitangent).normalize();

    let h_tangent = tangent.as_vec3() / 2.0;
    let h_bitangent = bitangent.as_vec3() / 2.0;
    let h_depth = normal.as_vec3() * depth / 2.0;

    let vertices = [
      // Top Face
      center.as_vec3() + h_tangent + h_bitangent + h_depth,
      center.as_vec3() - h_tangent + h_bitangent + h_depth,
      center.as_vec3() - h_tangent - h_bitangent + h_depth,
      center.as_vec3() + h_tangent - h_bitangent + h_depth,
      // Bottom Face
      center.as_vec3() + h_tangent + h_bitangent - h_depth,
      center.as_vec3() + h_tangent - h_bitangent - h_depth,
      center.as_vec3() - h_tangent - h_bitangent - h_depth,
      center.as_vec3() - h_tangent + h_bitangent - h_depth,
    ]
    .iter()
    .map(|x| Point::from_vec3(*x))
    .collect::<Vec<_>>();
    let faces = vec![
      PolygonFace::new(vec![vertices[0], vertices[1], vertices[2], vertices[3]]),
      PolygonFace::new(vec![vertices[4], vertices[5], vertices[6], vertices[7]]),
      PolygonFace::new(vec![vertices[0], vertices[3], vertices[5], vertices[4]]),
      PolygonFace::new(vec![vertices[2], vertices[1], vertices[7], vertices[6]]),
      PolygonFace::new(vec![vertices[0], vertices[4], vertices[7], vertices[1]]),
      PolygonFace::new(vec![vertices[3], vertices[2], vertices[6], vertices[5]]),
    ];
    faces
  }
}
