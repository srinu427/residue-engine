fn vec4_from_vec3(v: glam::Vec3, w: f32) -> glam::Vec4 {
  glam::Vec4::new(v.x, v.y, v.z, w)
}

pub struct PolygonMesh {
  vertices: Vec<glam::Vec3>,
  faces: Vec<(glam::Vec4, Vec<usize>)>,
  edges: Vec<(usize, usize)>,
}

impl PolygonMesh {
  pub fn new_rectangle(center: glam::Vec3, tangent: glam::Vec3, bitangent: glam::Vec3) -> Self {
    let n_tangent = tangent.normalize();
    let n_bitangent = bitangent.normalize();
    let normal = n_bitangent.cross(n_tangent).normalize();

    let h_tangent = tangent / 2.0;
    let h_bitangent = bitangent / 2.0;

    let vertices = vec![
      center + h_tangent + h_bitangent,
      center - h_tangent + h_bitangent,
      center - h_tangent - h_bitangent,
      center + h_tangent - h_bitangent,
    ];
    let faces = vec![(vec4_from_vec3(normal, -normal.dot(center)), vec![0, 1, 2, 3])];
    let edges = vec![
      (0, 1),
      (1, 2),
      (2, 3),
      (3, 0),
    ];
    Self{vertices, faces, edges}
  }

  pub fn new_cuboid(
    center: glam::Vec3,
    tangent: glam::Vec3,
    bitangent: glam::Vec3,
    depth: f32
  ) -> Self {
    let n_tangent = tangent.normalize();
    let n_bitangent = bitangent.normalize();
    let normal = n_bitangent.cross(n_tangent).normalize();

    let h_tangent = tangent / 2.0;
    let h_bitangent = bitangent / 2.0;
    let h_depth = normal * depth / 2.0;

    let vertices = vec![
      // Top Face
      center + h_tangent + h_bitangent + h_depth,
      center - h_tangent + h_bitangent + h_depth,
      center - h_tangent - h_bitangent + h_depth,
      center + h_tangent - h_bitangent + h_depth,
      // Bottom Face
      center + h_tangent + h_bitangent - h_depth,
      center + h_tangent - h_bitangent - h_depth,
      center - h_tangent - h_bitangent - h_depth,
      center - h_tangent + h_bitangent - h_depth,
    ];
    let faces = vec![
      (vec4_from_vec3(n_tangent, -n_tangent.dot(center + h_tangent)), vec![0, 3, 5, 4]),
      (vec4_from_vec3(-n_tangent, n_tangent.dot(center - h_tangent)), vec![2, 1, 7, 6]),
      (vec4_from_vec3(n_bitangent, -n_bitangent.dot(center + h_bitangent)), vec![0, 1, 7, 4]),
      (vec4_from_vec3(-n_bitangent, n_bitangent.dot(center - h_bitangent)), vec![3, 2, 6, 5]),
      (vec4_from_vec3(normal, -normal.dot(center + h_depth)), vec![0, 1, 2, 3]),
      (vec4_from_vec3(-normal, normal.dot(center - h_depth)), vec![4, 5, 6, 7]),
    ];
    let edges = vec![
      // Top Face
      (0, 1),
      (1, 2),
      (2, 3),
      (3, 0),
      // Bottom Face
      (5, 4),
      (6, 5),
      (7, 6),
      (8, 7),
      // Sides
      (4, 0),
      (5, 1),
      (6, 2),
      (7, 3),
    ];
    Self{vertices, faces, edges}
  }

  pub fn get_faces(&self) -> Vec<Vec<glam::Vec3>> {
    self
      .faces
      .iter()
      .map(|(_, vert_ids)| {
        vert_ids.iter().map(|id| self.vertices[*id]).collect()
      })
      .collect::<Vec<_>>()
  }
}
