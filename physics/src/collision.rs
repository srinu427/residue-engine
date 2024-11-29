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


  pub fn on_positive_side_of_plane(&self, plane: glam::Vec4) -> Option<bool> {
    let mut pos_side = None;
    for vertex in self.vertices.iter() {
      let dist_from_plane = plane.dot(vec4_from_vec3(*vertex, 1.0));
      if dist_from_plane > 0.0 {
        match pos_side {
          None => pos_side = Some(true),
          Some(is_pos) => if !is_pos {
            pos_side = None;
            break;
          },
        }
      }
      if dist_from_plane < 0.0 {
        match pos_side {
          None => pos_side = Some(false),
          Some(is_pos) => if is_pos {
            pos_side = None;
            break;
          },
        }
      }
    }
    None
  }

  pub fn get_separation_plane(&self, other: &Self) -> SeparationPlane {
    // Check self's faces
    for (i, face) in self.faces.iter().enumerate() {
      match other.on_positive_side_of_plane(face.0) {
        None => continue,
        Some(is_on_pos_side) => {
          if is_on_pos_side {
            return SeparationPlane::FirstObjectFace {idx: i};
          }
        }
      }
    }

    // Check other's faces
    for (i, face) in other.faces.iter().enumerate() {
      match self.on_positive_side_of_plane(face.0) {
        None => continue,
        Some(is_on_pos_side) => {
          if is_on_pos_side {
            return SeparationPlane::SecondObjectFace {idx: i};
          }
        }
      }
    }

    // Check Edge crosses
    for (i, edge_ids_self) in self.edges.iter().enumerate() {
      for (j, edge_ids_other) in other.edges.iter().enumerate() {
        let edge_self = self.vertices[edge_ids_self.1] - self.vertices[edge_ids_self.0];
        let edge_other = other.vertices[edge_ids_other.1] - other.vertices[edge_ids_other.0];
        let edge_cross = edge_self.cross(edge_other);
        if edge_cross.length_squared() == 0.0 {
          continue;
        }
        let edge_cross = edge_cross.cross(edge_other).normalize();
        let edge_cross_plane = vec4_from_vec3(edge_cross, -edge_cross.dot(self.vertices[edge_ids_self.0]));
        let Some(self_side) = self.on_positive_side_of_plane(edge_cross_plane) else { continue };
        let Some(other_side) = other.on_positive_side_of_plane(edge_cross_plane) else { continue };
        if self_side ^ other_side {
          return SeparationPlane::EdgeCross {first_idx: i, second_idx: j};
        }
      }
    }
    SeparationPlane::None
  }
}

pub enum SeparationPlane {
  None,
  FirstObjectFace{ idx: usize },
  SecondObjectFace{ idx: usize },
  EdgeCross{ first_idx: usize, second_idx: usize },
}
