use glam::Vec4Swizzles;

fn vec4_from_vec3(v: glam::Vec3, w: f32) -> glam::Vec4 {
  glam::Vec4::new(v.x, v.y, v.z, w)
}

#[derive(Debug, Clone, Copy)]
pub enum MinDistFromPlane {
  NoIntersect(f32),
  Intersect
}

#[derive(Debug, Clone)]
pub struct PolygonMesh {
  vertices: Vec<glam::Vec3>,
  faces: Vec<(glam::Vec4, Vec<usize>)>,
  collision_faces: Vec<glam::Vec4>,
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
    let faces = vec![(vec4_from_vec3(normal, -normal.dot(center)), vec![0, 1, 2, 3]), ];
    let collision_faces = vec![
      vec4_from_vec3(normal, -normal.dot(center)),
      vec4_from_vec3(normal.cross(-n_tangent), -normal.cross(-n_tangent).dot(vertices[0])),
      vec4_from_vec3(normal.cross(-n_bitangent), -normal.cross(-n_bitangent).dot(vertices[1])),
      vec4_from_vec3(normal.cross(n_tangent), -normal.cross(n_tangent).dot(vertices[2])),
      vec4_from_vec3(normal.cross(n_bitangent), -normal.cross(n_bitangent).dot(vertices[3])),
    ];
    let edges = vec![
      (0, 1),
      (1, 2),
      (2, 3),
      (3, 0),
    ];
    Self{vertices, faces, collision_faces, edges}
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
    let collision_faces = faces.iter().map(|face| face.0).collect();
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
      (4, 7),
      // Sides
      (4, 0),
      (5, 1),
      (6, 2),
      (7, 3),
    ];
    Self{vertices, faces, collision_faces, edges}
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

  pub fn get_min_dist_from_plane(
    &self,
    transform: glam::Mat4,
    plane: glam::Vec4
  ) -> MinDistFromPlane {
    let mut pos_side = MinDistFromPlane::Intersect;
    for vertex in self.vertices.iter() {
      let vertex_vec4 = vec4_from_vec3(*vertex, 1.0);
      let dist_from_plane = plane.dot(transform * vertex_vec4);
      match pos_side {
        MinDistFromPlane::NoIntersect(curr_min_dist) => {
          if curr_min_dist < 0.0 && dist_from_plane > 0.0 {
            return MinDistFromPlane::Intersect;
          }
          if curr_min_dist > 0.0 && dist_from_plane < 0.0 {
            return MinDistFromPlane::Intersect;
          }
          if dist_from_plane.abs() < curr_min_dist.abs() {
            pos_side = MinDistFromPlane::NoIntersect(dist_from_plane);
          } else {
            pos_side = MinDistFromPlane::NoIntersect(curr_min_dist);
          }
        }
        MinDistFromPlane::Intersect => {
          pos_side = MinDistFromPlane::NoIntersect(dist_from_plane);
        }
      }
    }
    pos_side
  }

  pub fn get_distance_inside_plane(&self, transform: glam::Mat4, plane: glam::Vec4) -> f32 {
    let max_dist_inside = self
      .vertices
      .iter()
      .map(|v| { plane.dot(transform * vec4_from_vec3(*v, 1.0)) })
      .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    f32::min(max_dist_inside.unwrap_or(0.0), 0.0)
  }

  pub fn will_plane_separate(
    plane: glam::Vec4,
    poly_mesh_1: &PolygonMesh,
    transform_1: glam::Mat4,
    poly_mesh_2: &PolygonMesh,
    transform_2: glam::Mat4,
  ) -> bool {
    let MinDistFromPlane::NoIntersect(on_pos_side_1) =
      poly_mesh_1.get_min_dist_from_plane(transform_1, plane) else {
      return false
    };
    let MinDistFromPlane::NoIntersect(on_pos_side_2) =
      poly_mesh_2.get_min_dist_from_plane(transform_2, plane) else {
      return false
    };
    (on_pos_side_1 > 0.0) ^ (on_pos_side_2 > 0.0)
  }

  pub fn is_separation_plane_valid(
    separation_plane: SeparationType,
    poly_mesh_1: &PolygonMesh,
    transform_1: glam::Mat4,
    poly_mesh_2: &PolygonMesh,
    transform_2: glam::Mat4,
  ) -> bool {
    match separation_plane {
      SeparationType::FirstObjectFace { idx } => {
        let plane = transform_1 * poly_mesh_1.collision_faces[idx];
        Self::will_plane_separate(plane, poly_mesh_1, transform_1, poly_mesh_2, transform_2)
      }
      SeparationType::SecondObjectFace { idx } => {
        let plane = transform_2 * poly_mesh_2.collision_faces[idx];
        Self::will_plane_separate(plane, poly_mesh_1, transform_1, poly_mesh_2, transform_2)
      }
      SeparationType::EdgeCross { first_idx, second_idx } => {
        let edge_ids_1 = poly_mesh_1.edges[first_idx];
        let edge_ids_2 = poly_mesh_2.edges[second_idx];
        let edge_1 = poly_mesh_1.vertices[edge_ids_1.1] - poly_mesh_1.vertices[edge_ids_1.0];
        let edge_1 = (transform_1 * vec4_from_vec3(edge_1, 0.0)).xyz();
        let edge_2 = poly_mesh_2.vertices[edge_ids_2.1] - poly_mesh_2.vertices[edge_ids_2.0];
        let edge_2 = (transform_2 * vec4_from_vec3(edge_2, 0.0)).xyz();
        let edge_cross = edge_1.cross(edge_2);
        if edge_cross.length_squared() == 0.0 {
          return false;
        }
        let edge_cross = edge_cross.normalize();
        let edge_cross = vec4_from_vec3(
          edge_cross,
          -edge_cross.dot(
            (transform_1 * vec4_from_vec3(poly_mesh_1.vertices[edge_ids_1.0], 1.0)).xyz()
          )
        );

        Self::will_plane_separate(edge_cross, poly_mesh_1, transform_1, poly_mesh_2, transform_2)
      }
    }
  }

  pub fn get_separation_plane(
    &self,
    self_transform: glam::Mat4,
    other: &Self,
    other_transform: glam::Mat4
  ) -> Option<SeparationType> {
    // Check self's faces
    for (i, face) in self.collision_faces.iter().enumerate() {
      let MinDistFromPlane::NoIntersect(is_on_pos_side) =
        other.get_min_dist_from_plane(other_transform, self_transform * *face) else {
        continue
      };
      if is_on_pos_side > 0.0 {
        return Some(SeparationType::FirstObjectFace {idx: i});
      }
    }

    // Check other's faces
    for (i, face) in other.collision_faces.iter().enumerate() {
      let MinDistFromPlane::NoIntersect(is_on_pos_side) =
        self.get_min_dist_from_plane(self_transform, other_transform * *face) else {
        continue
      };
      if is_on_pos_side > 0.0 {
        return Some(SeparationType::SecondObjectFace {idx: i});
      }
    }

    // Check Edge crosses
    for (i, edge_ids_self) in self.edges.iter().enumerate() {
      for (j, edge_ids_other) in other.edges.iter().enumerate() {
        let edge_self = self.vertices[edge_ids_self.1] - self.vertices[edge_ids_self.0];
        let edge_self = (self_transform * vec4_from_vec3(edge_self, 0.0)).xyz();
        let edge_other = other.vertices[edge_ids_other.1] - other.vertices[edge_ids_other.0];
        let edge_other = (other_transform * vec4_from_vec3(edge_other, 0.0)).xyz();
        let edge_cross = edge_self.cross(edge_other);
        if edge_cross.length_squared() == 0.0 {
          continue;
        }
        let edge_cross = edge_cross.cross(edge_other).normalize();
        let edge_cross_plane = vec4_from_vec3(
          edge_cross,
          -edge_cross.dot(
            (self_transform * vec4_from_vec3(self.vertices[edge_ids_self.0], 1.0)).xyz()
          )
        );
        if Self::will_plane_separate(
          edge_cross_plane,
          &self,
          self_transform,
          other,
          other_transform,
        ) {
          return Some(SeparationType::EdgeCross {first_idx: i, second_idx: j});
        }
      }
    }
    None
  }
}

#[derive(Debug, Clone, Copy)]
pub enum SeparationType {
  FirstObjectFace{ idx: usize },
  SecondObjectFace{ idx: usize },
  EdgeCross{ first_idx: usize, second_idx: usize },
}

impl SeparationType {
  pub fn decode_to_vec4(
    &self,
    poly_mesh_1: &PolygonMesh,
    transform_1: glam::Mat4,
    poly_mesh_2: &PolygonMesh,
    transform_2: glam::Mat4,
  ) -> Option<glam::Vec4> {
    match self {
      SeparationType::FirstObjectFace { idx } => {
        let plane = transform_1 * poly_mesh_1.collision_faces[*idx];
        Some(plane)
      }
      SeparationType::SecondObjectFace { idx } => {
        let plane = transform_2 * poly_mesh_2.collision_faces[*idx];
        Some(plane)
      }
      SeparationType::EdgeCross { first_idx, second_idx } => {
        let edge_ids_1 = poly_mesh_1.edges[*first_idx];
        let edge_ids_2 = poly_mesh_2.edges[*second_idx];
        let edge_1 = poly_mesh_1.vertices[edge_ids_1.1] - poly_mesh_1.vertices[edge_ids_1.0];
        let edge_1 = (transform_1 * vec4_from_vec3(edge_1, 0.0)).xyz();
        let edge_2 = poly_mesh_2.vertices[edge_ids_2.1] - poly_mesh_2.vertices[edge_ids_2.0];
        let edge_2 = (transform_2 * vec4_from_vec3(edge_2, 0.0)).xyz();
        let edge_cross = edge_1.cross(edge_2);
        if edge_cross.length_squared() == 0.0 {
          return None;
        }
        let edge_cross = edge_cross.normalize();
        let edge_cross = vec4_from_vec3(
          edge_cross,
          -edge_cross.dot(
            (transform_1 * vec4_from_vec3(poly_mesh_1.vertices[edge_ids_1.0], 1.0)).xyz()
          )
        );
        Some(edge_cross)
      }
    }
  }
}

#[derive(Debug, Clone, Copy)]
pub enum Separation {
  No(SeparationType),
  Yes(SeparationType),
}