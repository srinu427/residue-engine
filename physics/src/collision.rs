use glam::Vec4Swizzles;
use geometry::{Direction, Plane, Point};

fn vec4_from_vec3(v: glam::Vec3, w: f32) -> glam::Vec4 {
  glam::Vec4::new(v.x, v.y, v.z, w)
}

#[derive(Debug, Clone, Copy)]
pub enum SideOfPlane {
  Positive(f32),
  Negative(f32),
  OnPlane,
  Intersect
}

impl SideOfPlane {
  pub fn on_opposite_sides(o1: SideOfPlane, o2: SideOfPlane) -> bool {
    match o1 {
      SideOfPlane::Positive(_) => {
        match o2 {
          SideOfPlane::Positive(_) => false,
          SideOfPlane::Negative(_) => true,
          SideOfPlane::OnPlane => false,
          SideOfPlane::Intersect => false,
        }
      }
      SideOfPlane::Negative(_) => {
        match o2 {
          SideOfPlane::Positive(_) => true,
          SideOfPlane::Negative(_) => false,
          SideOfPlane::OnPlane => false,
          SideOfPlane::Intersect => false,
        }
      }
      SideOfPlane::OnPlane => false,
      SideOfPlane::Intersect => false,
    }
  }
}

#[derive(Debug, Clone)]
pub struct PolygonMesh {
  vertices: Vec<Point>,
  faces: Vec<(Plane, Vec<usize>)>,
  collision_faces: Vec<Plane>,
  edges: Vec<(usize, usize)>,
}

impl PolygonMesh {
  pub fn new_rectangle(center: Point, tangent: Direction, bitangent: Direction) -> Self {
    let n_tangent = tangent.normalize();
    let n_bitangent = bitangent.normalize();
    let normal = n_bitangent.cross(n_tangent).normalize();

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
    let faces = vec![(Plane::new(normal, center), vec![0, 1, 2, 3])];
    let collision_faces = vec![
      Plane::new(normal, center),
      Plane::new(normal.cross(n_tangent.opposite()), vertices[0]),
      Plane::new(normal.cross(n_bitangent.opposite()), vertices[1]),
      Plane::new(normal.cross(n_tangent), vertices[2]),
      Plane::new(normal.cross(n_bitangent), vertices[3]),
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
    center: Point,
    tangent: Direction,
    bitangent: Direction,
    depth: f32
  ) -> Self {
    let n_tangent = tangent.normalize();
    let n_bitangent = bitangent.normalize();
    let normal = n_bitangent.cross(n_tangent).normalize();

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
      (Plane::new(n_tangent, Point::from_vec3(center.as_vec3() + h_tangent)), vec![0, 3, 5, 4]),
      (Plane::new(n_tangent.opposite(), Point::from_vec3(center.as_vec3() - h_tangent)), vec![2, 1, 7, 6]),
      (Plane::new(n_bitangent, Point::from_vec3(center.as_vec3() + h_bitangent)), vec![0, 1, 7, 4]),
      (Plane::new(n_bitangent.opposite(), Point::from_vec3(center.as_vec3() - h_bitangent)), vec![3, 2, 6, 5]),
      (Plane::new(normal, Point::from_vec3(center.as_vec3() + h_depth)), vec![0, 1, 2, 3]),
      (Plane::new(normal.opposite(), Point::from_vec3(center.as_vec3() - h_depth)), vec![4, 5, 6, 7]),
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

  pub fn get_faces(&self) -> Vec<Vec<Point>> {
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
    plane: Plane
  ) -> SideOfPlane {
    let mut side_of_plane = SideOfPlane::Intersect;
    for vertex in self.vertices.iter() {
      let transformed_vert = vertex.transform(transform);
      let dist_from_plane = plane.dist_from_point(&transformed_vert);
      match side_of_plane {
        SideOfPlane::Positive(curr_min_dist) => {
          if dist_from_plane < 0.0 {
            return SideOfPlane::Intersect;
          }
          if dist_from_plane.abs() < curr_min_dist.abs() {
            side_of_plane = SideOfPlane::Positive(dist_from_plane);
          } else {
            side_of_plane = SideOfPlane::Positive(curr_min_dist);
          }
        }
        SideOfPlane::Negative(curr_min_dist) => {
          if dist_from_plane > 0.0 {
            return SideOfPlane::Intersect;
          }
          if dist_from_plane.abs() < curr_min_dist.abs() {
            side_of_plane = SideOfPlane::Negative(dist_from_plane);
          } else {
            side_of_plane = SideOfPlane::Negative(curr_min_dist);
          }
        }
        SideOfPlane::OnPlane => {
          if dist_from_plane < 0.0 {
            side_of_plane = SideOfPlane::Negative(0.0);
          }
          if dist_from_plane > 0.0 {
            side_of_plane = SideOfPlane::Positive(0.0);
          }
        }
        SideOfPlane::Intersect => {
          if dist_from_plane < 0.0 {
            side_of_plane = SideOfPlane::Negative(dist_from_plane);
          }
          if dist_from_plane > 0.0 {
            side_of_plane = SideOfPlane::Positive(dist_from_plane);
          }
          if dist_from_plane == 0.0 {
            side_of_plane = SideOfPlane::OnPlane;
          }
        }
      }
    }
    side_of_plane
  }

  pub fn get_distance_inside_plane(&self, transform: glam::Mat4, plane: Plane) -> f32 {
    let max_dist_inside = self
      .vertices
      .iter()
      .map(|v| { plane.dist_from_point(&v.transform(transform)) })
      .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    f32::min(max_dist_inside.unwrap_or(0.0), 0.0)
  }

  pub fn will_plane_separate(
    plane: Plane,
    poly_mesh_1: &PolygonMesh,
    transform_1: glam::Mat4,
    poly_mesh_2: &PolygonMesh,
    transform_2: glam::Mat4,
  ) -> bool {
    let side_of_plane_1 = poly_mesh_1.get_min_dist_from_plane(transform_1, plane);
    let side_of_plane_2 = poly_mesh_2.get_min_dist_from_plane(transform_2, plane);
    SideOfPlane::on_opposite_sides(side_of_plane_1, side_of_plane_2)
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
        let plane = poly_mesh_1.collision_faces[idx].transform(transform_1);
        Self::will_plane_separate(plane, poly_mesh_1, transform_1, poly_mesh_2, transform_2)
      }
      SeparationType::SecondObjectFace { idx } => {
        let plane = poly_mesh_2.collision_faces[idx].transform(transform_2);
        Self::will_plane_separate(plane, poly_mesh_1, transform_1, poly_mesh_2, transform_2)
      }
      SeparationType::EdgeCross { first_idx, second_idx } => {
        let edge_ids_1 = poly_mesh_1.edges[first_idx];
        let edge_ids_2 = poly_mesh_2.edges[second_idx];
        let edge_1 = Direction::from_points(
          poly_mesh_1.vertices[edge_ids_1.1],
          poly_mesh_1.vertices[edge_ids_1.0]
        );
        let edge_1 = edge_1.transform(transform_1);
        let edge_2 = Direction::from_points(
          poly_mesh_2.vertices[edge_ids_2.1],
          poly_mesh_2.vertices[edge_ids_2.0]
        );
        let edge_2 = edge_2.transform(transform_2);
        let edge_cross = edge_1.cross(edge_2);
        if edge_cross.as_vec3().length_squared() == 0.0 {
          return false;
        }
        let edge_cross = edge_cross.normalize();
        let edge_cross = Plane::new(
          edge_cross,
          poly_mesh_1.vertices[edge_ids_1.0].transform(transform_1)
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
      let SideOfPlane::Positive(is_on_pos_side) =
        other.get_min_dist_from_plane(other_transform, face.transform(self_transform)) else {
        continue
      };
      return Some(SeparationType::FirstObjectFace {idx: i});
    }

    // Check other's faces
    for (i, face) in other.collision_faces.iter().enumerate() {
      let SideOfPlane::Positive(is_on_pos_side) =
        self.get_min_dist_from_plane(self_transform, face.transform(other_transform)) else {
        continue
      };
      return Some(SeparationType::SecondObjectFace {idx: i});
    }

    // Check Edge crosses
    for (i, edge_ids_self) in self.edges.iter().enumerate() {
      for (j, edge_ids_other) in other.edges.iter().enumerate() {
        let edge_self = Direction::from_points(
          self.vertices[edge_ids_self.1],
          self.vertices[edge_ids_self.0]
        ).transform(self_transform);
        let edge_other = Direction::from_points(
          other.vertices[edge_ids_other.1],
          other.vertices[edge_ids_other.0]
        ).transform(other_transform);
        let edge_cross = edge_self.cross(edge_other);
        if edge_cross.is_zero() {
          continue;
        }
        let edge_cross = edge_cross.normalize();
        let edge_cross_plane = Plane::new(
          edge_cross,
          self.vertices[edge_ids_self.0].transform(self_transform),
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
  ) -> Option<Plane> {
    match self {
      SeparationType::FirstObjectFace { idx } => {
        Some(poly_mesh_1.collision_faces[*idx].transform(transform_1))
      }
      SeparationType::SecondObjectFace { idx } => {
        Some(poly_mesh_2.collision_faces[*idx].transform(transform_2))
      }
      SeparationType::EdgeCross { first_idx, second_idx } => {
        let edge_ids_1 = poly_mesh_1.edges[*first_idx];
        let edge_ids_2 = poly_mesh_2.edges[*second_idx];
        let edge_1 = Direction::from_points(
          poly_mesh_1.vertices[edge_ids_1.1],
          poly_mesh_1.vertices[edge_ids_1.0]
        ).transform(transform_1);
        let edge_2 = Direction::from_points(
          poly_mesh_2.vertices[edge_ids_2.1],
          poly_mesh_2.vertices[edge_ids_2.0]
        ).transform(transform_2);
        let edge_cross = edge_1.cross(edge_2);
        if edge_cross.is_zero() {
          return None;
        }
        let edge_cross = edge_cross.normalize();
        let edge_cross = Plane::new(
          edge_cross,
          poly_mesh_1.vertices[edge_ids_1.0].transform(transform_1),
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