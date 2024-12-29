use geometry::{Orientation, Plane, Point};
use sphere::Sphere;
use polygon_mesh::PolygonFace;

pub mod polygon_mesh;
pub mod sphere;

#[derive(Debug, Clone)]
pub enum RigidBodyType {
  PolygonPlane(PolygonFace),
  Sphere(Sphere),
}

impl RigidBodyType {
  pub fn get_plane_min_max_each_side(
    &self,
    orientation: Orientation,
    plane: Plane,
  ) -> (f32, f32, f32, f32) {
    let mut pos_min = f32::MAX;
    let mut pos_max = 0.0;
    let mut neg_min = 0.0;
    let mut neg_max = f32::MIN;

    let orientation_transform = orientation.get_full_transform();

    let points_to_test = match self {
      RigidBodyType::PolygonPlane(p_face) => {
        p_face.get_verts().iter().map(|x| x.transform(orientation_transform)).collect::<Vec<_>>()
      }
      RigidBodyType::Sphere(sphere) => {
        let transformed_center = sphere.center.transform(orientation_transform);
        vec![
          Point::from_vec3(
            transformed_center.as_vec3() + plane.get_direction().as_vec3() * sphere.radius,
          ),
          Point::from_vec3(
            transformed_center.as_vec3() - plane.get_direction().as_vec3() * sphere.radius,
          ),
        ]
      }
    };

    for vert in points_to_test {
      let transformed_vert = vert.transform(orientation_transform);
      let dist_from_plane = plane.dist_from_point(&transformed_vert);
      if dist_from_plane == 0.0 {
        if dist_from_plane < pos_min {
          pos_min = dist_from_plane;
        }
        if dist_from_plane > pos_max {
          pos_max = dist_from_plane;
        }
        if dist_from_plane < neg_min {
          neg_min = dist_from_plane;
        }
        if dist_from_plane > neg_max {
          neg_max = dist_from_plane;
        }
      } else if dist_from_plane > 0.0 {
        if dist_from_plane < pos_min {
          pos_min = dist_from_plane;
        }
        if dist_from_plane > pos_max {
          pos_max = dist_from_plane;
        }
      } else {
        if dist_from_plane < neg_min {
          neg_min = dist_from_plane;
        }
        if dist_from_plane > neg_max {
          neg_max = dist_from_plane;
        }
      }
    }

    (pos_min, pos_max, neg_min, neg_max)
  }
}
