use force::SingleBodyForce;
use geometry::{glam, Direction, LineSegment, Orientation, Plane, Point};
use std::collections::HashMap;
use structs::RigidBodyType;

mod force;
pub mod structs;



#[derive(Debug, Copy, Clone)]
pub struct RigidBodyInfo {
  mass: Mass,
  velocity: glam::Vec3,
  acceleration: glam::Vec3,
  moment_of_inertia: MomentOfInertia,
  angular_velocity: glam::Vec3,
  angular_acceleration: glam::Vec3,
  orientation: Orientation,
}

impl Default for RigidBodyInfo {
  fn default() -> Self {
    Self {
      mass: Mass::Infinite,
      velocity: glam::Vec3::ZERO,
      acceleration: glam::Vec3::ZERO,
      moment_of_inertia: MomentOfInertia::Infinite,
      angular_velocity: glam::Vec3::ZERO,
      angular_acceleration: glam::Vec3::ZERO,
      orientation: Orientation::new(glam::Vec3::ZERO, glam::Mat4::IDENTITY),
    }
  }
}

impl RigidBodyInfo {
  pub fn apply_bounds(&mut self, bounds: Vec<Direction>) {
    for bound in bounds.iter() {
      if self.velocity.dot(bound.as_vec3()) > 0.0 {
        self.velocity = self.velocity.reject_from(bound.as_vec3());
      }
      if self.acceleration.dot(bound.as_vec3()) > 0.0 {
        self.acceleration = self.acceleration.reject_from(bound.as_vec3());
      }
    }
  }

  pub fn update(&mut self, time_s: f32, bounds: Vec<Direction>) {
    self.apply_bounds(bounds);

    let translation = (self.velocity * time_s) + (0.5 * self.acceleration * time_s * time_s);
    let rotation =
      (self.angular_velocity * time_s) + (0.5 * self.angular_acceleration * time_s * time_s);
    let rotation_transform = if rotation.length_squared() != 0.0 {
      glam::Mat4::from_axis_angle(rotation.normalize(), rotation.length())
    } else {
      glam::Mat4::IDENTITY
    };

    self.orientation.position += translation;
    self.velocity += self.acceleration * time_s;

    self.orientation.rotation = rotation_transform * self.orientation.rotation;
    self.angular_velocity += self.angular_acceleration * time_s;
  }
}

#[derive(Debug, Clone)]
pub enum CollisionInfo {
  FutureCollision {
    time: f32,
    plane: Plane,
    bounds: Vec<Plane>,
    disp_1: glam::Vec3,
    disp_2: glam::Vec3
  },
  FutureSliding {
    time: f32,
    plane: Plane,
    bounds: Vec<Plane>,
    disp_1: glam::Vec3,
    disp_2: glam::Vec3
  },
  NoCollision
}

#[derive(Debug, Clone)]
pub struct RigidBody {
  name: String,
  mesh: Vec<RigidBodyType>,
  physics_info: RigidBodyInfo,
  collision_mask: u32,
  body_forces: Vec<SingleBodyForce>,
}

pub struct PhysicsEngine {
  rigid_bodies: Vec<RigidBody>,
  rigid_body_names: HashMap<String, usize>,
  coupling_forces: HashMap<(String, String), SingleBodyForce>,
}

impl PhysicsEngine {
  fn solve_const_acc(d: f32, u: f32, a: f32) -> Vec<f32> {
    let mut roots = Vec::with_capacity(2);

    if a == 0.0 {
      if u < 0.0 {
        roots.push(-d / u)
      }
    } else {
      let det = (u * u) - (2.0 * a * d);
      if det >= 0.0 {
        let sqrt_det = det.sqrt();
        let root_1 = (-u - sqrt_det) / a;
        let root_2 = (-u + sqrt_det) / a;
        if root_1 >= 0.0 {
          roots.push(root_1);
        }
        if root_2 >= 0.0 {
          roots.push(root_2);
        }
      }
    }

    roots
  }

  pub fn plane_slip_time(
    point: Point,
    point_vel: glam::Vec3,
    point_acc: glam::Vec3,
    plane: Plane,
    bounds: &[Plane],
    plane_vel: glam::Vec3,
    plane_acc: glam::Vec3,
  ) -> Option<f32> {
    let coll_times = bounds
      .iter()
      .filter_map(
        |x| Self::plane_point_coll_time(point, point_vel, point_acc, *x, &[], plane_vel, plane_acc)
      )
      .map(|x| x.0)
      .collect::<Vec<f32>>();
    coll_times.iter().reduce(f32::max).copied()
  }

  pub fn plane_point_coll_time(
    point: Point,
    point_vel: glam::Vec3,
    point_acc: glam::Vec3,
    plane: Plane,
    bounds: &[Plane],
    plane_vel: glam::Vec3,
    plane_acc: glam::Vec3,
  ) -> Option<(f32, glam::Vec3, glam::Vec3)> {
    let point_rel_vel = point_vel - plane_vel;
    let point_rel_acc = point_acc - plane_acc;
    let vert_dist = plane.dist_from_point(&point);
    if vert_dist >= 0.0 {
      let coll_times = Self::solve_const_acc(
        vert_dist,
        point_rel_vel.dot(plane.get_direction().as_vec3()),
        point_rel_acc.dot(plane.get_direction().as_vec3()),
      );

      let Some(time_s) = coll_times else { return None };
      let point_displacement = point_vel * time_s + 0.5 * point_acc * time_s * time_s;
      let plane_displacement = plane_vel * time_s + 0.5 * plane_acc * time_s * time_s;
      let vert_fwd = point.displace(point_displacement);
      for bound_face in bounds.iter() {
        let bound_face_moved = bound_face.displace(plane_displacement);
        let bound_face_dist = bound_face_moved.dist_from_point(&vert_fwd);
        if bound_face_dist < 0.0 {
          return None;
        }
      }
      return Some((time_s, point_displacement, plane_displacement));
    }
    None
  }

  pub fn line_seg_coll_time(
    line_segment_1: LineSegment,
    vel_1: glam::Vec3,
    acc_1: glam::Vec3,
    line_segment_2: LineSegment,
    vel_2: glam::Vec3,
    acc_2: glam::Vec3,
  ) -> Option<(f32, glam::Vec3, glam::Vec3)> {
    let perp_dir = line_segment_1.get_direction().cross(line_segment_2.get_direction());
    if perp_dir.is_zero() {
      return None;
    }
    let perp_plane = Plane::new(perp_dir, line_segment_1.get_start());
    let perp_plane = if perp_plane.dist_from_point(&line_segment_2.get_start()) < 0.0 {
      perp_plane.opposite()
    } else {
      perp_plane
    };
    let coll_time_opt = Self::plane_point_coll_time(
      line_segment_2.get_start(),
      vel_1,
      acc_1,
      perp_plane,
      &[],
      vel_2,
      acc_2,
    );

    let Some((coll_time, displacement_2, displacement_1)) = coll_time_opt else { return None };

    let displaced_ls_1 = line_segment_1.displace(displacement_1);
    let displaced_ls_2 = line_segment_2.displace(displacement_2);

    let bound_plane_1 = Plane::new(
      perp_plane.get_direction().cross(displaced_ls_1.get_direction()),
      displaced_ls_1.get_start(),
    );
    let bound_plane_2 = Plane::new(
      perp_plane.get_direction().cross(displaced_ls_2.get_direction()),
      displaced_ls_2.get_start(),
    );
    let ls_1_start_side = bound_plane_2.dist_from_point(&displaced_ls_1.get_start()) >= 0.0;
    let ls_1_end_side = bound_plane_2.dist_from_point(&displaced_ls_1.get_end()) >= 0.0;
    let ls_2_start_side = bound_plane_1.dist_from_point(&displaced_ls_2.get_start()) >= 0.0;
    let ls_2_end_side = bound_plane_1.dist_from_point(&displaced_ls_2.get_end()) >= 0.0;

    if (ls_1_start_side ^ ls_1_end_side) && (ls_2_start_side ^ ls_2_end_side) {
      Some((coll_time, displacement_1, displacement_2))
    } else {
      None
    }
  }

  pub fn rigid_body_coll_time(body_1: &RigidBody, body_2: &RigidBody) -> (f32, Plane, Point) {
    let mut min_collision_time = f32::MAX;
    let mut collision_plane =
      Plane::new(Direction::from_vec3(glam::Vec3::ZERO), Point::from_vec3(glam::Vec3::ZERO));
    let mut collision_point = Point::from_vec3(glam::Vec3::ZERO);

    let body_1_transform = body_1.physics_info.orientation.get_full_transform();
    let body_2_transform = body_2.physics_info.orientation.get_full_transform();

    if !(body_1.collision_mask & body_2.collision_mask) {
      return (min_collision_time, collision_plane, collision_point);
    }
    for prim_1 in body_1.mesh.iter() {
      for prim_2 in body_2.mesh.iter() {
        match prim_1 {
          RigidBodyType::PolygonPlane(p_mesh_1) => match &prim_2 {
            RigidBodyType::PolygonPlane(p_mesh_2) => {
              let transformed_mesh_1 = p_mesh_1.transformed(body_1_transform);
              let transformed_mesh_2 = p_mesh_2.transformed(body_2_transform);

              for vert_2 in p_mesh_2.get_vertices().iter() {
                let point_coll_time = Self::plane_point_coll_time(
                  *vert_2,
                  body_2.physics_info.velocity,
                  body_2.physics_info.acceleration,
                  transformed_mesh_1.get_face(),
                  &transformed_mesh_1.get_bound_planes(),
                  body_1.physics_info.velocity,
                  body_1.physics_info.acceleration,
                );
                let Some((time_s, point_displacement, plane_displacement)) = point_coll_time else {
                  continue;
                };
                if time_s < min_collision_time {
                  min_collision_time = time_s;
                  collision_plane = transformed_mesh_1.get_face().displace(plane_displacement);
                  collision_point = vert_2.displace(point_displacement);
                }
              }

              for vert_1 in p_mesh_1.get_vertices().iter() {
                let point_coll_time = Self::plane_point_coll_time(
                  *vert_1,
                  body_1.physics_info.velocity,
                  body_1.physics_info.acceleration,
                  transformed_mesh_2.get_face(),
                  &transformed_mesh_2.get_bound_planes(),
                  body_2.physics_info.velocity,
                  body_2.physics_info.acceleration,
                );
                let Some((time_s, point_displacement, plane_displacement)) = point_coll_time else {
                  continue;
                };
                if time_s < min_collision_time {
                  min_collision_time = time_s;
                  collision_plane = transformed_mesh_2.get_face().displace(plane_displacement);
                  collision_point = vert_1.displace(point_displacement);
                }
              }

              for edge_1 in transformed_mesh_1.get_edges() {
                for edge_2 in transformed_mesh_2.get_edges() {
                  let ls_coll_time = Self::line_seg_coll_time(
                    *edge_1,
                    body_1.physics_info.velocity,
                    body_1.physics_info.acceleration,
                    *edge_2,
                    body_2.physics_info.velocity,
                    body_2.physics_info.acceleration,
                  );
                  let Some((time_s, displacement_1, displacement_2)) = ls_coll_time else { continue };
                  if time_s < min_collision_time {
                    min_collision_time = time_s;
                    let displaced_ls_1 = edge_1.displace(displacement_1);
                    let displaced_ls_2 = edge_2.displace(displacement_2);
                    collision_plane = Plane::new(
                      displaced_ls_1.get_direction().cross(displaced_ls_2.get_direction()),
                      displaced_ls_1.get_start(),
                    );
                    collision_point = displaced_ls_1.get_start();
                  }
                }
              }
            }
            RigidBodyType::Sphere(_) => {}
          },
          RigidBodyType::Sphere(_) => {}
        }
      }
    }

    (min_collision_time, collision_plane, collision_point)
  }

  pub fn run_one_ms(&mut self) {
    let mut min_collision_time = f32::MAX;
    let mut remaining_sim_time = 0.001;
    let mut coll_details = (0..self.rigid_bodies.len())
      .map(|_| Vec::with_capacity(self.rigid_bodies.len()))
      .collect::<Vec<_>>();
    while remaining_sim_time > 0.0 {
      for i in 0..self.rigid_bodies.len() {
        for j in i + 1..self.rigid_bodies.len() {
          let (body_coll_time, collision_plane, collision_point) =
            Self::rigid_body_coll_time(&self.rigid_bodies[i], &self.rigid_bodies[j]);
          coll_details[i][j] = (body_coll_time, collision_plane, collision_point);
          coll_details[j][i] = (body_coll_time, collision_plane, collision_point);
        }
      }
    }
  }
}
