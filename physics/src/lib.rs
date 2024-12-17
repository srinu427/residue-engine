pub mod collision;
pub use geometry;
mod field;

use std::collections::HashMap;
use collision::{PolygonMesh, SeparationType};
use glam::Vec4Swizzles;
use geometry::Direction;
use crate::collision::Separation;

#[derive(Debug, Copy, Clone)]
pub enum PhysicsMass {
  Infinite,
  Finite(f32),
}

#[derive(Debug, Copy, Clone)]
pub enum PhysicsMomentOfInertia {
  Infinite,
  Finite(glam::Mat3),
}

#[derive(Debug, Copy, Clone)]
pub struct PhysicsInfo {
  mass: PhysicsMass,
  position: glam::Vec3,
  velocity: glam::Vec3,
  acceleration: glam::Vec3,
  moment_of_inertia: PhysicsMomentOfInertia,
  rotation: glam::Mat4,
  angular_velocity: glam::Vec3,
  angular_acceleration: glam::Vec3,
}

impl Default for PhysicsInfo {
  fn default() -> Self {
    Self {
      mass: PhysicsMass::Infinite,
      position: glam::Vec3::ZERO,
      velocity: glam::Vec3::ZERO,
      acceleration: glam::Vec3::ZERO,
      moment_of_inertia: PhysicsMomentOfInertia::Infinite,
      rotation: glam::Mat4::IDENTITY,
      angular_velocity: glam::Vec3::ZERO,
      angular_acceleration: glam::Vec3::ZERO,
    }
  }
}

impl PhysicsInfo {
  pub fn update(&mut self, time_ms: u128, bounds: Vec<Direction>) {
    let time_float = time_ms as f32 / 1000.0;

    for bound in bounds.iter() {
      if self.velocity.dot(bound.as_vec3()) > 0.0 {
        self.velocity = self.velocity.reject_from(bound.as_vec3());
      }
    }

    for bound in bounds.iter() {
      if self.acceleration.dot(bound.as_vec3()) > 0.0 {
        self.acceleration = self.acceleration.reject_from(bound.as_vec3());
      }
    }

    let translation =
      (self.velocity * time_float) + (0.5 * self.acceleration * time_float * time_float);
    let rotation = (self.angular_velocity * time_float) +
      (0.5 * self.angular_acceleration * time_float * time_float);
    let rotation_transform = if rotation.length_squared() != 0.0 {
      glam::Mat4::from_axis_angle(rotation.normalize(), rotation.length())
    } else {
      glam::Mat4::IDENTITY
    };

    self.position += translation;
    self.velocity += self.acceleration * time_float;

    self.rotation = rotation_transform * self.rotation;
    self.angular_velocity += self.angular_acceleration * time_float;
  }

  pub fn get_transform(&self) -> glam::Mat4 {
    glam::Mat4::from_translation(self.position) * self.rotation
  }
}

#[derive(Debug, Clone)]
pub struct PhysicsObject {
  mesh: PolygonMesh,
  physics_info: PhysicsInfo,
  stuck: bool,
}

impl PhysicsObject {
  pub fn new(mesh: PolygonMesh, pos: glam::Vec3, rotation: glam::Mat4) -> Self {
    let physics_info = PhysicsInfo {
      mass: PhysicsMass::Infinite,
      position: pos,
      velocity: Default::default(),
      acceleration: Default::default(),
      moment_of_inertia: PhysicsMomentOfInertia::Infinite,
      rotation,
      angular_velocity: Default::default(),
      angular_acceleration: Default::default(),
    };
    Self { mesh, physics_info, stuck: false }
  }

  pub fn set_velocity(&mut self, velocity: glam::Vec3) {
    self.physics_info.velocity = velocity;
  }

  pub fn update(&mut self, time_ms: u128, bounds: Vec<Direction>) {
    self.physics_info.update(time_ms, bounds);
  }

  pub fn is_separation_valid(&self, other: &Self, sep_plane: SeparationType) -> bool {
    PolygonMesh::is_separation_plane_valid(
      sep_plane,
      &self.mesh,
      self.physics_info.get_transform(),
      &other.mesh,
      other.physics_info.get_transform(),
    )
  }
}

pub struct PhysicsEngine {
  static_object_reserve_len: usize,
  dynamic_object_reserve_len: usize,
  static_objects: HashMap<String, PhysicsObject>,
  dynamic_objects: HashMap<String, PhysicsObject>,
  dyn_dyn_separations: HashMap<(String, String), Separation>,
  dyn_static_separations: HashMap<(String, String), Separation>,
}

impl PhysicsEngine {
  pub fn new(static_obj_inc_len: usize, dynamic_obj_inc_len: usize) -> Self {
    Self {
      static_object_reserve_len: static_obj_inc_len,
      dynamic_object_reserve_len: dynamic_obj_inc_len,
      static_objects: HashMap::with_capacity(static_obj_inc_len),
      dynamic_objects: HashMap::with_capacity(dynamic_obj_inc_len),
      dyn_dyn_separations: HashMap::with_capacity(dynamic_obj_inc_len * dynamic_obj_inc_len),
      dyn_static_separations: HashMap::with_capacity(dynamic_obj_inc_len * static_obj_inc_len),
    }
  }

  pub fn get_dyn_obj_mut(&mut self, name: &str) -> Option<&mut PhysicsObject> {
    self.dynamic_objects.get_mut(name)
  }

  pub fn get_dynamic_object_transform(&self, name: &str) -> Option<glam::Mat4> {
    self
      .dynamic_objects
      .get(name)
      .map(|dynamic_transform| dynamic_transform.physics_info.get_transform())
  }

  pub fn get_static_object_transform(&self, name: &str) -> Option<glam::Mat4> {
    self
      .static_objects
      .get(name)
      .map(|static_transform| static_transform.physics_info.get_transform())
  }

  pub fn add_static_physics_obj(&mut self, name: &str, physics_obj: PhysicsObject) -> Result<(), String> {
    // Check if the hashmap capacities are full
    if self.static_objects.len() == self.static_objects.capacity() {
      self.static_objects.reserve(self.static_object_reserve_len);
    }
    if self.dyn_static_separations.len() == self.dyn_static_separations.capacity() {
      self
        .dyn_static_separations
        .reserve(self.dynamic_object_reserve_len * self.static_object_reserve_len);
    }
    // Add the objects
    for (dyno_name, dyno) in self.dynamic_objects.iter() {
      let separation = dyno
        .mesh
        .get_separation_plane(dyno.physics_info.get_transform(), &physics_obj.mesh, physics_obj.physics_info.get_transform())
        .map(Separation::Yes)
        .ok_or(format!("Separation for {} not found with {}", name, dyno_name))?;
      self
        .dyn_static_separations
        .insert((dyno_name.clone(), name.to_string()), separation);
    }
    self
      .static_objects
      .insert(name.to_string(), physics_obj);
    Ok(())
  }

  pub fn add_dynamic_physics_obj(&mut self, name: &str, mut physics_obj: PhysicsObject) -> Result<(), String> {
    physics_obj.physics_info.acceleration = glam::Vec3::new(0.0, -10.0, 0.0);
    // Check if the hashmap capacities are full
    if self.dynamic_objects.len() == self.dynamic_objects.capacity() {
      self.dynamic_objects.reserve(self.dynamic_object_reserve_len);
    }
    if self.dyn_static_separations.len() == self.dyn_static_separations.capacity() {
      self
        .dyn_static_separations
        .reserve(self.dynamic_object_reserve_len * self.static_object_reserve_len);
    }
    if self.dyn_dyn_separations.len() == self.dyn_dyn_separations.capacity() {
      self
        .dyn_dyn_separations
        .reserve(self.dynamic_object_reserve_len * self.dynamic_object_reserve_len);
    }
    // Add the objects
    for (so_name, so) in self.static_objects.iter() {
      let separation = physics_obj
        .mesh
        .get_separation_plane(physics_obj.physics_info.get_transform(), &so.mesh, so.physics_info.get_transform())
        .map(Separation::Yes)
        .ok_or(format!("Separation for {} not found with {}", name, so_name))?;
      self
        .dyn_static_separations
        .insert((name.to_string(), so_name.clone()), separation);
    }
    for (dyno_name, dyno) in self.dynamic_objects.iter() {
      let separation = physics_obj
        .mesh
        .get_separation_plane(physics_obj.physics_info.get_transform(), &dyno.mesh, dyno.physics_info.get_transform())
        .map(Separation::Yes)
        .ok_or(format!("Separation for {} not found with {}", name, dyno_name))?;
      self
        .dyn_dyn_separations
        .insert((name.to_string(), dyno_name.clone()), separation);
    }

    self
      .dynamic_objects
      .insert(name.to_string(), physics_obj);
    Ok(())
  }

  pub fn run(&mut self, time_ms: u128) {
    for _ in 0..time_ms {
      self.run_one_ms();
    }
  }

  pub fn run_one_ms(&mut self) {
    let mut next_dyn_objs = self
      .dynamic_objects
      .iter()
      .map(|dyno| {
        let mut next_dyn_obj = dyno.1.clone();
        next_dyn_obj.physics_info.acceleration = glam::Vec3::new(0.0, -10.0, 0.0);
        let mut curr_bounds = vec![];
        for (so_name, so) in self.static_objects.iter() {
          let Some(s_coll_sep) = self
            .dyn_static_separations
            .get(&(dyno.0.clone(), so_name.clone())) else {
            continue;
          };
          let Separation::No(sep_plane) = s_coll_sep else { continue; };
          let Some(sep_plane_vec4) = sep_plane
            .decode_to_vec4(
              &dyno.1.mesh,
              dyno.1.physics_info.get_transform(),
              &so.mesh,
              so.physics_info.get_transform()
            ) else {
            continue;
          };
          let penetration_dir = match sep_plane {
            SeparationType::FirstObjectFace { .. } => {
              sep_plane_vec4.get_direction()
            }
            SeparationType::SecondObjectFace { .. } => {
              sep_plane_vec4.get_direction().opposite()
            }
            SeparationType::EdgeCross { .. } => {
              sep_plane_vec4.get_direction()
            }
          };
          curr_bounds.push(penetration_dir);
        }
        next_dyn_obj.update(1, curr_bounds);
        (dyno.0.clone(), next_dyn_obj)
      })
      .collect::<HashMap<_, _>>();
    for (dyno_name, dynamic_obj) in next_dyn_objs.iter_mut() {
      let max_collision_resolves = 128;
      let mut collision_resolved_remaining = max_collision_resolves;

      while collision_resolved_remaining > 0 {
        collision_resolved_remaining -= 1;
        let mut collider_found = false;
        for (so_name, so) in self.static_objects.iter() {
          let Some(curr_sep_plane) =
            self.dyn_static_separations.get(&(dyno_name.clone(), so_name.clone())) else {
            continue;
          };
          let new_sep = match curr_sep_plane {
            Separation::No(s_plane) => {
              dynamic_obj.mesh.get_separation_plane(
                dynamic_obj.physics_info.get_transform(),
                &so.mesh,
                so.physics_info.get_transform(),
              )
                .map(Separation::Yes)
                .unwrap_or(Separation::No(s_plane.clone()))
            }
            Separation::Yes(s_plane) => {
              let sep_plane_still_valid = dynamic_obj.is_separation_valid(so, *s_plane);
              if sep_plane_still_valid {
                Separation::Yes(s_plane.clone())
              } else {
                dynamic_obj.mesh.get_separation_plane(
                  dynamic_obj.physics_info.get_transform(),
                  &so.mesh,
                  so.physics_info.get_transform(),
                )
                  .map(Separation::Yes)
                  .unwrap_or(Separation::No(s_plane.clone()))
              }
            }
          };
          self
            .dyn_static_separations
            .insert((dyno_name.clone(), so_name.clone()), new_sep);
          match new_sep {
            Separation::No(sep_plane) => {
              let Some(sep_plane_vec4) = sep_plane
                .decode_to_vec4(&dynamic_obj.mesh, dynamic_obj.physics_info.get_transform(), &so.mesh, so.physics_info.get_transform()) else {
                continue;
              };
              let penetration = match sep_plane {
                SeparationType::FirstObjectFace { .. } => {
                  so.mesh.get_distance_inside_plane(so.physics_info.get_transform(), sep_plane_vec4) * sep_plane_vec4.get_direction().as_vec3()
                }
                SeparationType::SecondObjectFace { .. } => {
                  -dynamic_obj.mesh.get_distance_inside_plane(dynamic_obj.physics_info.get_transform(), sep_plane_vec4) * sep_plane_vec4.get_direction().as_vec3()
                }
                SeparationType::EdgeCross { .. } => {
                  so.mesh.get_distance_inside_plane(so.physics_info.get_transform(), sep_plane_vec4) * sep_plane_vec4.get_direction().as_vec3()
                }
              };

              if penetration.length_squared() > 0.0 {
                dynamic_obj.physics_info.position += penetration;
                collider_found = true;
                break;
              }
            }
            Separation::Yes(_) => {}
          }
        }
        if !collider_found {
          break;
        }
      }
      if collision_resolved_remaining == 0 {
        dynamic_obj.stuck = true;
        println!("penetrations unresolvable for {dyno_name}");
      }
      self.dynamic_objects.insert(dyno_name.clone(), dynamic_obj.clone());
    }
  }
}