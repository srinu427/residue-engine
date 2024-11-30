pub mod collision;
mod field;

use std::collections::HashMap;
use collision::{PolygonMesh, SeparationPlane};

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
  velocity: glam::Vec3,
  acceleration: glam::Vec3,
  moment_of_inertia: PhysicsMomentOfInertia,
  angular_velocity: glam::Vec3,
  angular_acceleration: glam::Vec3,
}

impl Default for PhysicsInfo {
  fn default() -> Self {
    Self {
      mass: PhysicsMass::Infinite,
      velocity: glam::Vec3::ZERO,
      acceleration: glam::Vec3::ZERO,
      moment_of_inertia: PhysicsMomentOfInertia::Infinite,
      angular_velocity: glam::Vec3::ZERO,
      angular_acceleration: glam::Vec3::ZERO,
    }
  }
}

impl PhysicsInfo {
  pub fn update_and_get_transform(&mut self, time_ms: u128) -> glam::Mat4 {
    let time_float = time_ms as f32 / 1000.0;
    let translation =
      (self.velocity * time_float) + (0.5 * self.acceleration * time_float * time_float);
    let rotation = (self.angular_velocity * time_float) +
      (0.5 * self.angular_acceleration * time_float * time_float);
    let rotation_transform = glam::Mat4::from_axis_angle(rotation.normalize(), rotation.length());
    let translation_transform = glam::Mat4::from_translation(translation);

    self.velocity = self.acceleration * time_float;
    self.angular_velocity = self.angular_acceleration * time_float;

    translation_transform * rotation_transform
  }

  pub fn get_transform(&self, time_ms: u128) -> glam::Mat4 {
    let time_float = time_ms as f32 / 1000.0;
    let translation =
      (self.velocity * time_float) + (0.5 * self.acceleration * time_float * time_float);
    let rotation = (self.angular_velocity * time_float) +
      (0.5 * self.angular_acceleration * time_float * time_float);
    let rotation_transform = glam::Mat4::from_axis_angle(rotation.normalize(), rotation.length());
    let translation_transform = glam::Mat4::from_translation(translation);
    translation_transform * rotation_transform
  }
}

pub struct PhysicsObject {
  mesh: PolygonMesh,
  transform: glam::Mat4,
  physics_info: PhysicsInfo
}

impl PhysicsObject {
  pub fn new(mesh: PolygonMesh, transform: glam::Mat4) -> Self {
    Self { mesh, transform, physics_info: PhysicsInfo::default() }
  }
}

pub struct PhysicsEngine {
  static_object_reserve_len: usize,
  dynamic_object_reserve_len: usize,
  static_objects: HashMap<String, PolygonMesh>,
  dynamic_objects: HashMap<String, PolygonMesh>,
  dyn_dyn_separations: HashMap<(String, String), SeparationPlane>,
  dyn_static_separations: HashMap<(String, String), SeparationPlane>,
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

  pub fn add_static_polygon(&mut self, name: &str, polygon: PolygonMesh) {
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
      self
        .dyn_static_separations
        .insert((dyno_name.clone(), name.to_string()), polygon.get_separation_plane(dyno));
    }
    self
      .static_objects
      .insert(name.to_string(), polygon);
  }

  pub fn add_dynamic_polygon(&mut self, name: &str, polygon: PolygonMesh) {
    // Check if the hashmap capacities are full
    if self.static_objects.len() == self.static_objects.capacity() {
      self.static_objects.reserve(self.static_object_reserve_len);
    }
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
      self
        .dyn_static_separations
        .insert((name.to_string(), so_name.clone()), polygon.get_separation_plane(so));
    }
    for (dyno_name, dyno) in self.dynamic_objects.iter() {
      self
        .dyn_dyn_separations
        .insert((name.to_string(), dyno_name.clone()), polygon.get_separation_plane(dyno));
    }
    self
      .dynamic_objects
      .insert(name.to_string(), polygon);
  }
}