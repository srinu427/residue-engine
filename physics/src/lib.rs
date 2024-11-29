pub mod collision;
mod field;

use heapless::FnvIndexMap;
use collision::{PolygonMesh, SeparationPlane};

pub struct PhysicsObject {
  mesh: PolygonMesh,
  transform: glam::Mat4,
}

pub struct PhysicsEngine<const S: usize ,const D: usize> {
  static_objects: FnvIndexMap<String, PolygonMesh, S>,
  dynamic_objects: FnvIndexMap<String, PolygonMesh, D>,
  dyn_dyn_separations: FnvIndexMap<String, FnvIndexMap<String, SeparationPlane, D>, D>,
  dyn_static_separations: FnvIndexMap<String, FnvIndexMap<String, SeparationPlane, S>, D>,
}

impl<const S: usize ,const D: usize> PhysicsEngine<S,D> {
  pub fn new() -> Self<S,D> {
    Self {
      static_objects: FnvIndexMap::new(),
      dynamic_objects: FnvIndexMap::new(),
      dyn_dyn_separations: FnvIndexMap::new(),
      dyn_static_separations: FnvIndexMap::new(),
    }
  }

  pub fn add_static_polygon(&mut self, name: &str, polygon: PolygonMesh) -> Result<(), String> {
    for (do_id, sep_list) in self.dyn_static_separations.iter_mut() {
      sep_list.push(self.dynamic_objects[do_id].get_separation_plane(&polygon))
        .map_err(|_| "max static objects count reached".to_string())?;
    }
    self
      .static_objects
      .insert(name.to_string(), polygon)
      .map_err(|_| "max static objects count reached".to_string())
  }

  pub fn add_dynamic_polygon(&mut self, polygon: PolygonMesh) -> Result<(), String> {
    self
      .dyn_static_separations
      .push(self.static_objects.iter().map(|so| polygon.get_separation_plane(so)).collect())
      .map_err(|_| "max dynamic objects count reached".to_string())?;
    self
      .dyn_static_separations
      .push(self.dynamic_objects.iter().map(|dyn_o| polygon.get_separation_plane(dyn_o)).collect())
      .map_err(|_| "max dynamic objects count reached".to_string())?;
    self.dynamic_objects.push(polygon).map_err(|_| "max dynamic objects count reached".to_string())
  }
}