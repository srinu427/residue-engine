pub use glam;
use glam::Vec4Swizzles;
pub mod flat_texture;
pub mod triangle_mesh;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Camera3D {
  pub pos: glam::Vec4,
  pub look_dir: glam::Vec4,
  pub view_proj_mat: glam::Mat4,
}

impl Camera3D {
  pub fn new(pos: glam::Vec4, look_dir: glam::Vec4, fov: f32) -> Self {
    let mut cam = Camera3D { pos, look_dir, view_proj_mat: glam::Mat4::IDENTITY };
    cam.refresh_vp_matrix(fov, 1.0);
    cam
  }

  pub fn refresh_vp_matrix(&mut self, fov: f32, aspect_ratio: f32) {
    self.view_proj_mat = glam::Mat4::perspective_rh(fov, aspect_ratio, 1.0, 1000.0)
      * glam::Mat4::look_at_rh(
        self.pos.xyz(),
        self.pos.xyz() + self.look_dir.xyz(),
        glam::Vec3 { x: 0.0f32, y: 1.0f32, z: 0.0f32 },
      );
  }
}
