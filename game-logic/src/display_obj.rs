use std::{collections::HashMap, path::Path, sync::{Arc, OnceLock}};

use animation::{RTSAnimation, RTSAnimator};
use render_manager::{FlatTextureGPU, RendererMessage, TriMeshCPU, TriMeshGPU, TriMeshTransform};

pub struct TriMeshFlatTex {
  mesh: Arc<OnceLock<Arc<TriMeshGPU>>>,
  tex: Arc<OnceLock<Arc<FlatTextureGPU>>>,
  rts_animations: HashMap<String, RTSAnimation>,
  current_animation: Option<RTSAnimator>,
  object_transform: TriMeshTransform,
}

impl TriMeshFlatTex {
  pub fn new(name: &str, mesh: TriMeshCPU, tex: &Path) -> (Self, Vec<RendererMessage>) {
    let mesh_ptr = Arc::new(OnceLock::new());
    let tex_ptr = Arc::new(OnceLock::new());
    let renderer_cmds = vec![
      RendererMessage::UploadTriMesh(name.to_string(), mesh, mesh_ptr.clone()),
      RendererMessage::UploadFlatTex(
        tex.to_string_lossy().to_string(),
        tex.to_string_lossy().to_string(),
        tex_ptr.clone(),
      ),
    ];
    let self_obj = Self {
      mesh: mesh_ptr,
      tex: tex_ptr,
      rts_animations: HashMap::new(),
      current_animation: None,
      object_transform: TriMeshTransform { transform: glam::Mat4::IDENTITY }
    };
    (self_obj, renderer_cmds)
  }

  pub fn update(&mut self, frame_time: u128) -> Result<(), String> {
    self
      .current_animation
      .as_mut()
      .map(|a| { a.forward(frame_time); a.get_transform() })
      .inspect(|t| self.object_transform.transform = *t);
    self
      .mesh
      .get()
      .as_ref()
      .map(|mesh|
        mesh
          .update_transform(self.object_transform)
          .inspect_err(|e| eprintln!("at obj transform update: {e}")));
    Ok(())
  }

  pub fn get_draw_info(&self) -> Result<(Arc<TriMeshGPU>, Arc<FlatTextureGPU>), String> {
    let mesh = self
      .mesh
      .get()
      .ok_or("mesh not uploaded to GPU")?
      .clone();
    let tex = self
      .tex
      .get()
      .ok_or("tex not uploaded to GPU")?
      .clone();
      
    Ok((mesh, tex))
  }
}
