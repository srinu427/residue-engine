use std::sync::{Arc, RwLock};

use render_manager::{AdSurface, FlatTextureGPU, Renderer, RendererMessage, TriMeshCPU, TriMeshGPU, TriMeshTransform};

mod physics;

pub struct GameObject {
  pub display_mesh: Arc<RwLock<Option<Arc<TriMeshGPU>>>>,
  pub display_tex: Arc<RwLock<Option<Arc<FlatTextureGPU>>>>,
  pub object_transform: TriMeshTransform,
}

impl GameObject {
  pub fn update(&mut self, frame_time: u128) -> Result<(), String> {
    let rot_mat = glam::Mat4::from_rotation_y(frame_time as f32/ 1000.0);
    self.object_transform.transform = self.object_transform.transform * rot_mat;
    self
        .display_mesh
        .read()
        .map_err(|e| format!("at locking mesh to render: {e}"))?
        .as_ref()
        .map(|mesh| mesh.update_transform(self.object_transform).inspect_err(|e| eprintln!("at obj transform update: {e}")));
    Ok(())
  }
}

pub struct Game {
  renderer: Renderer,
  game_objects: Vec<GameObject>,
  start_time: std::time::Instant,
  last_update: std::time::Duration,
}

impl Game {
  pub fn new(surface: Arc<AdSurface>) -> Result<Self, String> {
    let mut renderer = Renderer::new(surface.clone()).map_err(|e| format!("at renderer init: {e}"))?;
    let start_time = std::time::Instant::now();
    let tri_verts_cpu = TriMeshCPU::make_cuboid(
      glam::vec3(0.0, 0.0, 0.0),
      glam::vec3(1.0, 0.0, 0.0),
      glam::vec3(0.0, 1.0, 0.0),
      1.0,
    );
    let game_obj = GameObject {
      display_mesh: Arc::new(RwLock::new(None)),
      display_tex: Arc::new(RwLock::new(None)),
      object_transform: TriMeshTransform { transform: glam::Mat4::IDENTITY }
    };
    renderer
      .send_batch_sync(vec![
        RendererMessage::UploadTriMesh(
          "triangle_main".to_string(),
          tri_verts_cpu,
          game_obj.display_mesh.clone()
        ),
        RendererMessage::UploadFlatTex(
          "./background.png".to_string(),
          "./background.png".to_string(),
          game_obj.display_tex.clone(),
        ),
      ])
      .map_err(|e| format!("at sending work to renderer: {e}"))?;
    Ok(Self { renderer, game_objects: vec![game_obj], start_time, last_update: start_time.elapsed() })
  }

  pub fn update(&mut self) -> Result<(), String> {
    let current_dur = self.start_time.elapsed();
    let frame_time = current_dur.as_millis() - self.last_update.as_millis();
    self.last_update = current_dur;

    let mut mesh_ftex_list = vec![];
    for go in self.game_objects.iter_mut() {
      go.update(frame_time)?;
    }
    for go in self.game_objects.iter() {
      let Some(mesh) = go
        .display_mesh
        .read()
        .map_err(|e| format!("at locking mesh to render: {e}"))?
        .clone() else { continue; };
      let Some(ftex) = go
        .display_tex
        .read()
        .map_err(|e| format!("at locking tex to render: {e}"))?
        .clone() else { continue; };
      mesh_ftex_list.push((mesh, ftex));
    }
    self.renderer.send_batch_sync(vec![RendererMessage::Draw(mesh_ftex_list)])?;
    Ok(())
  }
}
