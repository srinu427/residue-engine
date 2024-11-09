use std::sync::{Arc, RwLock};

use animation::KeyFramed;
use input_aggregator::{InputAggregator, Key};
use render_manager::{AdSurface, Camera3D, FlatTextureGPU, Renderer, RendererMessage, TriMeshCPU, TriMeshGPU, TriMeshTransform};

mod animation;
mod physics;

pub struct GameObject {
  pub display_mesh: Arc<RwLock<Option<Arc<TriMeshGPU>>>>,
  pub display_tex: Arc<RwLock<Option<Arc<FlatTextureGPU>>>>,
  pub animation_time: u128,
  pub rotation_animation: KeyFramed<f32>,
  pub object_transform: TriMeshTransform,
}

impl GameObject {
  pub fn update(&mut self, frame_time: u128) -> Result<(), String> {
    self.animation_time += frame_time;
    let y_angle = self.rotation_animation.value_at(self.animation_time % 6000);
    self.object_transform.transform = glam::Mat4::from_rotation_y(y_angle);
    // let rot_mat = glam::Mat4::from_rotation_y(frame_time as f32/ 500.0);
    // self.object_transform.transform = self.object_transform.transform * rot_mat;
    self
      .display_mesh
      .read()
      .map_err(|e| format!("at locking mesh to render: {e}"))?
      .as_ref()
      .map(|mesh|
        mesh
          .update_transform(self.object_transform)
          .inspect_err(|e| eprintln!("at obj transform update: {e}")));
    Ok(())
  }
}

pub struct Game {
  renderer: Renderer,
  game_objects: Vec<GameObject>,
  camera: Camera3D,
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
      object_transform: TriMeshTransform { transform: glam::Mat4::IDENTITY },
      animation_time: 0,
      rotation_animation: KeyFramed { key_frames: vec![(0, 0.0), (3000, 6.28), (6000, 0.0)] },
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
    Ok(Self {
      renderer,
      game_objects: vec![game_obj],
      start_time,
      last_update: start_time.elapsed(),
      camera: Camera3D::new(
        glam::vec4(2.0, 2.0, 2.0, 1.0),
        glam::vec4(-1.0, -1.0, -1.0, 1.0),
        1.0
      )
    })
  }

  pub fn update(&mut self, inputs: &InputAggregator) -> Result<(), String> {
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
        .clone() else { continue };
      let Some(ftex) = go
        .display_tex
        .read()
        .map_err(|e| format!("at locking tex to render: {e}"))?
        .clone() else { continue };
      mesh_ftex_list.push((mesh, ftex));
    }
    if inputs.is_key_pressed(Key::Character("a".into())).is_pressed() {
      self.camera.pos += glam::vec4(-1.0, 0.0, 1.0, 0.0) * frame_time as f32/500.0;
    }
    if inputs.is_key_pressed(Key::Character("d".into())).is_pressed() {
      self.camera.pos -= glam::vec4(-1.0, 0.0, 1.0, 0.0) * frame_time as f32/500.0;
    }
    self.renderer.send_batch_sync(vec![
      RendererMessage::SetCamera(self.camera),
      RendererMessage::Draw(mesh_ftex_list),
    ])?;
    Ok(())
  }
}
