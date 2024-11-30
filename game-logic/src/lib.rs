use std::sync::{Arc, OnceLock};

use animation::KeyFramed;
use input_aggregator::{InputAggregator, Key};
use physics::{collision::PolygonMesh, PhysicsEngine, PhysicsObject};
use render_manager::{AdSurface, Camera3D, FlatTextureGPU, Renderer, RendererMessage, TriMeshCPU, TriMeshGPU, TriMeshTransform};

mod animation;

pub struct GameObject {
  pub display_mesh: Arc<OnceLock<Arc<TriMeshGPU>>>,
  pub display_tex: Arc<OnceLock<Arc<FlatTextureGPU>>>,
  pub physics_name: Option<(bool, String)>,
  pub animation_time: u128,
  pub rotation_animation: KeyFramed<f32>,
  pub object_transform: TriMeshTransform,
}

impl GameObject {
  pub fn update(&mut self, frame_time: u128) -> Result<(), String> {
    self.animation_time += frame_time;
    let y_angle = self.rotation_animation.value_at(self.animation_time % 5000);
    self.object_transform.transform = glam::Mat4::from_rotation_y(y_angle);
    // let rot_mat = glam::Mat4::from_rotation_y(frame_time as f32/ 500.0);
    // self.object_transform.transform = self.object_transform.transform * rot_mat;
    self
      .display_mesh
      .get()
      .map(|mesh|
        mesh
          .update_transform(self.object_transform)
          .inspect_err(|e| eprintln!("at obj transform update: {e}")));
    Ok(())
  }
}

pub struct Game {
  game_objects: Vec<GameObject>,
  renderer: Renderer,
  physics_engine: PhysicsEngine,
  camera: Camera3D,
  start_time: std::time::Instant,
  last_update: std::time::Duration,
}

impl Game {
  pub fn new(surface: Arc<AdSurface>) -> Result<Self, String> {
    let mut renderer = Renderer::new(surface.clone()).map_err(|e| format!("at renderer init: {e}"))?;
    let mut physics_engine = PhysicsEngine::new(1000, 100);
    let start_time = std::time::Instant::now();
    let tri_verts_cpu = TriMeshCPU::make_cuboid(
      glam::vec3(0.0, 0.0, 0.0),
      glam::vec3(1.0, 0.0, 0.0),
      glam::vec3(0.0, 1.0, 0.0),
      1.0,
    );
    let cube_phy_object = PhysicsObject::new(
      PolygonMesh::new_cuboid(
        glam::vec3(0.0, 0.0, 0.0),
        glam::vec3(1.0, 0.0, 0.0),
        glam::vec3(0.0, 1.0, 0.0),
        1.0
      ),
      glam::Mat4::IDENTITY
    );
    physics_engine.add_dynamic_physics_obj("cube_physics", cube_phy_object);
    let game_obj = GameObject {
      display_mesh: Arc::new(OnceLock::new()),
      display_tex: Arc::new(OnceLock::new()),
      physics_name: Some((true, "cube_physics".to_string())),
      object_transform: TriMeshTransform { transform: glam::Mat4::IDENTITY },
      animation_time: 0,
      rotation_animation: KeyFramed { key_frames: vec![(0, 0.0), (2000, 360f32.to_radians()), (4000, 0.0)] },
    };

    let floor_verts_cpu = TriMeshCPU::make_planar_polygon(
      PolygonMesh::new_rectangle(
        glam::vec3(0.0, -2.0, 0.0),
        glam::vec3(10.0, 0.0, 0.0),
        glam::vec3(0.0, 0.0, -10.0),
      ).get_faces().remove(0));
    let floor_phy_object = PhysicsObject::new(
      PolygonMesh::new_rectangle(
        glam::vec3(0.0, -2.0, 0.0),
        glam::vec3(10.0, 0.0, 0.0),
        glam::vec3(0.0, 0.0, -10.0)
      ),
      glam::Mat4::IDENTITY
    );
    physics_engine.add_dynamic_physics_obj("floor_physics", floor_phy_object);
    let floor = GameObject {
      display_mesh: Arc::new(OnceLock::new()),
      display_tex: Arc::new(OnceLock::new()),
      physics_name: Some((false, "floor_physics".to_string())),
      object_transform: TriMeshTransform { transform: glam::Mat4::IDENTITY },
      animation_time: 0,
      rotation_animation: KeyFramed { key_frames: vec![(0, 0.0)] },
    };

    renderer
      .send_batch_sync(vec![
        RendererMessage::UploadTriMesh(
          "triangle_main".to_string(),
          tri_verts_cpu,
          game_obj.display_mesh.clone()
        ),
        RendererMessage::UploadTriMesh(
          "floor".to_string(),
          floor_verts_cpu,
          floor.display_mesh.clone()
        ),
        RendererMessage::UploadFlatTex(
          "./background.png".to_string(),
          "./background.png".to_string(),
          game_obj.display_tex.clone(),
        ),
        RendererMessage::UploadFlatTex(
          "./background.png".to_string(),
          "./background.png".to_string(),
          floor.display_tex.clone(),
        ),
      ])
      .map_err(|e| format!("at sending work to renderer: {e}"))?;
    Ok(Self {
      renderer,
      physics_engine,
      game_objects: vec![game_obj, floor],
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
        .get()
        .cloned() else { continue };
      let Some(ftex) = go
        .display_tex
        .get()
        .cloned() else { continue };
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
