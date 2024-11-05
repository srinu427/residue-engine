use render_manager::{glam, AdAshInstance, AdSurface, AdSurfaceInstance, Renderer, RendererMessage, TriMeshCPU};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes, WindowId};

pub struct AppActivity {
  surface: Option<Arc<AdSurface>>,
  window: Option<Window>,
  render_manager: Option<Renderer>,
  ash_instance: Arc<AdAshInstance>,
}

impl AppActivity {
  pub fn new() -> Result<Self, String> {
    let ash_instance = Arc::new(AdAshInstance::new()?);
    Ok(Self { ash_instance, window: None, render_manager: None, surface: None })
  }
}

impl ApplicationHandler for AppActivity {
  fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    if self.window.is_none() {
      match event_loop.create_window(WindowAttributes::default()) {
        Ok(w) => {
          let surface_instance = Arc::new(AdSurfaceInstance::new(self.ash_instance.clone()));
          let surface = match AdSurface::new(surface_instance, &w) {
            Ok(x) => Arc::new(x),
            Err(e) => {
              eprintln!("error creating surface: {e}");
              event_loop.exit();
              return;
            }
          };
          let mut renderer =
            match Renderer::new(surface.clone()) {
              Ok(x) => x,
              Err(e) => {
                eprintln!("error creating window: {e}");
                event_loop.exit();
                return;
              }
            };
          let tri_verts_cpu = TriMeshCPU::make_cuboid(
            glam::vec3(0.0, 0.0, 0.0),
            glam::vec3(1.0, 0.0, 0.0),
            glam::vec3(0.0, 1.0, 0.0),
            1.0,
          );
          let _ = renderer.send_batch_sync(
            vec![
              RendererMessage::AddTriMeshIfNotPresent(
                "triangle_main".to_string(),
                tri_verts_cpu,
              ),
              RendererMessage::AddFlatTexIfNotPresent(
                "./background.png".to_string(),
                "./background.png".to_string(),
              ),
              RendererMessage::AddTriMeshFlatTexToRender(
                "triangle_main".to_string(),
                "./background.png".to_string(),
              ),
            ])
            .inspect_err(|e| println!("at sending work to renderer: {e}"));
        
          self.surface = Some(surface);
          self.window = Some(w);
          self.render_manager = Some(renderer);
        }
        Err(e) => {
          eprintln!("error creating window: {e}");
          event_loop.exit()
        }
      }
    }
  }

  fn window_event(
    &mut self,
    event_loop: &ActiveEventLoop,
    _window_id: WindowId,
    event: WindowEvent,
  ) {
    // println!("event: {event:?}");
    match event {
      WindowEvent::ActivationTokenDone { .. } => {}
      WindowEvent::Resized(_) => {}
      WindowEvent::Moved(_) => {}
      WindowEvent::CloseRequested => {
        #[cfg(target_os = "macos")]
        let _ = self.window.take();
        event_loop.exit();
      }
      WindowEvent::Destroyed => {}
      WindowEvent::DroppedFile(_) => {}
      WindowEvent::HoveredFile(_) => {}
      WindowEvent::HoveredFileCancelled => {}
      WindowEvent::Focused(_) => {}
      WindowEvent::KeyboardInput { .. } => {}
      WindowEvent::ModifiersChanged(_) => {}
      WindowEvent::Ime(_) => {}
      WindowEvent::CursorMoved { .. } => {}
      WindowEvent::CursorEntered { .. } => {}
      WindowEvent::CursorLeft { .. } => {}
      WindowEvent::MouseWheel { .. } => {}
      WindowEvent::MouseInput { .. } => {}
      WindowEvent::PinchGesture { .. } => {}
      WindowEvent::PanGesture { .. } => {}
      WindowEvent::DoubleTapGesture { .. } => {}
      WindowEvent::RotationGesture { .. } => {}
      WindowEvent::TouchpadPressure { .. } => {}
      WindowEvent::AxisMotion { .. } => {}
      WindowEvent::Touch(_) => {}
      WindowEvent::ScaleFactorChanged { .. } => {}
      WindowEvent::ThemeChanged(_) => {}
      WindowEvent::Occluded(_) => {}
      WindowEvent::RedrawRequested => {
        self.render_manager.as_mut().map(|x| {
          let _ = x
            .send_batch_sync(vec![RendererMessage::Draw])
            .inspect_err(|e| eprintln!("at sending draw message: {e}"));
        });
      }
    }
  }

  fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
    self.render_manager.as_mut().map(|x| {
      let _ = x
        .send_batch_sync(vec![RendererMessage::Draw])
        .inspect_err(|e| eprintln!("at sending draw message: {e}"));
    });
  }
}
