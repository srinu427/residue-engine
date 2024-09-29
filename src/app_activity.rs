use std::sync::Arc;
use render_manager::{ AdAshInstance, AdSurface, AdSurfaceInstance, RenderManager};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes, WindowId};

pub struct AppActivity {
  surface: Option<Arc<AdSurface>>,
  window: Option<Window>,
  render_manager: Option<RenderManager>,
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
            Ok(x) => {Arc::new(x)}
            Err(e) => {
              eprintln!("error creating surface: {e}");
              event_loop.exit();
              return
            }
          };
          let render_manager =
            match RenderManager::new(Arc::clone(&self.ash_instance.clone()), surface.clone()) {
            Ok(x) => {x}
            Err(e) => {
              eprintln!("error creating window: {e}");
              event_loop.exit();
              return
            }
          };
          self.surface = Some(surface);
          self.window = Some(w);
          self.render_manager = Some(render_manager);
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
        self.render_manager.as_mut().map(
          |x| for _ in 0..3 {
            if let Ok(d_res) = x.draw().inspect_err(|e| eprintln!("{}", e)) {
              if !d_res {break}
            }
          }
        );
      }
    }
  }

  fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
    self.render_manager.as_mut().map(|x| x.draw().inspect_err(|e| eprintln!("{}", e)));
  }
}
