use game_logic::Game;
use input_aggregator::InputAggregator;
use render_manager::{AdAshInstance, AdSurface, AdSurfaceInstance};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes, WindowId};

pub struct AppActivity {
  surface: Option<Arc<AdSurface>>,
  window: Option<Window>,
  game: Option<Game>,
  input_aggregator: InputAggregator,
  ash_instance: Arc<AdAshInstance>,
}

impl AppActivity {
  pub fn new() -> Result<Self, String> {
    let ash_instance = Arc::new(AdAshInstance::new()?);
    Ok(Self {
      ash_instance,
      input_aggregator: InputAggregator::new(),
      window: None,
      game: None,
      surface: None,
    })
  }
}

impl ApplicationHandler for AppActivity {
  fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    if self.window.is_none() {
      let Ok(w) = event_loop
        .create_window(WindowAttributes::default())
        .inspect_err(|e| eprintln!("error creating window: {e}")) else {
        event_loop.exit();
        return
      };

      let surface_instance = Arc::new(AdSurfaceInstance::new(self.ash_instance.clone()));
      let surface = match AdSurface::new(surface_instance, &w) {
        Ok(x) => Arc::new(x),
        Err(e) => {
          eprintln!("error creating surface: {e}");
          event_loop.exit();
          return;
        }
      };
      let game = match Game::new(surface.clone()) {
        Ok(x) => x,
        Err(e) => {
          eprintln!("error creating window: {e}");
          event_loop.exit();
          return;
        }
      };
      self.surface = Some(surface);
      self.window = Some(w);
      self.game = Some(game);
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
        // #[cfg(target_os = "macos")]
        // let _ = self.window.take();
        event_loop.exit();
      }
      WindowEvent::Destroyed => {}
      WindowEvent::DroppedFile(_) => {}
      WindowEvent::HoveredFile(_) => {}
      WindowEvent::HoveredFileCancelled => {}
      WindowEvent::Focused(_) => {}
      WindowEvent::KeyboardInput { device_id, event, is_synthetic } => {
        match event.state {
          winit::event::ElementState::Pressed => {
            self.input_aggregator.update_key_pressed(event.key_without_modifiers());
          },
          winit::event::ElementState::Released => {
            self.input_aggregator.update_key_released(event.key_without_modifiers());
          },
        }
      }
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
        self
          .game
          .as_mut()
          .map(|x| {
            x.update(&self.input_aggregator);
            self.input_aggregator.clear_key_states();
          });
      }
    }
  }

  fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
    self
      .game
      .as_mut()
      .map(|x| {
        x.update(&self.input_aggregator);
        self.input_aggregator.clear_key_states();
      });
  }
}
