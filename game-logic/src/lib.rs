use std::sync::Arc;

use render_manager::{AdSurface, Renderer};

mod physics;

pub struct Game {
  renderer: Renderer,
  start_time: std::time::Instant,
  last_update: std::time::Duration,
}

impl Game {
  pub fn new(surface: Arc<AdSurface>) -> Result<Self, String>{
    let renderer =
      Renderer::new(surface.clone()).map_err(|e| format!("at renderer init: {e}"))?;
    let start_time = std::time::Instant::now();
    Ok(Self { renderer, start_time, last_update: start_time.elapsed() })
  }

  pub fn update(&mut self) {
    let current_dur = self.start_time.elapsed();
    let frame_time = current_dur.as_millis() - self.last_update.as_millis();
    self.last_update = current_dur;
  }
}
