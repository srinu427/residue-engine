use winit::event_loop::{ControlFlow, EventLoop};
use crate::app_activity::AppActivity;

mod app_activity;

fn main() {
  let mut app = AppActivity::new()
    .inspect_err(|e| eprintln!("{e}"))
    .expect("error initializing app activity");
  let window_event_loop = EventLoop::new()
    .inspect_err(|e| eprintln!("{e}"))
    .expect("error initializing window event loop");
  window_event_loop.set_control_flow(ControlFlow::Poll);
  let _ = window_event_loop.run_app(&mut app).inspect_err(|e| eprintln!("{e}"));
}
