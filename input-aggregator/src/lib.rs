use std::collections::HashMap;
pub use winit::keyboard::Key;


#[derive(Debug, Clone, Copy)]
pub enum KeyState {
  Idle,
  Pressed,
  Held,
  Released,
}

impl KeyState {
  pub fn is_pressed(&self) -> bool {
    match self {
      KeyState::Idle => false,
      KeyState::Pressed => true,
      KeyState::Held => true,
      KeyState::Released => false,
    }
  }
}

pub struct InputAggregator {
  key_states: HashMap<winit::keyboard::Key, KeyState>,
}

impl InputAggregator {
  pub fn new() -> Self {
    InputAggregator { key_states: HashMap::new() }
  }

  pub fn is_key_pressed(&self, key: winit::keyboard::Key) -> KeyState {
    self.key_states.get(&key).cloned().unwrap_or(KeyState::Idle)
  }

  pub fn update_key_pressed(&mut self, key: winit::keyboard::Key) {
    self
      .key_states
      .entry(key)
      .and_modify(|x| *x = KeyState::Pressed)
      .or_insert(KeyState::Pressed);
  }

  pub fn update_key_released(&mut self, key: winit::keyboard::Key) {
    self
      .key_states
      .entry(key)
      .and_modify(|x| *x = KeyState::Released)
      .or_insert(KeyState::Released);
  }

  pub fn clear_key_states(&mut self) {
    for (_, v) in self.key_states.iter_mut() {
      *v = match v {
        KeyState::Idle => KeyState::Idle,
        KeyState::Pressed => KeyState::Held,
        KeyState::Held => KeyState::Held,
        KeyState::Released => KeyState::Idle,
      }
    }
  }
}
