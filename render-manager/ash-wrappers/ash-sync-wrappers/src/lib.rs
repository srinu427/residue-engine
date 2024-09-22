use ash_common_imports::ash::{self, vk};
use std::sync::Arc;

pub struct AdSemaphore {
  vk_device: Arc<ash::Device>,
  inner: vk::Semaphore,
}

impl AdSemaphore {
  pub fn new(vk_device: Arc<ash::Device>, vk_semaphore: vk::Semaphore) -> Self {
    Self { vk_device, inner: vk_semaphore }
  }

  pub fn inner(&self) -> vk::Semaphore {
    self.inner
  }
}

impl Drop for AdSemaphore {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.destroy_semaphore(self.inner, None);
    }
  }
}

pub struct AdFence {
  vk_device: Arc<ash::Device>,
  inner: vk::Fence,
}

impl AdFence {
  pub fn new(vk_device: Arc<ash::Device>, vk_fence: vk::Fence) -> Self {
    Self { vk_device, inner: vk_fence }
  }

  pub fn inner(&self) -> vk::Fence {
    self.inner
  }

  pub fn wait(&self, timeout: u64) -> Result<(), String> {
    unsafe {
      self
        .vk_device
        .wait_for_fences(&[self.inner], true, timeout)
        .map_err(|e| format!("at vk fence wait: {e}"))
    }
  }

  pub fn reset(&self) -> Result<(), String> {
    unsafe {
      self.vk_device.reset_fences(&[self.inner]).map_err(|e| format!("at vk fence reset: {e}"))
    }
  }
}

impl Drop for AdFence {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.destroy_fence(self.inner, None);
    }
  }
}
