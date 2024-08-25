use ash::vk;
use std::sync::Arc;

pub struct AdSemaphore {
  pub(crate) vk_device: Arc<ash::Device>,
  pub inner: vk::Semaphore,
}

impl Drop for AdSemaphore {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.destroy_semaphore(self.inner, None);
    }
  }
}

pub struct AdFence {
  pub(crate) vk_device: Arc<ash::Device>,
  pub inner: vk::Fence,
}

impl AdFence {
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
