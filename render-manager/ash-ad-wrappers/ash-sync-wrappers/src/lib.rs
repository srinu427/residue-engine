use std::sync::Arc;

use ash_context::{ash::vk, getset, AdAshDevice};

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdSemaphore {
  ash_device: Arc<AdAshDevice>,
  #[getset(get_copy = "pub")]
  inner: vk::Semaphore,
}

impl AdSemaphore {
  pub fn new(
    ash_device: Arc<AdAshDevice>,
    flags: vk::SemaphoreCreateFlags,
  ) -> Result<Self, String> {
    unsafe {
      ash_device
        .inner()
        .create_semaphore(&vk::SemaphoreCreateInfo::default().flags(flags), None)
        .map_err(|e| format!("at create vk semaphore: {e}"))
        .map(|vk_semaphore| Self { ash_device, inner: vk_semaphore })
    }
  }
}

impl Drop for AdSemaphore {
  fn drop(&mut self) {
    unsafe {
      self.ash_device.inner().destroy_semaphore(self.inner, None);
    }
  }
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdFence {
  ash_device: Arc<AdAshDevice>,
  #[getset(get_copy = "pub")]
  inner: vk::Fence,
}

impl AdFence {
  pub fn new(ash_device: Arc<AdAshDevice>, flags: vk::FenceCreateFlags) -> Result<Self, String> {
    unsafe {
      ash_device
        .inner()
        .create_fence(&vk::FenceCreateInfo::default().flags(flags), None)
        .map_err(|e| format!("at create vk semaphore: {e}"))
        .map(|vk_fence| Self { ash_device, inner: vk_fence })
    }
  }

  pub fn wait(&self, timeout: u64) -> Result<(), String> {
    unsafe {
      self
        .ash_device
        .inner()
        .wait_for_fences(&[self.inner], true, timeout)
        .map_err(|e| format!("at vk fence wait: {e}"))
    }
  }

  pub fn reset(&self) -> Result<(), String> {
    unsafe {
      self
        .ash_device
        .inner()
        .reset_fences(&[self.inner])
        .map_err(|e| format!("at vk fence reset: {e}"))
    }
  }

  pub fn wait_and_reset(&self, timeout: u64) -> Result<(), String> {
    self.wait(timeout)?;
    self.reset()
  }
}

impl Drop for AdFence {
  fn drop(&mut self) {
    unsafe {
      self.ash_device.inner().destroy_fence(self.inner, None);
    }
  }
}
