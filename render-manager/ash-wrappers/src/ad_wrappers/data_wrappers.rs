use ash::vk;
pub use gpu_allocator::vulkan::{Allocation, Allocator};
use std::sync::{Arc, Mutex};

pub struct AdBuffer {
  pub inner: vk::Buffer,
  pub size: vk::DeviceSize,
  pub name: String,
  pub(crate) vk_device: Arc<ash::Device>,
  pub(crate) allocator: Arc<Mutex<Allocator>>,
  pub allocation: Option<Allocation>,
}

impl AdBuffer {}

impl Drop for AdBuffer {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.destroy_buffer(self.inner, None);
    }
    let _ = self
      .allocator
      .lock()
      .map(|mut altr| self.allocation.take().map(|altn| altr.free(altn)))
      .inspect_err(|e| eprintln!("at getting allocator lock while buffer destroy: {e}"));
  }
}

pub struct AdImage2D {
  pub inner: vk::Image,
  pub format: vk::Format,
  pub resolution: vk::Extent2D,
  pub name: String,
  pub(crate) vk_device: Arc<ash::Device>,
  pub(crate) allocator: Option<Arc<Mutex<Allocator>>>,
  pub(crate) allocation: Option<Allocation>,
}

impl AdImage2D {
  pub fn full_range_offset_3d(&self) -> [vk::Offset3D; 2] {
    [
      vk::Offset3D::default(),
      vk::Offset3D::default().x(self.resolution.width as i32).y(self.resolution.height as i32).z(1),
    ]
  }
}

impl Drop for AdImage2D {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.destroy_image(self.inner, None);
    }
    let _ = self.allocator.as_ref().map(|mtx_altr| {
      mtx_altr
        .lock()
        .map(|mut altr| self.allocation.take().map(|altn| altr.free(altn)))
        .inspect_err(|e| eprintln!("at getting allocator lock while image destroy: {e}"))
    });
  }
}
