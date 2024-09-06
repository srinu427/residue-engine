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

  pub fn create_view(&self, aspect_mask: vk::ImageAspectFlags) -> Result<AdImageView, String> {
    let view_create_info = vk::ImageViewCreateInfo::default()
      .image(self.inner)
      .format(self.format)
      .view_type(vk::ImageViewType::TYPE_2D)
      .subresource_range(
        vk::ImageSubresourceRange::default()
          .aspect_mask(aspect_mask)
          .layer_count(1)
          .base_array_layer(0)
          .level_count(1)
          .base_mip_level(0),
      )
      .components(vk::ComponentMapping {
        r: vk::ComponentSwizzle::R,
        g: vk::ComponentSwizzle::G,
        b: vk::ComponentSwizzle::B,
        a: vk::ComponentSwizzle::A,
      });
    let image_view = unsafe {
      self.vk_device.create_image_view(&view_create_info, None)
        .map_err(|e| format!("at creating vk image view: {e}"))?
    };
    Ok(AdImageView {
      vk_device: Arc::clone(&self.vk_device),
      inner: image_view,
    })
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

pub struct AdImageView {
  pub(crate) vk_device: Arc<ash::Device>,
  pub inner: vk::ImageView,
}

impl Drop for AdImageView {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.destroy_image_view(self.inner, None);
    }
  }
}
