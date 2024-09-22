use ash_common_imports::ash::{self, vk};
pub use ash_common_imports::gpu_allocator::vulkan::{Allocation, Allocator};
use std::sync::{Arc, Mutex};

pub struct AdBuffer {
  inner: vk::Buffer,
  size: vk::DeviceSize,
  name: String,
  vk_device: Arc<ash::Device>,
  allocator: Arc<Mutex<Allocator>>,
  allocation: Option<Allocation>,
}

impl AdBuffer {
  pub fn new(
    vk_device: Arc<ash::Device>,
    vk_buffer: vk::Buffer,
    name: &str,
    size: vk::DeviceSize,
    allocator: Arc<Mutex<Allocator>>,
    allocation: Option<Allocation>
  ) -> Self {
    Self { inner: vk_buffer, size, name: name.to_string(), vk_device, allocator, allocation }
  }

  pub fn size(&self) -> vk::DeviceSize {
    self.size
  }

  pub fn name(&self) -> &str {
    &self.name
  }

  pub fn inner(&self) -> vk::Buffer {
    self.inner
  }

  pub fn write_data(&mut self, offset: usize, data: &[u8]) -> Result<(), String> {
    if offset + data.len() > self.size as usize {
      return Err(format!("too much data lol. buffer {} only supports {} bytes", &self.name, self.size))
    }
    self
      .allocation
      .as_mut()
      .map(|alloc| {
        alloc
          .mapped_slice_mut()
          .map(|x| {
            x[offset..data.len()]
              .copy_from_slice(data)
          })
          .ok_or(format!("at mapping buffer {} 's memory", &self.name))
      })
      .ok_or(format!("no memory allocated for buffer {}", &self.name))??; // second ? for failure in mapped_slice_mut
    Ok(())
  }
}

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
  inner: vk::Image,
  format: vk::Format,
  resolution: vk::Extent2D,
  name: String,
  vk_device: Arc<ash::Device>,
  allocator: Arc<Mutex<Allocator>>,
  allocation: Option<Allocation>,
}

impl AdImage2D {
  pub fn new(
    vk_device: Arc<ash::Device>,
    vk_image: vk::Image,
    name: &str,
    resolution: vk::Extent2D,
    format: vk::Format,
    allocator: Arc<Mutex<Allocator>>,
    allocation: Option<Allocation>
  ) -> Self {
    Self { vk_device, inner: vk_image, name: name.to_string(), resolution, format, allocator, allocation }
  }

  pub fn name(&self) -> &str {
    &self.name
  }

  pub fn inner(&self) -> vk::Image {
    self.inner
  }

  pub fn resolution(&self) -> vk::Extent2D {
    self.resolution
  }

  pub fn format(&self) -> vk::Format {
    self.format
  }

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
      vk_device: self.vk_device.clone(),
      inner: image_view,
    })
  }
}

impl Drop for AdImage2D {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.destroy_image(self.inner, None);
    }
    let _ = self.allocator
      .lock()
      .map(|mut altr| self.allocation.take().map(|altn| altr.free(altn)))
      .inspect_err(|e| eprintln!("at getting allocator lock while image destroy: {e}"));
  }
}

pub struct AdImageView {
  vk_device: Arc<ash::Device>,
  inner: vk::ImageView,
}

impl AdImageView {
  pub fn new(vk_device: Arc<ash::Device>, vk_image_view: vk::ImageView) -> Self {
    Self { vk_device, inner: vk_image_view }
  }

  pub fn inner(&self) -> vk::ImageView {
    self.inner
  }
}

impl Drop for AdImageView {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.destroy_image_view(self.inner, None);
    }
  }
}
