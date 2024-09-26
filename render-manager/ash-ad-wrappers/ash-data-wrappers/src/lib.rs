use std::sync::{Arc, Mutex};

use ash_context::{ash::vk, AdAshDevice};
use ash_queue_wrappers::AdCommandBuffer;
use ash_sync_wrappers::AdFence;
use ash_context::gpu_allocator::{
  vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
  MemoryLocation
};

pub struct AdAllocation {
  allocator: Arc<Mutex<Allocator>>,
  inner: Option<Allocation>,
  name: String
}

impl AdAllocation {
  pub fn new(
    allocator: Arc<Mutex<Allocator>>,
    name: &str,
    mem_location: MemoryLocation,
    requirements: vk::MemoryRequirements,
  ) -> Result<Self, String>{
    let altn = allocator
      .lock()
      .map_err(|e| format!("at getting allocator lock: {e}"))?
      .allocate(&AllocationCreateDesc {
        name,
        requirements,
        location: mem_location,
        linear: false,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
      })
      .map_err(|e| format!("at allocating buffer mem: {e}"))?;
    Ok(Self { allocator, inner: Some(altn), name: name.to_string() })
  }

  pub fn inner(&self) -> Result<&Allocation, String> {
    self.inner.as_ref().ok_or(format!("no allocation found for {}", &self.name))
  }

  pub fn write_data(&mut self, offset: usize, bytes: &[u8]) -> Result<(), String> {
    self
      .inner
      .as_mut()
      .map(|alloc| {
        alloc
          .mapped_slice_mut()
          .map(|x| {
            x[offset..bytes.len()]
              .copy_from_slice(bytes)
          })
          .ok_or(format!("at mapping buffer {} 's memory", &self.name))
      })
      .ok_or(format!("no memory allocated for buffer {}", &self.name))??; // second ? for failure in mapped_slice_mut
    Ok(())
  }
}

impl Drop for AdAllocation {
  fn drop(&mut self) {
    let _ = self
      .allocator
      .lock()
      .map(|mut altr| self.inner.take().map(|altn| altr.free(altn)))
      .inspect_err(|e| eprintln!("at getting allocator lock to free allocation: {e}"));
  }
}

pub struct AdBuffer {
  inner: vk::Buffer,
  size: vk::DeviceSize,
  name: String,
  ash_device: Arc<AdAshDevice>,
  allocation: AdAllocation,
}

impl AdBuffer {
  pub fn new(
    ash_device: Arc<AdAshDevice>,
    allocator: Arc<Mutex<Allocator>>,
    mem_location: MemoryLocation,
    name: &str,
    flags: vk::BufferCreateFlags,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
  ) -> Result<AdBuffer, String> {
    unsafe {
      let buffer = ash_device
        .inner()
        .create_buffer(&vk::BufferCreateInfo::default().flags(flags).size(size).usage(usage), None)
        .map_err(|e| format!("at vk buffer create: {e}"))?;
      let allocation = AdAllocation::new(
        allocator,
        name,
        mem_location,
        ash_device.inner().get_buffer_memory_requirements(buffer)
      )?;
      ash_device
        .inner()
        .bind_buffer_memory(buffer, allocation.inner()?.memory(), allocation.inner()?.offset())
        .map_err(|e| format!("at buffer mem bind: {e}"))?;
      Ok(Self {
        inner: buffer,
        size,
        name: name.to_string(),
        ash_device,
        allocation
      })
    }
  }

  pub fn from_data<T>(
    ash_device: Arc<AdAshDevice>,
    allocator: Arc<Mutex<Allocator>>,
    mem_location: MemoryLocation,
    name: &str,
    flags: vk::BufferCreateFlags,
    usage: vk::BufferUsageFlags,
    struct_slice: &[T],
    cmd_buffer: &AdCommandBuffer
  ) -> Result<AdBuffer, String> {
    let data = Self::get_byte_slice(struct_slice);
    let mut stage_buffer = Self::new(
      ash_device.clone(),
      allocator.clone(),
      MemoryLocation::CpuToGpu,
      &format!("{name}_stage_buffer"),
      flags,
      data.len() as u64,
      vk::BufferUsageFlags::TRANSFER_SRC,
    )?;
    stage_buffer.write_data(0, data)?;

    let buffer = Self::new(
      ash_device.clone(),
      allocator.clone(),
      mem_location,
      name,
      flags,
      data.len() as u64,
      usage | vk::BufferUsageFlags::TRANSFER_DST,
    )?;
    cmd_buffer.begin(vk::CommandBufferUsageFlags::default())?;
    Self::add_copy_buffer_to_buffer_cmd(
      &cmd_buffer,
      &stage_buffer,
      &buffer,
      &[vk::BufferCopy{ src_offset: 0, dst_offset: 0, size: data.len() as u64 }]
    );
    cmd_buffer.end()?;

    let tmp_fence = AdFence::new(ash_device.clone(), vk::FenceCreateFlags::default())?;
    cmd_buffer.submit(&[], &[], Some(&tmp_fence))?;
    tmp_fence.wait(999999999)?;

    Ok(buffer)
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

  pub fn write_data<T>(&mut self, offset: usize, struct_slice: &[T]) -> Result<(), String> {
    let data = Self::get_byte_slice(struct_slice);
    if offset + data.len() > self.size as usize {
      return Err(format!("buffer {} only supports {} bytes", &self.name, self.size))
    }
    self
      .allocation
      .write_data(offset, data)
  }

  pub fn add_copy_buffer_to_buffer_cmd(
    cmd_buffer :&AdCommandBuffer,
    src_buffer: &AdBuffer,
    dst_buffer: &AdBuffer,
    regions: &[vk::BufferCopy]
  ) {
    unsafe {
      cmd_buffer
        .get_ash_device()
        .inner()
        .cmd_copy_buffer(cmd_buffer.inner(), src_buffer.inner(), dst_buffer.inner(), regions);
    }
  }

  pub fn get_byte_slice<T>(struct_slice: &[T]) -> &[u8] {
    unsafe {
      std::slice::from_raw_parts(
        struct_slice.as_ptr() as *const u8,
        std::mem::size_of::<T>() * struct_slice.len()
      )
    }
  }
}

impl Drop for AdBuffer {
  fn drop(&mut self) {
    unsafe {
      self.ash_device.inner().destroy_buffer(self.inner, None);
    }
  }
}

pub struct AdImage2D {
  inner: vk::Image,
  format: vk::Format,
  resolution: vk::Extent2D,
  name: String,
  ash_device: Arc<AdAshDevice>,
  allocator: Arc<Mutex<Allocator>>,
  allocation: Option<Allocation>,
}

impl AdImage2D {
  pub fn new(
    ash_device: Arc<AdAshDevice>,
    vk_image: vk::Image,
    name: &str,
    resolution: vk::Extent2D,
    format: vk::Format,
    allocator: Arc<Mutex<Allocator>>,
    allocation: Option<Allocation>
  ) -> Self {
    Self {
      ash_device,
      inner: vk_image,
      name: name.to_string(),
      resolution,
      format,
      allocator,
      allocation
    }
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
      vk::Offset3D::default()
        .x(self.resolution.width as i32)
        .y(self.resolution.height as i32)
        .z(1),
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
      self.ash_device.inner().create_image_view(&view_create_info, None)
        .map_err(|e| format!("at creating vk image view: {e}"))?
    };
    Ok(AdImageView {
      ash_device: self.ash_device.clone(),
      inner: image_view,
    })
  }
}

impl Drop for AdImage2D {
  fn drop(&mut self) {
    unsafe {
      self.ash_device.inner().destroy_image(self.inner, None);
    }
    let _ = self.allocator
      .lock()
      .map(|mut altr| self.allocation.take().map(|altn| altr.free(altn)))
      .inspect_err(|e| eprintln!("at getting allocator lock while image destroy: {e}"));
  }
}

pub struct AdImageView {
  ash_device: Arc<AdAshDevice>,
  inner: vk::ImageView,
}

impl AdImageView {
  pub fn inner(&self) -> vk::ImageView {
    self.inner
  }
}

impl Drop for AdImageView {
  fn drop(&mut self) {
    unsafe {
      self.ash_device.inner().destroy_image_view(self.inner, None);
    }
  }
}

#[derive(Clone)]
pub enum AdDescriptorBinding {
  StorageBuffer(Vec<Option<Arc<AdBuffer>>>),
  UniformBuffer(Vec<Option<Arc<AdBuffer>>>),
  Image2D(Vec<Option<Arc<AdImage2D>>>),
}

impl AdDescriptorBinding {
  pub fn get_descriptor_type(&self) -> vk::DescriptorType {
    match self {
      Self::StorageBuffer(_) => vk::DescriptorType::STORAGE_BUFFER,
      Self::UniformBuffer(_) => vk::DescriptorType::UNIFORM_BUFFER,
      Self::Image2D(_) => vk::DescriptorType::SAMPLED_IMAGE,
    }
  }

  pub fn get_descriptor_count(&self) -> u32 {
    match self {
      Self::StorageBuffer(x) => x.len() as u32,
      Self::UniformBuffer(x) => x.len() as u32,
      Self::Image2D(x) => x.len() as u32,
    }
  }

  pub fn drop_embedded(self) -> Self {
    match self {
      Self::StorageBuffer(x) => Self::StorageBuffer(vec![None; x.len()]),
      Self::UniformBuffer(x) => Self::UniformBuffer(vec![None; x.len()]),
      Self::Image2D(x) => Self::Image2D(vec![None; x.len()]),
    }
  }
}

pub struct AdDescriptorSetLayout {
  ash_device: Arc<AdAshDevice>,
  inner: vk::DescriptorSetLayout,
  empty_bindings: Vec<(vk::ShaderStageFlags, AdDescriptorBinding)>,
}

impl AdDescriptorSetLayout {
  pub fn new(
    ash_device: Arc<AdAshDevice>,
    bindings: &[(vk::ShaderStageFlags, AdDescriptorBinding)]
  ) -> Result<Self, String> {
    let empty_bindings = bindings
      .iter()
      .map(|x| { (x.0, x.1.clone().drop_embedded()) })
      .collect::<Vec<_>>();
    let vk_descriptor_bindings = bindings
      .iter()
      .enumerate()
      .map(|(i, binding)|{
        vk::DescriptorSetLayoutBinding::default()
          .binding(i as u32)
          .stage_flags(binding.0)
          .descriptor_type(binding.1.get_descriptor_type())
          .descriptor_count(binding.1.get_descriptor_count())
      })
      .collect::<Vec<_>>();
    let dsl_create_info = vk::DescriptorSetLayoutCreateInfo::default()
      .bindings(&vk_descriptor_bindings);
    unsafe {
      let descriptor_set_layout = ash_device
        .inner()
        .create_descriptor_set_layout(&dsl_create_info, None)
        .map_err(|e| format!("at creating vk descriptor set layout: {e}"))?;
      Ok(AdDescriptorSetLayout{ ash_device, inner: descriptor_set_layout, empty_bindings })
    }
  }

  pub fn get_bindings_info(&self) -> &[(vk::ShaderStageFlags, AdDescriptorBinding)] {
    &self.empty_bindings
  }
}

impl Drop for AdDescriptorSetLayout {
  fn drop(&mut self) {
    unsafe {
      self.ash_device.inner().destroy_descriptor_set_layout(self.inner, None);
    }
  }
}

pub struct AdDescriptorPool {
  ash_device: Arc<AdAshDevice>,
  inner: vk::DescriptorPool,
  free_supported: bool,
}

impl AdDescriptorPool {
  pub fn new(
    ash_device: Arc<AdAshDevice>,
    flags: vk::DescriptorPoolCreateFlags,
    max_sets: u32,
    pool_sizes: &[vk::DescriptorPoolSize]
  ) -> Result<Self, String> {
    unsafe {
      let descriptor_pool = ash_device.inner().create_descriptor_pool(
        &vk::DescriptorPoolCreateInfo::default()
          .flags(flags)
          .max_sets(max_sets)
          .pool_sizes(pool_sizes),
        None
      )
        .map_err(|e| format!("at creating vk descriptor pool: {e}"))?;
      Ok(Self {
        ash_device,
        inner: descriptor_pool,
        free_supported: flags.contains(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
      })
    }
  }

  pub fn can_free_sets(&self) -> bool {
    self.free_supported
  }
}

impl Drop for AdDescriptorPool {
  fn drop(&mut self) {
    unsafe {
      self.ash_device.inner().destroy_descriptor_pool(self.inner, None);
    }
  }
}

pub struct AdDescriptorSet {
  inner: vk::DescriptorSet,
  bindings: Vec<AdDescriptorBinding>,
  desc_pool: Arc<AdDescriptorPool>,
  desc_layout: Arc<AdDescriptorSetLayout>,
}

impl AdDescriptorSet {
  pub fn new(
    desc_pool: Arc<AdDescriptorPool>,
    desc_layouts: &[&Arc<AdDescriptorSetLayout>]
  ) -> Result<Vec<Self>, String> {
    unsafe {
      desc_pool
        .ash_device
        .inner()
        .allocate_descriptor_sets(
          &vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool.inner)
            .set_layouts(&desc_layouts.iter().map(|x| x.inner).collect::<Vec<_>>())
        )
        .map_err(|e| format!("at allocating vk dsets: {e}"))
        .map(|vk_dsets| {
          vk_dsets
            .iter()
            .enumerate()
            .map(|(i, vk_dset)| {
              Self {
                inner: *vk_dset,
                bindings: desc_layouts[i]
                  .get_bindings_info()
                  .iter()
                  .map(|x| x.1.clone())
                  .collect::<Vec<_>>(),
                desc_pool: desc_pool.clone(),
                desc_layout: desc_layouts[i].clone(),
              }
            })
            .collect::<Vec<_>>()
        })
    }    
  }
}

impl Drop for AdDescriptorSet {
  fn drop(&mut self) {
    unsafe {
      let _ = self
        .desc_pool
        .ash_device
        .inner()
        .free_descriptor_sets(self.desc_pool.inner, &[self.inner]);
    }
  }
}
