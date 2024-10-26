use std::sync::{Arc, Mutex};

use ash_context::gpu_allocator::{
  vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
  MemoryLocation,
};
use ash_context::{ash::vk, getset, AdAshDevice};
use ash_queue_wrappers::AdCommandBuffer;
use ash_sync_wrappers::AdFence;

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdAllocation {
  allocator: Arc<Mutex<Allocator>>,
  #[getset(get = "pub")]
  inner: Option<Allocation>,
  #[getset(get = "pub")]
  name: String,
}

impl AdAllocation {
  pub fn new(
    allocator: Arc<Mutex<Allocator>>,
    name: &str,
    mem_location: MemoryLocation,
    requirements: vk::MemoryRequirements,
  ) -> Result<Self, String> {
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

  pub fn write_data(&mut self, offset: usize, bytes: &[u8]) -> Result<(), String> {
    self
      .inner
      .as_mut()
      .map(|alloc| {
        alloc
          .mapped_slice_mut()
          .map(|x| x[offset..bytes.len()].copy_from_slice(bytes))
          .ok_or(format!("at mapping buffer {} 's memory", &self.name))
      })
      .ok_or(format!("no memory allocated for buffer {}", &self.name))??; // second ? for failure in mapped_slice_mut
    Ok(())
  }

  pub fn rename(&mut self, name: &str) -> Result<(), String> {
    let curr_allocation = self.inner.as_mut().ok_or(format!("memory not allocated to rename"))?;
    self
      .allocator
      .lock()
      .as_mut()
      .map_err(|e| format!("at getting lock for mem allocator: {e}"))?
      .rename_allocation(curr_allocation, name)
      .map_err(|e| format!("at renaming mem allocation: {e}"))?;
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

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdBuffer {
  #[getset(get_copy = "pub")]
  inner: vk::Buffer,
  #[getset(get_copy = "pub")]
  size: vk::DeviceSize,
  #[getset(get = "pub")]
  name: String,
  ash_device: Arc<AdAshDevice>,
  #[getset(get = "pub")]
  allocation: Mutex<AdAllocation>,
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
        ash_device.inner().get_buffer_memory_requirements(buffer),
      )?;
      ash_device
        .inner()
        .bind_buffer_memory(
          buffer,
          allocation.inner().as_ref().ok_or("no allocation".to_string())?.memory(),
          allocation.inner().as_ref().ok_or("no allocation".to_string())?.offset(),
        )
        .map_err(|e| format!("at buffer mem bind: {e}"))?;
      Ok(Self {
        inner: buffer,
        size,
        name: name.to_string(),
        ash_device,
        allocation: Mutex::new(allocation),
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
    cmd_buffer: &AdCommandBuffer,
  ) -> Result<AdBuffer, String> {
    let data = Self::get_byte_slice(struct_slice);
    let stage_buffer = Self::new(
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
    cmd_buffer.copy_buffer_to_buffer_cmd(
      stage_buffer.inner(),
      buffer.inner(),
      &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: data.len() as u64 }],
    );
    cmd_buffer.end()?;

    let tmp_fence = AdFence::new(ash_device.clone(), vk::FenceCreateFlags::default())?;
    cmd_buffer.submit(&[], &[], Some(&tmp_fence))?;
    tmp_fence.wait(999999999)?;

    Ok(buffer)
  }

  pub fn write_data<T>(&self, offset: usize, struct_slice: &[T]) -> Result<(), String> {
    let data = Self::get_byte_slice(struct_slice);
    if offset + data.len() > self.size as usize {
      return Err(format!("buffer {} only supports {} bytes", &self.name, self.size));
    }
    self
      .allocation
      .lock()
      .map_err(|e| format!("at getting lock for buffer mem allocation: {e}"))?
      .write_data(offset, data)
  }

  pub fn get_byte_slice<T>(struct_slice: &[T]) -> &[u8] {
    unsafe {
      std::slice::from_raw_parts(
        struct_slice.as_ptr() as *const u8,
        std::mem::size_of::<T>() * struct_slice.len(),
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

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdImage {
  #[getset(get_copy = "pub")]
  inner: vk::Image,
  #[getset(get_copy = "pub")]
  itype: vk::ImageType,
  #[getset(get_copy = "pub")]
  format: vk::Format,
  #[getset(get_copy = "pub")]
  resolution: vk::Extent3D,
  #[getset(get = "pub")]
  name: String,
  ash_device: Arc<AdAshDevice>,
  #[getset(get = "pub")]
  allocation: Mutex<AdAllocation>,
}

impl AdImage {
  pub fn new_2d(
    ash_device: Arc<AdAshDevice>,
    allocator: Arc<Mutex<Allocator>>,
    mem_location: MemoryLocation,
    name: &str,
    format: vk::Format,
    resolution: vk::Extent2D,
    usage: vk::ImageUsageFlags,
    samples: vk::SampleCountFlags,
    mip_levels: u32,
  ) -> Result<Arc<Self>, String> {
    unsafe {
      let vk_image = ash_device
        .inner()
        .create_image(
          &vk::ImageCreateInfo::default()
            .usage(usage)
            .format(format)
            .extent(vk::Extent3D::from(resolution).depth(1))
            .samples(samples)
            .mip_levels(mip_levels)
            .image_type(vk::ImageType::TYPE_2D)
            .array_layers(1),
          None,
        )
        .map_err(|e| format!("at vk image create: {e}"))?;
      let allocation = AdAllocation::new(
        allocator,
        name,
        mem_location,
        ash_device.inner().get_image_memory_requirements(vk_image),
      )?;
      ash_device
        .inner()
        .bind_image_memory(
          vk_image,
          allocation.inner().as_ref().ok_or("mem not allocated")?.memory(),
          allocation.inner().as_ref().ok_or("mem not allocated")?.offset(),
        )
        .map_err(|e| format!("at image mem bind: {e}"))?;
      Ok(Arc::new(Self {
        ash_device,
        inner: vk_image,
        itype: vk::ImageType::TYPE_2D,
        name: name.to_string(),
        resolution: vk::Extent3D::default()
          .width(resolution.width)
          .height(resolution.height)
          .depth(1),
        format,
        allocation: Mutex::new(allocation),
      }))
    }
  }

  pub fn new_2d_from_file(
    ash_device: Arc<AdAshDevice>,
    allocator: Arc<Mutex<Allocator>>,
    mem_location: MemoryLocation,
    name: &str,
    usage: vk::ImageUsageFlags,
    file_path: &str,
    cmd_buffer: &AdCommandBuffer,
    init_layout: vk::ImageLayout,
  ) -> Result<Arc<Self>, String> {
    let image_info = image::open(file_path).map_err(|e| format!("at loading file: {e}"))?;
    let image_rgba8 = image_info.to_rgba8();

    let stage_buffer = AdBuffer::new(
      ash_device.clone(),
      allocator.clone(),
      MemoryLocation::CpuToGpu,
      &format!("{name}_stage_buffer"),
      vk::BufferCreateFlags::default(),
      image_rgba8.len() as vk::DeviceSize,
      vk::BufferUsageFlags::TRANSFER_SRC,
    )
      .map_err(|e| format!("at stage buffer create:: {e}"))?;
    stage_buffer.write_data(0, &image_rgba8)?;

    let image_2d = AdImage::new_2d(
      ash_device.clone(),
      allocator,
      mem_location,
      name,
      vk::Format::R8G8B8A8_SRGB,
      vk::Extent2D::default().width(image_info.width()).height(image_info.height()),
      vk::ImageUsageFlags::TRANSFER_DST | usage,
      vk::SampleCountFlags::TYPE_1,
      1,
    )?;

    cmd_buffer.begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;

    cmd_buffer.pipeline_barrier(
      vk::PipelineStageFlags::ALL_COMMANDS,
      vk::PipelineStageFlags::ALL_COMMANDS,
      vk::DependencyFlags::BY_REGION,
      &[],
      &[],
      &[
        vk::ImageMemoryBarrier::default()
          .image(image_2d.inner)
          .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).base_array_layer(0).layer_count(1).base_mip_level(0).level_count(1))
          .src_queue_family_index(cmd_buffer.cmd_pool().queue().family_index())
          .dst_queue_family_index(cmd_buffer.cmd_pool().queue().family_index())
          .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
          .old_layout(vk::ImageLayout::UNDEFINED)
          .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
          .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
      ]
    );
    cmd_buffer.copy_buffer_to_image(
      stage_buffer.inner(),
      image_2d.inner,
      vk::ImageLayout::TRANSFER_DST_OPTIMAL,
      &[vk::BufferImageCopy::default()
        .image_offset(vk::Offset3D::default())
        .image_extent(image_2d.resolution())
        .image_subresource(vk::ImageSubresourceLayers::default()
          .aspect_mask(vk::ImageAspectFlags::COLOR)
          .base_array_layer(0)
          .layer_count(1)
          .mip_level(0)
        )]
    );
    cmd_buffer.pipeline_barrier(
      vk::PipelineStageFlags::ALL_COMMANDS,
      vk::PipelineStageFlags::ALL_COMMANDS,
      vk::DependencyFlags::BY_REGION,
      &[],
      &[],
      &[
        vk::ImageMemoryBarrier::default()
          .image(image_2d.inner)
          .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).base_array_layer(0).layer_count(1).base_mip_level(0).level_count(1))
          .src_queue_family_index(cmd_buffer.cmd_pool().queue().family_index())
          .dst_queue_family_index(cmd_buffer.cmd_pool().queue().family_index())
          .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
          .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
          .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
          .new_layout(init_layout)
      ]
    );

    cmd_buffer.end()?;
    let fence = AdFence::new(ash_device.clone(), vk::FenceCreateFlags::empty())?;
    cmd_buffer.submit(&[], &[], Some(&fence))?;
    fence.wait(999999999)?;
    Ok(image_2d)
  }

  pub fn full_range_offset_3d(&self) -> [vk::Offset3D; 2] {
    [
      vk::Offset3D::default(),
      vk::Offset3D::default()
        .x(self.resolution.width as i32)
        .y(self.resolution.height as i32)
        .z(self.resolution.depth as i32),
    ]
  }
}

impl Drop for AdImage {
  fn drop(&mut self) {
    unsafe {
      self.ash_device.inner().destroy_image(self.inner, None);
    }
  }
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdImageView {
  ash_device: Arc<AdAshDevice>,
  #[getset(get = "pub")]
  image: Arc<AdImage>,
  #[getset(get_copy = "pub")]
  view_type: vk::ImageViewType,
  #[getset(get_copy = "pub")]
  inner: vk::ImageView,
}

impl AdImageView {
  pub fn create_view(
    image: Arc<AdImage>,
    view_type: vk::ImageViewType,
    subresource_range: vk::ImageSubresourceRange,
  ) -> Result<Arc<AdImageView>, String> {
    // Check view type support
    if (view_type == vk::ImageViewType::TYPE_1D && image.itype() != vk::ImageType::TYPE_1D)
      || (view_type == vk::ImageViewType::TYPE_1D_ARRAY && image.itype() != vk::ImageType::TYPE_1D)
      || (view_type == vk::ImageViewType::TYPE_2D
        && (image.itype() != vk::ImageType::TYPE_2D && image.itype() != vk::ImageType::TYPE_3D))
      || (view_type == vk::ImageViewType::TYPE_2D_ARRAY
        && (image.itype() != vk::ImageType::TYPE_1D && image.itype() != vk::ImageType::TYPE_3D))
      || (view_type == vk::ImageViewType::CUBE && image.itype() != vk::ImageType::TYPE_2D)
      || (view_type == vk::ImageViewType::CUBE_ARRAY && image.itype() != vk::ImageType::TYPE_2D)
      || (view_type == vk::ImageViewType::TYPE_3D && image.itype() != vk::ImageType::TYPE_3D)
    {
      return Err("unsupported view type".to_string());
    }

    let view_create_info = vk::ImageViewCreateInfo::default()
      .image(image.inner())
      .format(image.format())
      .view_type(view_type)
      .subresource_range(subresource_range)
      .components(vk::ComponentMapping {
        r: vk::ComponentSwizzle::R,
        g: vk::ComponentSwizzle::G,
        b: vk::ComponentSwizzle::B,
        a: vk::ComponentSwizzle::A,
      });
    let image_view = unsafe {
      image
        .ash_device
        .inner()
        .create_image_view(&view_create_info, None)
        .map_err(|e| format!("at creating vk image view: {e}"))?
    };
    Ok(Arc::new(AdImageView {
      ash_device: image.ash_device.clone(),
      inner: image_view,
      image,
      view_type,
    }))
  }
}

impl Drop for AdImageView {
  fn drop(&mut self) {
    unsafe {
      self.ash_device.inner().destroy_image_view(self.inner, None);
    }
  }
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdSampler {
  ash_device: Arc<AdAshDevice>,
  #[getset(get_copy = "pub")]
  inner: vk::Sampler,
}

impl AdSampler {
  pub fn new(ash_device: Arc<AdAshDevice>) -> Result<Self, String> {
    unsafe {
      let vk_sampler = ash_device
        .inner()
        .create_sampler(&vk::SamplerCreateInfo::default(), None)
        .map_err(|e| format!("at vk sampler create: {e}"))?;
      Ok(Self { ash_device, inner: vk_sampler })
    }
  }
}

impl Drop for AdSampler {
  fn drop(&mut self) {
    unsafe {
      self.ash_device.inner().destroy_sampler(self.inner, None);
    }
  }
}

#[derive(Clone)]
pub enum AdDescriptorBinding {
  StorageBuffer(Option<Arc<AdBuffer>>),
  UniformBuffer(Option<Arc<AdBuffer>>),
  Image2D(Option<(Arc<AdImageView>, vk::ImageLayout)>),
  Sampler2D(Option<(Arc<AdImageView>, vk::ImageLayout, Arc<AdSampler>)>),
}

impl AdDescriptorBinding {
  pub fn get_descriptor_type(&self) -> vk::DescriptorType {
    match self {
      Self::StorageBuffer(_) => vk::DescriptorType::STORAGE_BUFFER,
      Self::UniformBuffer(_) => vk::DescriptorType::UNIFORM_BUFFER,
      Self::Image2D(_) => vk::DescriptorType::SAMPLED_IMAGE,
      Self::Sampler2D(_) => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
    }
  }

  pub fn get_descriptor_info(
    &self,
  ) -> (Option<vk::DescriptorBufferInfo>, Option<vk::DescriptorImageInfo>) {
    match self {
      AdDescriptorBinding::StorageBuffer(v) => {
        let buffer_info = v
          .as_ref()
          .map(|b| {
            vk::DescriptorBufferInfo::default().buffer(b.inner()).offset(0).range(b.size())
          });
        (buffer_info, None)
      }
      AdDescriptorBinding::UniformBuffer(v) => {
        let buffer_info = v
          .as_ref()
          .map(|b| {
            vk::DescriptorBufferInfo::default().buffer(b.inner()).offset(0).range(b.size())
          });
        (buffer_info, None)
      }
      AdDescriptorBinding::Image2D(v) => {
        let image_info = v
          .as_ref()
          .map(|id| {
            vk::DescriptorImageInfo::default().image_view(id.0.inner()).image_layout(id.1)
          });
        (None, image_info)
      }
      AdDescriptorBinding::Sampler2D(v) => {
        let image_info = v
          .as_ref()
          .map(|id| {
            vk::DescriptorImageInfo::default().sampler(id.2.inner()).image_view(id.0.inner()).image_layout(id.1)
          });
        (None, image_info)
      }
    }
  }

  pub fn drop_embedded(self) -> Self {
    match self {
      Self::StorageBuffer(_x) => Self::StorageBuffer(None),
      Self::UniformBuffer(_x) => Self::UniformBuffer(None),
      Self::Image2D(_x) => Self::Image2D(None),
      Self::Sampler2D(_x) => Self::Sampler2D(None),
    }
  }
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdDescriptorSetLayout {
  ash_device: Arc<AdAshDevice>,
  #[getset(get_copy = "pub")]
  inner: vk::DescriptorSetLayout,
  #[getset(get = "pub")]
  empty_bindings: Vec<(vk::ShaderStageFlags, AdDescriptorBinding)>,
}

impl AdDescriptorSetLayout {
  pub fn new(
    ash_device: Arc<AdAshDevice>,
    bindings: &[(vk::ShaderStageFlags, AdDescriptorBinding)],
  ) -> Result<Self, String> {
    let empty_bindings =
      bindings.iter().map(|x| (x.0, x.1.clone().drop_embedded())).collect::<Vec<_>>();
    let vk_descriptor_bindings = bindings
      .iter()
      .enumerate()
      .map(|(i, binding)| {
        vk::DescriptorSetLayoutBinding::default()
          .binding(i as u32)
          .stage_flags(binding.0)
          .descriptor_type(binding.1.get_descriptor_type())
          .descriptor_count(1)
      })
      .collect::<Vec<_>>();
    let dsl_create_info =
      vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_descriptor_bindings);
    unsafe {
      let descriptor_set_layout = ash_device
        .inner()
        .create_descriptor_set_layout(&dsl_create_info, None)
        .map_err(|e| format!("at creating vk descriptor set layout: {e}"))?;
      Ok(AdDescriptorSetLayout { ash_device, inner: descriptor_set_layout, empty_bindings })
    }
  }

  pub fn new_sparse(ash_device: Arc<AdAshDevice>, bindings: &[(u32, vk::ShaderStageFlags, AdDescriptorBinding)]) -> Result<Self, String> {
    let empty_bindings =
      bindings.iter().map(|x| (x.1, x.2.clone().drop_embedded())).collect::<Vec<_>>();
    let vk_descriptor_bindings = bindings
      .iter()
      .map(|binding| {
        vk::DescriptorSetLayoutBinding::default()
          .binding(binding.0)
          .stage_flags(binding.1)
          .descriptor_type(binding.2.get_descriptor_type())
          .descriptor_count(1)
      })
      .collect::<Vec<_>>();
    let dsl_create_info =
      vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_descriptor_bindings);
    unsafe {
      let descriptor_set_layout = ash_device
        .inner()
        .create_descriptor_set_layout(&dsl_create_info, None)
        .map_err(|e| format!("at creating vk descriptor set layout: {e}"))?;
      Ok(AdDescriptorSetLayout { ash_device, inner: descriptor_set_layout, empty_bindings })
    }
  }
}

impl Drop for AdDescriptorSetLayout {
  fn drop(&mut self) {
    unsafe {
      self.ash_device.inner().destroy_descriptor_set_layout(self.inner, None);
    }
  }
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdDescriptorPool {
  ash_device: Arc<AdAshDevice>,
  inner: vk::DescriptorPool,
  #[getset(get_copy = "pub")]
  free_supported: bool,
}

impl AdDescriptorPool {
  pub fn new(
    ash_device: Arc<AdAshDevice>,
    flags: vk::DescriptorPoolCreateFlags,
    max_sets: u32,
    pool_sizes: &[vk::DescriptorPoolSize],
  ) -> Result<Self, String> {
    unsafe {
      let descriptor_pool = ash_device
        .inner()
        .create_descriptor_pool(
          &vk::DescriptorPoolCreateInfo::default()
            .flags(flags)
            .max_sets(max_sets)
            .pool_sizes(pool_sizes),
          None,
        )
        .map_err(|e| format!("at creating vk descriptor pool: {e}"))?;
      Ok(Self {
        ash_device,
        inner: descriptor_pool,
        free_supported: flags.contains(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET),
      })
    }
  }
}

impl Drop for AdDescriptorPool {
  fn drop(&mut self) {
    unsafe {
      self.ash_device.inner().destroy_descriptor_pool(self.inner, None);
    }
  }
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdDescriptorSet {
  #[getset(get_copy = "pub")]
  inner: vk::DescriptorSet,
  #[getset(get = "pub")]
  bindings: Vec<AdDescriptorBinding>,
  #[getset(get = "pub")]
  desc_pool: Arc<AdDescriptorPool>,
  #[getset(get = "pub")]
  desc_layout: Arc<AdDescriptorSetLayout>,
}

impl AdDescriptorSet {
  pub fn new(
    desc_pool: Arc<AdDescriptorPool>,
    desc_layouts: &[&Arc<AdDescriptorSetLayout>],
  ) -> Result<Vec<Self>, String> {
    unsafe {
      desc_pool
        .ash_device
        .inner()
        .allocate_descriptor_sets(
          &vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool.inner)
            .set_layouts(&desc_layouts.iter().map(|x| x.inner).collect::<Vec<_>>()),
        )
        .map_err(|e| format!("at allocating vk dsets: {e}"))
        .map(|vk_dsets| {
          vk_dsets
            .iter()
            .enumerate()
            .map(|(i, vk_dset)| Self {
              inner: *vk_dset,
              bindings: desc_layouts[i]
                .empty_bindings()
                .iter()
                .map(|x| x.1.clone())
                .collect::<Vec<_>>(),
              desc_pool: desc_pool.clone(),
              desc_layout: desc_layouts[i].clone(),
            })
            .collect::<Vec<_>>()
        })
    }
  }

  pub fn set_binding(&mut self, binding_id: u32, binding: AdDescriptorBinding) {
    let (buffer_info, image_info) = binding.get_descriptor_info();
    let buffer_info = buffer_info.map(|x| vec![x]).unwrap_or(vec![]);
    let image_info = image_info.map(|x| vec![x]).unwrap_or(vec![]);
    let mut write_info = vk::WriteDescriptorSet::default()
      .dst_set(self.inner)
      .dst_binding(binding_id)
      .descriptor_type(binding.get_descriptor_type())
      .descriptor_count(1);

    if buffer_info.len() > 0 {
      write_info = write_info.buffer_info(&buffer_info);
    }
    if image_info.len() > 0 {
      write_info = write_info.image_info(&image_info);
    }
    unsafe {
      self.desc_pool.ash_device.inner().update_descriptor_sets(&[write_info], &[]);
    }
    self.bindings[binding_id as usize] = binding;
  }
}

impl Drop for AdDescriptorSet {
  fn drop(&mut self) {
    unsafe {
      if self.desc_pool.free_supported() {
        let _ = self
          .desc_pool
          .ash_device
          .inner()
          .free_descriptor_sets(self.desc_pool.inner, &[self.inner]);
      }
    }
  }
}
