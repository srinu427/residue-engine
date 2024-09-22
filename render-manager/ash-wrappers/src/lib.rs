pub mod ad_wrappers;
pub mod builders;
mod init_helpers;

pub use ash::{ext, khr, vk};
use gpu_allocator::vulkan::{
  AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
pub use gpu_allocator::MemoryLocation;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
pub use raw_window_handle;
use spirv_cross::{spirv, glsl};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::ad_wrappers::AdQueue;
use ad_wrappers::data_wrappers::{AdBuffer, AdImage2D};
use ad_wrappers::sync_wrappers::{AdFence, AdSemaphore};
use ad_wrappers::{AdCommandBuffer, AdCommandPool, AdDescriptorPool, AdDescriptorSetLayout, AdShaderModule, AdSurface, AdSwapchain};
use builders::AdRenderPassBuilder;

pub struct VkInstances {
  surface_instance: Arc<khr::surface::Instance>,
  #[cfg(debug_assertions)]
  dbg_utils_instance: ext::debug_utils::Instance,
  vk_instance: ash::Instance,
  entry: ash::Entry,
}

impl VkInstances {
  pub fn new() -> Result<Self, String> {
    unsafe {
      let entry = ash::Entry::load().map_err(|e| format!("at VK load: {e}"))?;
      let vk_instance = init_helpers::init_instance(&entry, vec![], vec![])?;
      #[cfg(debug_assertions)]
      let dbg_utils_instance = ext::debug_utils::Instance::new(&entry, &vk_instance);
      let surface_instance = Arc::new(khr::surface::Instance::new(&entry, &vk_instance));

      Ok(Self {
        surface_instance,
        #[cfg(debug_assertions)]
        dbg_utils_instance,
        vk_instance,
        entry,
      })
    }
  }

  pub fn make_surface(&self, window: &(impl HasWindowHandle + HasDisplayHandle))
    -> Result<AdSurface, String> {
    unsafe {
      ash_window::create_surface(
        &self.entry,
        &self.vk_instance,
        window.display_handle().map_err(|_| "unsupported window".to_string())?.as_raw(),
        window.window_handle().map_err(|_| "unsupported window".to_string())?.as_raw(),
        None,
      )
      .map(|x| AdSurface { surface_instance: Arc::clone(&self.surface_instance), inner: x })
      .map_err(|e| format!("at surface create: {e}"))
    }
  }
}

impl Drop for VkInstances {
  fn drop(&mut self) {
    unsafe { self.vk_instance.destroy_instance(None); }
  }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone)]
pub enum GPUQueueType {
  Graphics,
  Compute,
  Transfer,
  Present,
}

pub struct VkContext {
  swapchain_device: Arc<khr::swapchain::Device>,
  pub queues: HashMap<GPUQueueType, Arc<AdQueue>>,
  vk_device: Arc<ash::Device>,
  pub gpu: vk::PhysicalDevice,
  #[cfg(debug_assertions)]
  dbg_utils_messenger: vk::DebugUtilsMessengerEXT,
  vk_instances: Arc<VkInstances>,
}

impl VkContext {
  pub fn new(vk_instances: Arc<VkInstances>, surface: &AdSurface) -> Result<Self, String> {
    unsafe {
      #[cfg(debug_assertions)]
      let dbg_utils_messenger = vk_instances
        .dbg_utils_instance
        .create_debug_utils_messenger(&init_helpers::make_debug_mgr_create_info(), None)
        .map_err(|e| format!("at dbg messenger init: {e}"))?;

      let gpu = vk_instances
        .vk_instance
        .enumerate_physical_devices()
        .map_err(|e| format!("can't get GPU list: {e}"))?
        .iter()
        .next()
        .cloned()
        .ok_or("no GPUs found".to_string())?;

      let q_indices = init_helpers::select_gpu_queues(
        &vk_instances.vk_instance,
        gpu,
        &vk_instances.surface_instance,
        surface.inner,
      )
      .ok_or("can't find needed queues".to_string())?;

      let device_extensions = vec![
        khr::swapchain::NAME.as_ptr(),
        #[cfg(target_os = "macos")]
        khr::portability_subset::NAME.as_ptr(),
      ];

      let (vk_device, mut queues) = init_helpers::create_device_and_queues(
        &vk_instances.vk_instance,
        gpu,
        device_extensions,
        vk::PhysicalDeviceFeatures::default(),
        q_indices,
      )?;

      let swapchain_device =
        Arc::new(khr::swapchain::Device::new(&vk_instances.vk_instance, &vk_device));

      let g_queue = Arc::new(queues.remove(0));
      let c_queue = Arc::new(queues.remove(0));
      let t_queue = Arc::new(queues.remove(0));
      let p_queue = Arc::new(queues.remove(0));

      Ok(Self {
        vk_instances,
        #[cfg(debug_assertions)]
        dbg_utils_messenger,
        gpu,
        vk_device,
        queues: HashMap::from([
          (GPUQueueType::Graphics, g_queue),
          (GPUQueueType::Compute, c_queue),
          (GPUQueueType::Transfer, t_queue),
          (GPUQueueType::Present, p_queue),
        ]),
        swapchain_device,
      })
    }
  }

  pub fn get_vk_device(&self) -> Arc<ash::Device> {
    self.vk_device.clone()
  }

  pub fn create_ad_swapchain(
    &self,
    surface: Arc<AdSurface>,
    image_count: u32,
    color_space: vk::ColorSpaceKHR,
    format: vk::Format,
    resolution: vk::Extent2D,
    usage: vk::ImageUsageFlags,
    pre_transform: vk::SurfaceTransformFlagsKHR,
    present_mode: vk::PresentModeKHR,
    old_swapchain: Option<AdSwapchain>,
  ) -> Result<AdSwapchain, String> {
    let swapchain_info = vk::SwapchainCreateInfoKHR::default()
      .surface(surface.inner)
      .old_swapchain(old_swapchain.as_ref().map(|x| x.inner).unwrap_or(vk::SwapchainKHR::null()))
      .min_image_count(image_count)
      .image_color_space(color_space)
      .image_format(format)
      .image_extent(resolution)
      .image_usage(usage)
      .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
      .pre_transform(pre_transform)
      .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
      .present_mode(present_mode)
      .clipped(true)
      .image_array_layers(1);

    unsafe {
      let swapchain = self
        .swapchain_device
        .create_swapchain(&swapchain_info, None)
        .map_err(|e| format!("at vk swapchain create: {e}"))?;
      let images = self
        .swapchain_device
        .get_swapchain_images(swapchain)
        .map_err(|e| format!("at getting swapchain images: {e}"))?;
      Ok(AdSwapchain {
        swapchain_device: Arc::clone(&self.swapchain_device),
        surface,
        gpu: self.gpu,
        present_queue: Arc::clone(&self.queues[&GPUQueueType::Present]),
        inner: swapchain,
        images,
        image_count,
        color_space,
        format,
        resolution,
        usage,
        pre_transform,
        present_mode,
        initialized: false,
      })
    }
  }

  pub fn create_ad_semaphore(&self, flags: vk::SemaphoreCreateFlags)
    -> Result<AdSemaphore, String> {
    unsafe {
      let semaphore = self
        .vk_device
        .create_semaphore(&vk::SemaphoreCreateInfo::default().flags(flags), None)
        .map_err(|e| format!("at create vk semaphore: {e}"))?;
      Ok(AdSemaphore { vk_device: Arc::clone(&self.vk_device), inner: semaphore })
    }
  }

  pub fn create_ad_fence(&self, flags: vk::FenceCreateFlags) -> Result<AdFence, String> {
    unsafe {
      let fence = self
        .vk_device
        .create_fence(&vk::FenceCreateInfo::default().flags(flags), None)
        .map_err(|e| format!("at create vk semaphore: {e}"))?;
      Ok(AdFence { vk_device: Arc::clone(&self.vk_device), inner: fence })
    }
  }

  pub fn create_ad_buffer(
    &self,
    allocator: Arc<Mutex<Allocator>>,
    mem_location: MemoryLocation,
    name: &str,
    flags: vk::BufferCreateFlags,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
  ) -> Result<AdBuffer, String> {
    unsafe {
      let buffer = self
        .vk_device
        .create_buffer(&vk::BufferCreateInfo::default().flags(flags).size(size).usage(usage), None)
        .map_err(|e| format!("at vk buffer create: {e}"))?;
      let allocation = allocator
        .lock()
        .map_err(|e| format!("at getting allocator lock: {e}"))?
        .allocate(&AllocationCreateDesc {
          name,
          requirements: self.vk_device.get_buffer_memory_requirements(buffer),
          location: mem_location,
          linear: false,
          allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("at allocating buffer mem: {e}"))?;
      self
        .vk_device
        .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        .map_err(|e| format!("at buffer mem bind: {e}"))?;
      Ok(AdBuffer {
        inner: buffer,
        size,
        name: name.to_string(),
        vk_device: Arc::clone(&self.vk_device),
        allocator,
        allocation: Some(allocation),
      })
    }
  }

  pub fn create_ad_buffer_from_data(
    &self,
    allocator: Arc<Mutex<Allocator>>,
    mem_location: MemoryLocation,
    name: &str,
    flags: vk::BufferCreateFlags,
    usage: vk::BufferUsageFlags,
    data: &[u8],
    cmd_buffer: &AdCommandBuffer
  ) -> Result<AdBuffer, String> {
    let mut stage_buffer = self.create_ad_buffer(
      Arc::clone(&allocator),
      MemoryLocation::CpuToGpu,
      &format!("{name}_stage_buffer"),
      flags,
      data.len() as u64,
      vk::BufferUsageFlags::TRANSFER_SRC,
    )?;
    stage_buffer
      .allocation
      .as_mut()
      .map(|alloc| {alloc.mapped_slice_mut().map(|x| {x[..data.len()].copy_from_slice(data)})})
      .ok_or("Error allocating stage buffer")?;
    let buffer = self.create_ad_buffer(
      Arc::clone(&allocator),
      mem_location,
      name,
      flags,
      data.len() as u64,
      usage | vk::BufferUsageFlags::TRANSFER_DST,
    )?;
    cmd_buffer.begin(vk::CommandBufferBeginInfo::default())?;
    cmd_buffer.copy_buffer_to_buffer(
      stage_buffer.inner,
      buffer.inner,
      &[vk::BufferCopy{ src_offset: 0, dst_offset: 0, size: data.len() as u64 }]
    );
    cmd_buffer.end()?;

    let tmp_fence = self.create_ad_fence(vk::FenceCreateFlags::default())?;
    cmd_buffer.submit(&[], &[], Some(&tmp_fence))?;
    tmp_fence.wait(999999999)?;

    Ok(buffer)
  }

  pub fn create_allocator(&self) -> Result<Allocator, String> {
    Allocator::new(&AllocatorCreateDesc {
      instance: self.vk_instances.vk_instance.clone(),
      device: (*self.vk_device).clone(),
      physical_device: self.gpu,
      debug_settings: Default::default(),
      buffer_device_address: false,
      allocation_sizes: Default::default(),
    })
    .map_err(|e| format!("at gpu mem allocator create: {e}"))
  }

  pub fn create_ad_image_2d(
    &self,
    allocator: Arc<Mutex<Allocator>>,
    mem_location: MemoryLocation,
    name: &str,
    format: vk::Format,
    resolution: vk::Extent2D,
    usage: vk::ImageUsageFlags,
    samples: vk::SampleCountFlags,
    mip_levels: u32,
  ) -> Result<AdImage2D, String> {
    unsafe {
      let image = self
        .vk_device
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
      let allocation = allocator
        .lock()
        .map_err(|e| format!("at getting allocator lock: {e}"))?
        .allocate(&AllocationCreateDesc {
          name,
          requirements: self.vk_device.get_image_memory_requirements(image),
          location: mem_location,
          linear: false,
          allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("at allocating image mem: {e}"))?;
      self
        .vk_device
        .bind_image_memory(image, allocation.memory(), allocation.offset())
        .map_err(|e| format!("at image mem bind: {e}"))?;

      Ok(AdImage2D {
        inner: image,
        format,
        resolution,
        name: name.to_string(),
        vk_device: Arc::clone(&self.vk_device),
        allocator: Some(allocator),
        allocation: Some(allocation),
      })
    }
  }

  pub fn create_ad_image_2d_from_file(
    &self,
    allocator: Arc<Mutex<Allocator>>,
    mem_location: MemoryLocation,
    transfer_cmd_pool: &AdCommandPool,
    name: &str,
    format: vk::Format,
    file_path: &Path,
    usage: vk::ImageUsageFlags,
    samples: vk::SampleCountFlags,
    mip_levels: u32,
  ) -> Result<AdImage2D, String> {
    let image_info = image::open(file_path).map_err(|e| format!("at loading file: {e}"))?;
    let image_rgba8 = image_info.to_rgba8();

    let mut stage_buffer = self
      .create_ad_buffer(
        Arc::clone(&allocator),
        MemoryLocation::CpuToGpu,
        &format!("{name}_stage_buffer"),
        vk::BufferCreateFlags::default(),
        image_rgba8.len() as vk::DeviceSize,
        vk::BufferUsageFlags::TRANSFER_SRC,
      )
      .map_err(|e| format!("at stage buffer create:: {e}"))?;

    stage_buffer
      .allocation
      .as_mut()
      .ok_or("stage buffer not allocated, hmmm".to_string())?
      .mapped_slice_mut()
      .ok_or("at mapping stage buffer memory to CPU".to_string())?
      .copy_from_slice(image_rgba8.as_raw().as_slice());

    let image_2d = self.create_ad_image_2d(
      allocator,
      mem_location,
      name,
      format,
      vk::Extent2D::default().width(image_info.width()).height(image_info.height()),
      vk::ImageUsageFlags::TRANSFER_DST | usage,
      samples,
      mip_levels,
    )?;

    let cmd_buffer = transfer_cmd_pool
      .allocate_command_buffers(vk::CommandBufferLevel::PRIMARY, 1)?
      .swap_remove(0);
    let upload_fence = self.create_ad_fence(vk::FenceCreateFlags::default())?;

    cmd_buffer.begin(vk::CommandBufferBeginInfo::default())?;

    cmd_buffer.pipeline_barrier(
      vk::PipelineStageFlags::BOTTOM_OF_PIPE,
      vk::PipelineStageFlags::TRANSFER,
      vk::DependencyFlags::BY_REGION,
      &[],
      &[],
      &[vk::ImageMemoryBarrier::default()
        .image(image_2d.inner)
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1),
        )
        .src_access_mask(vk::AccessFlags::NONE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(self.queues[&GPUQueueType::Transfer].qf_idx)
        .dst_queue_family_index(self.queues[&GPUQueueType::Transfer].qf_idx)],
    );
    cmd_buffer.copy_buffer_to_image(
      stage_buffer.inner,
      image_2d.inner,
      vk::ImageLayout::TRANSFER_DST_OPTIMAL,
      &[vk::BufferImageCopy::default()
        .image_subresource(
          vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1),
        )
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(
          vk::Extent3D::default().width(image_info.width()).height(image_info.height()).depth(1),
        )],
    );
    cmd_buffer.pipeline_barrier(
      vk::PipelineStageFlags::TRANSFER,
      vk::PipelineStageFlags::TRANSFER,
      vk::DependencyFlags::BY_REGION,
      &[],
      &[],
      &[vk::ImageMemoryBarrier::default()
        .image(image_2d.inner)
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1),
        )
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .src_queue_family_index(self.queues[&GPUQueueType::Transfer].qf_idx)
        .dst_queue_family_index(self.queues[&GPUQueueType::Transfer].qf_idx)],
    );

    cmd_buffer.end()?;

    cmd_buffer
      .submit(&[], &[], Some(&upload_fence))
      .map_err(|e| format!("at copying data to image: {e}"))?;

    upload_fence.wait(999999999).map_err(|e| format!("at waiting for fence: {e}"))?;

    Ok(image_2d)
  }

  pub fn create_ad_render_pass_builder(&self, flags: vk::RenderPassCreateFlags) -> AdRenderPassBuilder {
    AdRenderPassBuilder::new(Arc::clone(&self.vk_device), flags)
  }

  pub fn create_ad_shader(&self, create_info: &vk::ShaderModuleCreateInfo)
    -> Result<AdShaderModule, String> {
    unsafe {
      let shader_module = self.vk_device.create_shader_module(create_info, None)
        .map_err(|e| format!("error creating vk shader module: {e}"))?;
      Ok(AdShaderModule {
        vk_device: Arc::clone(&self.vk_device),
        inner: shader_module,
        dropped: false,
      })
    }
  }

  pub fn create_ad_shader_from_spv_file(&self, file_path: &Path) -> Result<AdShaderModule, String> {
    let mut fr = fs::File::open(file_path)
      .map_err(|e| format!("error opening file {:?}: {e}", file_path))?;
    let shader_code = ash::util::read_spv(&mut fr)
      .map_err(|e| format!("error reading ords from spv file: {e}"))?;
    let create_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
    self.create_ad_shader(&create_info)
  }

  pub fn create_ad_descriptor_set_layout(&self, bindings: &[vk::DescriptorSetLayoutBinding])
    -> Result<AdDescriptorSetLayout, String> {
    unsafe {
      let descriptor_set_layout = self.vk_device.create_descriptor_set_layout(
        &vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings),
        None
      )
        .map_err(|e| format!("at creating vk descriptor set layout: {e}"))?;
      Ok(AdDescriptorSetLayout { vk_device: Arc::clone(&self.vk_device), inner: descriptor_set_layout })
    }
  }

  pub fn create_ad_descriptor_pool(
    &self,
    flags: vk::DescriptorPoolCreateFlags,
    max_sets: u32,
    pool_sizes: &[vk::DescriptorPoolSize]
  ) -> Result<AdDescriptorPool, String> {
    unsafe {
      let descriptor_pool = self.vk_device.create_descriptor_pool(
        &vk::DescriptorPoolCreateInfo::default()
        .flags(flags)
          .max_sets(max_sets)
          .pool_sizes(pool_sizes),
        None
      )
        .map_err(|e| format!("at creating vk descriptor pool: {e}"))?;
      Ok(AdDescriptorPool {
        vk_device: Arc::clone(&self.vk_device),
        free_sets_supported: flags.contains(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET),
        inner: descriptor_pool
      })
    }
  }
}

impl Drop for VkContext {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.destroy_device(None);
      #[cfg(debug_assertions)]
      self
        .vk_instances
        .dbg_utils_instance
        .destroy_debug_utils_messenger(self.dbg_utils_messenger, None);
    }
  }
}

pub fn parse_spv_resources(path: &Path) -> Result<spirv::Ast<glsl::Target>, String> {
  let mut file = std::fs::File::open(path)
    .map_err(|e| format!("at opening spv file: {e}"))?;
  let words = ash::util::read_spv(&mut file)
    .map_err(|e| format!("at reading spv file: {e}"))?;
  let module = spirv::Module::from_words(&words);
  spirv::Ast::<glsl::Target>::parse(&module)
    .map_err(|e| format!("at parsing spv file: {e}"))
}
