pub mod data_wrappers;
pub mod sync_wrappers;

use std::sync::Arc;

use ash::{khr, vk};

pub struct AdSurface {
  pub(crate) surface_instance: Arc<khr::surface::Instance>,
  pub inner: vk::SurfaceKHR,
}

impl AdSurface {
  pub fn get_gpu_formats(
    &self,
    gpu: vk::PhysicalDevice,
  ) -> Result<Vec<vk::SurfaceFormatKHR>, String> {
    unsafe {
      self
        .surface_instance
        .get_physical_device_surface_formats(gpu, self.inner)
        .map_err(|e| format!("can't get surface formats: {e}"))
    }
  }

  pub fn get_gpu_capabilities(
    &self,
    gpu: vk::PhysicalDevice,
  ) -> Result<vk::SurfaceCapabilitiesKHR, String> {
    unsafe {
      self
        .surface_instance
        .get_physical_device_surface_capabilities(gpu, self.inner)
        .map_err(|e| format!("can't get surface capabilities: {e}"))
    }
  }

  pub fn get_gpu_present_modes(
    &self,
    gpu: vk::PhysicalDevice,
  ) -> Result<Vec<vk::PresentModeKHR>, String> {
    unsafe {
      self
        .surface_instance
        .get_physical_device_surface_present_modes(gpu, self.inner)
        .map_err(|e| format!("can't get surface present modes: {e}"))
    }
  }
}

impl Drop for AdSurface {
  fn drop(&mut self) {
    unsafe {
      self.surface_instance.destroy_surface(self.inner, None);
    }
  }
}

pub struct AdSwapchain {
  pub(crate) swapchain_device: Arc<khr::swapchain::Device>,
  pub(crate) surface: Arc<AdSurface>,
  pub(crate) gpu: vk::PhysicalDevice,
  pub(crate) present_queue: Arc<AdQueue>,
  pub inner: vk::SwapchainKHR,
  pub images: Vec<vk::Image>,
  pub(crate) image_count: u32,
  pub(crate) color_space: vk::ColorSpaceKHR,
  pub(crate) format: vk::Format,
  pub(crate) resolution: vk::Extent2D,
  pub(crate) usage: vk::ImageUsageFlags,
  pub(crate) pre_transform: vk::SurfaceTransformFlagsKHR,
  pub(crate) present_mode: vk::PresentModeKHR,
  pub(crate) initialized: bool,
}

impl AdSwapchain {
  pub fn refresh_resolution(&mut self) -> Result<(), String> {
    let surface_caps = self.surface.get_gpu_capabilities(self.gpu)?;

    let swapchain_info = vk::SwapchainCreateInfoKHR::default()
      .surface(self.surface.inner)
      .old_swapchain(self.inner)
      .min_image_count(self.image_count)
      .image_color_space(self.color_space)
      .image_format(self.format)
      .image_extent(surface_caps.current_extent)
      .image_usage(self.usage)
      .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
      .pre_transform(self.pre_transform)
      .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
      .present_mode(self.present_mode)
      .clipped(true)
      .image_array_layers(1);
    unsafe {
      let new_swapchain = self
        .swapchain_device
        .create_swapchain(&swapchain_info, None)
        .map_err(|e| format!("at refreshing vk swapchain: {e}"))?;
      let new_images = self
        .swapchain_device
        .get_swapchain_images(new_swapchain)
        .map_err(|e| format!("at getting new swapchain images: {e}"))?;
      self.swapchain_device.destroy_swapchain(self.inner, None);
      self.inner = new_swapchain;
      self.images = new_images;
      self.resolution = surface_caps.current_extent;
      println!("new swapchain res: {:?}", surface_caps.current_extent)
    }
    self.initialized = false;
    Ok(())
  }

  pub fn acquire_next_image(
    &mut self,
    semaphore: Option<&sync_wrappers::AdSemaphore>,
    fence: Option<&sync_wrappers::AdFence>,
  ) -> Result<(u32, bool), String> {
    unsafe {
      match self.swapchain_device.acquire_next_image(
        self.inner,
        999999999,
        semaphore.map(|x| x.inner).unwrap_or(vk::Semaphore::null()),
        fence.map(|x| x.inner).unwrap_or(vk::Fence::null()),
      ) {
        Ok((idx, refresh_needed)) => {
          if refresh_needed {
            Ok((idx, true))
          } else {
            Ok((idx, false))
          }
        }
        Err(e) => {
          if e == vk::Result::ERROR_OUT_OF_DATE_KHR {
            return Ok((0, true));
          }
          Err(format!("at vk acquire image: {e}"))
        }
      }
    }
  }

  pub fn present_image(
    &self,
    image_idx: u32,
    wait_semaphores: Vec<&sync_wrappers::AdSemaphore>,
  ) -> Result<(), String> {
    unsafe {
      self
        .swapchain_device
        .queue_present(
          self.present_queue.inner,
          &vk::PresentInfoKHR::default()
            .swapchains(&[self.inner])
            .wait_semaphores(&wait_semaphores.iter().map(|x| x.inner).collect::<Vec<_>>())
            .image_indices(&[image_idx]),
        )
        .map_err(|e| format!("at vk present: {e}"))?;
    }
    Ok(())
  }

  pub fn full_range_offset_3d(&self) -> [vk::Offset3D; 2] {
    [
      vk::Offset3D::default(),
      vk::Offset3D::default().x(self.resolution.width as i32).y(self.resolution.height as i32).z(1),
    ]
  }

  pub fn is_initialized(&self) -> bool {
    self.initialized
  }

  pub fn set_initialized(&mut self) {
    self.initialized = true;
  }

  pub fn initialize(&mut self, cmd_buffer: &AdCommandBuffer, q_id: u32) -> Result<(), String> {
    if !self.initialized {
      cmd_buffer.begin(vk::CommandBufferBeginInfo::default())?;
      cmd_buffer.pipeline_barrier(
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::BY_REGION,
        &[],
        &[],
        &self
          .images
          .iter()
          .map(|x| {
            vk::ImageMemoryBarrier::default()
              .image(*x)
              .subresource_range(
                vk::ImageSubresourceRange::default()
                  .aspect_mask(vk::ImageAspectFlags::COLOR)
                  .layer_count(1)
                  .base_array_layer(0)
                  .level_count(1)
                  .base_mip_level(0),
              )
              .src_queue_family_index(q_id)
              .dst_queue_family_index(q_id)
              .src_access_mask(vk::AccessFlags::NONE)
              .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
              .old_layout(vk::ImageLayout::UNDEFINED)
              .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
          })
          .collect::<Vec<_>>(),
      );
      cmd_buffer.end()?;
    }
    Ok(())
  }
}

impl Drop for AdSwapchain {
  fn drop(&mut self) {
    unsafe {
      self.swapchain_device.destroy_swapchain(self.inner, None);
    }
  }
}

pub struct ADRenderPass {
  pub(crate) vk_device: Arc<ash::Device>,
  pub inner: vk::RenderPass,
}

impl Drop for ADRenderPass {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.destroy_render_pass(self.inner, None);
    }
  }
}

pub struct AdCommandPool {
  pub(crate) vk_device: Arc<ash::Device>,
  pub inner: vk::CommandPool,
  pub qf_idx: u32,
}

impl AdCommandPool {
  pub fn allocate_command_buffers(
    &self,
    level: vk::CommandBufferLevel,
    count: u32,
  ) -> Result<Vec<AdCommandBuffer>, String> {
    let cmd_buffers = unsafe {
      self
        .vk_device
        .allocate_command_buffers(
          &vk::CommandBufferAllocateInfo::default()
            .command_pool(self.inner)
            .level(level)
            .command_buffer_count(count),
        )
        .map_err(|e| format!("at creating command buffer: {e}"))?
        .iter()
        .map(|&x| AdCommandBuffer {
          vk_device: Arc::clone(&self.vk_device),
          pool: self.inner,
          inner: x,
        })
        .collect::<Vec<_>>()
    };
    Ok(cmd_buffers)
  }
}

impl Drop for AdCommandPool {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.destroy_command_pool(self.inner, None);
    }
  }
}

pub struct AdCommandBuffer {
  pub(crate) vk_device: Arc<ash::Device>,
  pool: vk::CommandPool,
  pub inner: vk::CommandBuffer,
}

impl AdCommandBuffer {
  pub fn begin(&self, info: vk::CommandBufferBeginInfo) -> Result<(), String> {
    unsafe {
      self
        .vk_device
        .begin_command_buffer(self.inner, &info)
        .map_err(|e| format!("at cmd buffer begin: {e}"))
    }
  }

  pub fn end(&self) -> Result<(), String> {
    unsafe {
      self.vk_device.end_command_buffer(self.inner).map_err(|e| format!("at cmd buffer end: {e}"))
    }
  }

  pub fn reset(&self) -> Result<(), String> {
    unsafe {
      self
        .vk_device
        .reset_command_buffer(self.inner, vk::CommandBufferResetFlags::default())
        .map_err(|e| format!("at cmd buffer reset: {e}"))
    }
  }

  pub fn begin_render_pass(
    &self,
    render_pass_begin_info: vk::RenderPassBeginInfo,
    subpass_contents: vk::SubpassContents,
  ) {
    unsafe {
      self.vk_device.cmd_begin_render_pass(self.inner, &render_pass_begin_info, subpass_contents);
    }
  }

  pub fn end_render_pass(&self) {
    unsafe {
      self.vk_device.cmd_end_render_pass(self.inner);
    }
  }

  pub fn bind_pipeline(&self, pipeline_bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline) {
    unsafe {
      self.vk_device.cmd_bind_pipeline(self.inner, pipeline_bind_point, pipeline);
    }
  }

  pub fn bind_vertex_buffer(
    &self,
    binding_count: u32,
    buffers: &[vk::Buffer],
    offsets: &[vk::DeviceSize],
  ) {
    unsafe {
      self.vk_device.cmd_bind_vertex_buffers(self.inner, binding_count, buffers, offsets);
    }
  }

  pub fn bind_index_buffer(
    &self,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    index_type: vk::IndexType,
  ) {
    unsafe {
      self.vk_device.cmd_bind_index_buffer(self.inner, buffer, offset, index_type);
    }
  }

  pub fn pipeline_barrier(
    &self,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
    dependency_flags: vk::DependencyFlags,
    memory_barriers: &[vk::MemoryBarrier],
    buffer_memory_barriers: &[vk::BufferMemoryBarrier],
    image_memory_barriers: &[vk::ImageMemoryBarrier],
  ) {
    unsafe {
      self.vk_device.cmd_pipeline_barrier(
        self.inner,
        src_stage,
        dst_stage,
        dependency_flags,
        memory_barriers,
        buffer_memory_barriers,
        image_memory_barriers,
      );
    }
  }

  pub fn blit_image(
    &self,
    src_image: vk::Image,
    src_image_layout: vk::ImageLayout,
    dst_image: vk::Image,
    dst_image_layout: vk::ImageLayout,
    regions: &[vk::ImageBlit],
    filter: vk::Filter,
  ) {
    unsafe {
      self.vk_device.cmd_blit_image(
        self.inner,
        src_image,
        src_image_layout,
        dst_image,
        dst_image_layout,
        regions,
        filter,
      );
    }
  }

  pub fn copy_buffer_to_image(
    &self,
    src_buffer: vk::Buffer,
    dst_image: vk::Image,
    dst_image_layout: vk::ImageLayout,
    regions: &[vk::BufferImageCopy],
  ) {
    unsafe {
      self.vk_device.cmd_copy_buffer_to_image(
        self.inner,
        src_buffer,
        dst_image,
        dst_image_layout,
        regions,
      );
    }
  }
}

impl Drop for AdCommandBuffer {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.free_command_buffers(self.pool, &[self.inner]);
    }
  }
}

pub struct AdQueue {
  pub(crate) vk_device: Arc<ash::Device>,
  pub qf_idx: u32,
  pub inner: vk::Queue,
}

impl AdQueue {
  pub fn create_ad_command_pool(
    &self,
    flags: vk::CommandPoolCreateFlags,
    queue_idx: u32,
  ) -> Result<AdCommandPool, String> {
    unsafe {
      let cmd_pool = self
        .vk_device
        .create_command_pool(
          &vk::CommandPoolCreateInfo::default().flags(flags).queue_family_index(queue_idx),
          None,
        )
        .map_err(|e| format!("at vk cmd pool create: {e}"))?;
      Ok(AdCommandPool {
        vk_device: Arc::clone(&self.vk_device),
        inner: cmd_pool,
        qf_idx: self.qf_idx,
      })
    }
  }

  pub fn wait(&self) -> Result<(), String> {
    unsafe {
      self
        .vk_device
        .queue_wait_idle(self.inner)
        .map_err(|e| format!("error waiting for queue idle: {e}"))
    }
  }
}
