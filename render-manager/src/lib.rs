use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub use ash_wrappers::ad_wrappers::AdSurface;
pub use ash_wrappers::VkInstances;
use ash_wrappers::{GPUQueueType, MemoryLocation, vk, VkContext};
use ash_wrappers::ad_wrappers::{AdCommandBuffer, AdCommandPool, AdSwapchain};
use ash_wrappers::ad_wrappers::data_wrappers::{AdImage2D, Allocator};
use ash_wrappers::ad_wrappers::sync_wrappers::AdFence;

pub struct RenderManager {
  bg_image: AdImage2D,
  gen_allocator: Arc<Mutex<Allocator>>,
  transfer_cmd_pool: AdCommandPool,
  render_cmd_buffers: Vec<AdCommandBuffer>,
  render_cmd_pool: AdCommandPool,
  image_acquire_fence: AdFence,
  swapchain: AdSwapchain,
  vk_context: VkContext,
}

impl RenderManager {
  pub fn new(vk_instances: Arc<VkInstances>, surface: Arc<AdSurface>) -> Result<Self, String> {
    let vk_context = VkContext::new(vk_instances, &surface)?;

    let surface_formats = surface.get_gpu_formats(vk_context.gpu)?;
    let surface_caps = surface.get_gpu_capabilities(vk_context.gpu)?;
    let surface_present_modes = surface.get_gpu_present_modes(vk_context.gpu)?;

    let surface_format = surface_formats
      .iter()
      .find(|f| f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
      .cloned()
      .unwrap_or(surface_formats[0]);
    let present_mode = surface_present_modes
      .iter()
      .find(|m| **m == vk::PresentModeKHR::MAILBOX)
      .cloned()
      .unwrap_or(vk::PresentModeKHR::FIFO);

    let swapchain_resolution = match surface_caps.current_extent.width {
      u32::MAX => vk::Extent2D::default().width(640).height(480),
      _ => surface_caps.current_extent,
    };

    let swapchain_image_count = std::cmp::min(
      surface_caps.min_image_count + 1,
      std::cmp::max(surface_caps.max_image_count, std::cmp::min(surface_caps.min_image_count, 3)),
    );

    let swapchain = vk_context.create_ad_swapchain(
      Arc::clone(&surface),
      swapchain_image_count,
      surface_format.color_space,
      surface_format.format,
      swapchain_resolution,
      vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
      surface_caps.current_transform,
      present_mode,
      None,
    )?;

    let image_acquire_fence = vk_context.create_ad_fence(vk::FenceCreateFlags::default())?;
    let render_cmd_pool = vk_context
      .create_ad_command_pool(
        vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        vk_context.qf_indices[&GPUQueueType::Graphics],
      )?;
    let render_cmd_buffers = render_cmd_pool
      .allocate_command_buffers(vk::CommandBufferLevel::PRIMARY, 3)?;

    let transfer_cmd_pool = vk_context
      .create_ad_command_pool(
        vk::CommandPoolCreateFlags::TRANSIENT,
        vk_context.qf_indices[&GPUQueueType::Transfer]
      )?;

    let gen_allocator = Arc::new(Mutex::new(vk_context.create_allocator()?));

    let bg_image = vk_context.create_ad_image_2d_from_file(
      Arc::clone(&gen_allocator),
      MemoryLocation::GpuOnly,
      &transfer_cmd_pool,
      "background",
      vk::Format::R8G8B8A8_UNORM,
      &PathBuf::from("./background.png"),
      vk::ImageUsageFlags::TRANSFER_SRC,
      vk::SampleCountFlags::TYPE_1,
      1
    )?;

    Ok(Self {
      vk_context,
      swapchain,
      image_acquire_fence,
      render_cmd_pool,
      render_cmd_buffers,
      transfer_cmd_pool,
      gen_allocator,
      bg_image
    })
  }

  pub fn draw(&mut self) -> Result<(), String> {
    let (image_idx, refresh_needed) = self
      .swapchain
      .acquire_next_image(None, Some(&self.image_acquire_fence))
      .map_err(|e| format!("at acquiring next image: {e}"))?;
    self.image_acquire_fence.wait(999999999)?;
    self.image_acquire_fence.reset()?;

    self
      .render_cmd_buffers[image_idx as usize]
      .begin(vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::default()))
      .map_err(|e| format!("at beginning render cmd buffer:  {e}"))?;

    self.render_cmd_buffers[image_idx as usize].blit_image(
      self.bg_image.inner,
      vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
      self.swapchain.images[image_idx as usize],
      vk::ImageLayout::TRANSFER_DST_OPTIMAL,
      &[vk::ImageBlit::default()
        .src_subresource(vk::ImageSubresourceLayers::default()
          .aspect_mask(vk::ImageAspectFlags::COLOR)
          .mip_level(0)
          .base_array_layer(0)
          .layer_count(1))
        .src_offsets(self.bg_image.full_range_offset_3d())
        .dst_subresource(
          vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1),
        )
        .dst_offsets(self.swapchain.full_range_offset_3d())],
      vk::Filter::NEAREST
    );

    self
      .render_cmd_buffers[image_idx as usize]
      .end()
      .map_err(|e| format!("at ending render cmd buffer: {e}"))?;

    self.swapchain.present_image(image_idx, vec![])?;

    Ok(())
  }

  pub fn refresh_swapchain_resolution(&mut self) {
    let _ = self.swapchain.refresh_resolution()
      .inspect_err(|e| eprintln!("error while refreshing swapchain resolution: {e}"));
  }
}
