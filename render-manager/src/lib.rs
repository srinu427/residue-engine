use std::sync::{Arc, Mutex};

pub use ash_wrappers::ash_present_wrappers::AdSurface;
pub use ash_wrappers::VkInstances;
use ash_wrappers::{
  ash_data_wrappers::{AdImage2D, AdImageView},
  ash_pipeline_wrappers::AdFrameBuffer,
  ash_present_wrappers::AdSwapchain,
  ash_queue_wrappers::{AdCommandBuffer, AdCommandPool, GPUQueueType},
  ash_sync_wrappers::{AdFence, AdSemaphore},
  vk, Allocator, MemoryLocation, VkContext,
};
use triangle_mesh_renderer::{TriMeshCPU, TriMeshRenderer, TriMeshVertex};

pub struct RenderManager {
  triangle_frame_buffers: Vec<AdFrameBuffer>,
  triangle_out_image_views: Vec<AdImageView>,
  triangle_out_images: Vec<AdImage2D>,
  triangle_mesh_renderer: TriMeshRenderer,
  gen_allocator: Arc<Mutex<Allocator>>,
  render_semaphores: Vec<AdSemaphore>,
  render_fences: Vec<AdFence>,
  render_cmd_buffers: Vec<AdCommandBuffer>,
  render_cmd_pool: AdCommandPool,
  image_acquire_fence: AdFence,
  swapchain: AdSwapchain,
  vk_context: Arc<VkContext>,
}

impl RenderManager {
  pub fn new(vk_instances: Arc<VkInstances>, surface: Arc<AdSurface>) -> Result<Self, String> {
    let vk_context = Arc::new(VkContext::new(vk_instances, &surface)?);

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
      surface.clone(),
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

    let render_cmd_pool = vk_context.queues[&GPUQueueType::Graphics]
      .create_ad_command_pool(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)?;

    let render_cmd_buffers =
      render_cmd_pool.allocate_command_buffers(vk::CommandBufferLevel::PRIMARY, 3)?;

    let render_semaphores = (0..3)
      .map(|_| vk_context.create_ad_semaphore(vk::SemaphoreCreateFlags::default()))
      .collect::<Result<Vec<_>, _>>()?;

    let render_fences = (0..3)
      .map(|_| vk_context.create_ad_fence(vk::FenceCreateFlags::SIGNALED))
      .collect::<Result<Vec<_>, _>>()?;

    let gen_allocator = Arc::new(Mutex::new(vk_context.create_allocator()?));

    let triangle_out_images = (0..3)
      .map(|i| {
        vk_context.create_ad_image_2d(
          gen_allocator.clone(),
          MemoryLocation::GpuOnly,
          &format!("triangle_out_image_{i}"),
          vk::Format::R8G8B8A8_UNORM,
          vk::Extent2D { width: 800, height: 600 },
          vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::COLOR_ATTACHMENT,
          vk::SampleCountFlags::TYPE_1,
          1,
        )
      })
      .collect::<Result<Vec<_>, _>>()?;

    render_cmd_buffers[0].begin(vk::CommandBufferBeginInfo::default())?;
    render_cmd_buffers[0].pipeline_barrier(
      vk::PipelineStageFlags::TRANSFER,
      vk::PipelineStageFlags::TRANSFER,
      vk::DependencyFlags::BY_REGION,
      &[],
      &[],
      &triangle_out_images
        .iter()
        .map(|x| {
          vk::ImageMemoryBarrier::default()
            .image(x.inner())
            .subresource_range(
              vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1)
                .base_array_layer(0)
                .level_count(1)
                .base_mip_level(0),
            )
            .src_queue_family_index(render_cmd_buffers[0].qf_idx)
            .dst_queue_family_index(render_cmd_buffers[0].qf_idx)
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        })
        .collect::<Vec<_>>(),
    );
    render_cmd_buffers[0].end()?;
    render_cmd_buffers[0].submit(&[], &[], Some(&image_acquire_fence))?;
    image_acquire_fence.wait(999999999)?;
    image_acquire_fence.reset()?;

    let triangle_out_image_views = (0..3)
      .map(|i| triangle_out_images[i].create_view(vk::ImageAspectFlags::COLOR))
      .collect::<Result<Vec<_>, _>>()?;

    let mut triangle_mesh_renderer = TriMeshRenderer::new(vk_context.clone())?;
    let tri_verts_cpu = TriMeshCPU {
      verts: vec![
        TriMeshVertex { pos: [0.0f32, -0.5f32, 0.0f32, 1.0f32] },
        TriMeshVertex { pos: [0.5f32, 0.5f32, 0.0f32, 1.0f32] },
        TriMeshVertex { pos: [-0.5f32, 0.5f32, 0.0f32, 1.0f32] },
      ],
      triangles: vec![[0, 1, 2]],
    };
    triangle_mesh_renderer.add_mesh("triangle_main", &tri_verts_cpu)?;

    let triangle_frame_buffers = (0..3)
      .map(|i| {
        triangle_mesh_renderer.render_pass.create_frame_buffer(
          &[&triangle_out_image_views[i]],
          swapchain_resolution,
          1,
        )
      })
      .collect::<Result<Vec<_>, String>>()?;

    Ok(Self {
      vk_context,
      swapchain,
      image_acquire_fence,
      render_cmd_pool,
      render_cmd_buffers,
      render_semaphores,
      render_fences,
      gen_allocator,
      triangle_mesh_renderer,
      triangle_out_images,
      triangle_out_image_views,
      triangle_frame_buffers,
    })
  }

  pub fn draw(&mut self) -> Result<bool, String> {
    // Acquiring next image to draw
    let (image_idx, refresh_needed) = self
      .swapchain
      .acquire_next_image(None, Some(&self.image_acquire_fence))
      .map_err(|e| format!("at acquiring next image: {e}"))?;
    self.image_acquire_fence.wait(999999999)?;
    self.image_acquire_fence.reset()?;

    if refresh_needed {
      let _ = self
        .swapchain
        .refresh_resolution()
        .inspect_err(|e| eprintln!("at refreshing swapchain res: {e}"));
      return Ok(true);
    }

    self.render_fences[image_idx as usize].wait(999999999)?;
    self.render_fences[image_idx as usize].reset()?;

    if !self.swapchain.is_initialized() {
      self
        .swapchain
        .initialize(&self.render_cmd_buffers[image_idx as usize])
        .map_err(|e| format!("at adding init cmds:  {e}"))?;

      self.render_cmd_buffers[image_idx as usize]
        .submit(&[], &[], Some(&self.image_acquire_fence))
        .map_err(|e| format!("error submitting cmds: {e}"))?;

      self.image_acquire_fence.wait(999999999)?;
      self.image_acquire_fence.reset()?;
      self.swapchain.set_initialized();
    }

    let current_sc_res = self.swapchain.resolution();

    self.render_cmd_buffers[image_idx as usize]
      .begin(vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::default()))
      .map_err(|e| format!("at beginning render cmd buffer:  {e}"))?;

    self.triangle_mesh_renderer.render_meshes(
      &self.render_cmd_buffers[image_idx as usize],
      &self.triangle_frame_buffers[image_idx as usize],
    );

    self.render_cmd_buffers[image_idx as usize].pipeline_barrier(
      vk::PipelineStageFlags::TRANSFER,
      vk::PipelineStageFlags::TRANSFER,
      vk::DependencyFlags::BY_REGION,
      &[],
      &[],
      &[vk::ImageMemoryBarrier::default()
        .image(self.swapchain.get_image(image_idx as usize))
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1)
            .base_array_layer(0)
            .level_count(1)
            .base_mip_level(0),
        )
        .src_queue_family_index(self.vk_context.queues[&GPUQueueType::Graphics].family_idx())
        .dst_queue_family_index(self.vk_context.queues[&GPUQueueType::Graphics].family_idx())
        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)],
    );

    self.render_cmd_buffers[image_idx as usize].blit_image(
      self.triangle_out_images[image_idx as usize].inner(),
      vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
      self.swapchain.get_image(image_idx as usize),
      vk::ImageLayout::TRANSFER_DST_OPTIMAL,
      &[vk::ImageBlit::default()
        .src_subresource(
          vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1),
        )
        .src_offsets(self.triangle_out_images[image_idx as usize].full_range_offset_3d())
        .dst_subresource(
          vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1),
        )
        .dst_offsets(self.swapchain.full_range_offset_3d())],
      vk::Filter::NEAREST,
    );

    self.render_cmd_buffers[image_idx as usize].pipeline_barrier(
      vk::PipelineStageFlags::TRANSFER,
      vk::PipelineStageFlags::TRANSFER,
      vk::DependencyFlags::BY_REGION,
      &[],
      &[],
      &[vk::ImageMemoryBarrier::default()
        .image(self.swapchain.get_image(image_idx as usize))
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1)
            .base_array_layer(0)
            .level_count(1)
            .base_mip_level(0),
        )
        .src_queue_family_index(self.vk_context.queues[&GPUQueueType::Graphics].family_idx())
        .dst_queue_family_index(self.vk_context.queues[&GPUQueueType::Graphics].family_idx())
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)],
    );

    self.render_cmd_buffers[image_idx as usize]
      .end()
      .map_err(|e| format!("at ending render cmd buffer: {e}"))?;

    self.render_cmd_buffers[image_idx as usize]
      .submit(
        &[&self.render_semaphores[image_idx as usize]],
        &[],
        Some(&self.render_fences[image_idx as usize]),
      )
      .map_err(|e| format!("error submitting cmds: {e}"))?;

    if let Err(e) =
      self.swapchain.present_image(image_idx, vec![&self.render_semaphores[image_idx as usize]])
    {
      if e.ends_with("ERROR_OUT_OF_DATE_KHR") {
        return Ok(true);
      }
    }
    Ok(false)
  }
}

impl Drop for RenderManager {
  fn drop(&mut self) {
    for fence in self.render_fences.iter() {
      let _ = fence.wait(999999999);
      let _ = fence.reset();
    }
  }
}
