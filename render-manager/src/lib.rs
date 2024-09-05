use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ash_wrappers::ad_wrappers::data_wrappers::{AdBuffer, AdImage2D, Allocator};
use ash_wrappers::ad_wrappers::sync_wrappers::{AdFence, AdSemaphore};
pub use ash_wrappers::ad_wrappers::AdSurface;
use ash_wrappers::ad_wrappers::{AdCommandBuffer, AdCommandPool, AdDescriptorPool, AdDescriptorSet, AdDescriptorSetLayout, AdPipeline, AdRenderPass, AdSwapchain};
pub use ash_wrappers::VkInstances;
use ash_wrappers::{vk, GPUQueueType, MemoryLocation, VkContext};

pub struct RenderManager {
  triangle_pipeline: AdPipeline,
  triangle_dset: AdDescriptorSet,
  triangle_dset_pool: AdDescriptorPool,
  triangle_dset_layout: AdDescriptorSetLayout,
  triangle_render_pass: AdRenderPass,
  triangle_out_images: Vec<AdImage2D>,
  triangle_vb: AdBuffer,
  bg_image: AdImage2D,
  gen_allocator: Arc<Mutex<Allocator>>,
  transfer_cmd_pool: AdCommandPool,
  render_semaphores: Vec<AdSemaphore>,
  render_fences: Vec<AdFence>,
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
      .queues[&GPUQueueType::Graphics]
      .create_ad_command_pool(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)?;

    let render_cmd_buffers =
      render_cmd_pool.allocate_command_buffers(vk::CommandBufferLevel::PRIMARY, 3)?;

    let render_semaphores = (0..3)
      .map(|_| vk_context.create_ad_semaphore(vk::SemaphoreCreateFlags::default()))
      .collect::<Result<Vec<_>, _>>()?;

    let render_fences = (0..3)
      .map(|_| vk_context.create_ad_fence(vk::FenceCreateFlags::SIGNALED))
      .collect::<Result<Vec<_>, _>>()?;

    let transfer_cmd_pool = vk_context
      .queues[&GPUQueueType::Transfer]
      .create_ad_command_pool(vk::CommandPoolCreateFlags::TRANSIENT)?;

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
      1,
    )?;

    let triangle_out_images = (0..3)
      .map(|i| {
        vk_context.create_ad_image_2d(
          Arc::clone(&gen_allocator),
          MemoryLocation::GpuOnly,
          &format!("triangle_out_image_{i}"),
          vk::Format::R8G8B8A8_UNORM,
          swapchain_resolution,
          vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::COLOR_ATTACHMENT,
          vk::SampleCountFlags::TYPE_1,
          1
        )
      })
      .collect::<Result<Vec<_>, _>>()?;

    let tri_verts = [
      [0.0f32, 0.0f32, 1.0f32, 1.0f32],
      [1.0f32, 0.0f32, 1.0f32, 1.0f32],
      [1.0f32, 1.0f32, 1.0f32, 1.0f32],
    ];

    let tmp_cmd_buffer = transfer_cmd_pool
      .allocate_command_buffers(vk::CommandBufferLevel::PRIMARY, 1)?
      .remove(0);

    let triangle_vb = vk_context.create_ad_buffer_from_data(
      Arc::clone(&gen_allocator),
      MemoryLocation::GpuOnly,
      "triangle",
      vk::BufferCreateFlags::default(),
      vk::BufferUsageFlags::STORAGE_BUFFER,
      unsafe {std::slice::from_raw_parts(tri_verts.as_ptr() as *const u8, 48)},
      &tmp_cmd_buffer
    )?;

    let triangle_dset_layout = vk_context.create_ad_descriptor_set_layout(&[
      vk::DescriptorSetLayoutBinding::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
    ])?;

    let triangle_dset_pool = vk_context.create_ad_descriptor_pool(
      vk::DescriptorPoolCreateFlags::default(),
      1,
      &[vk::DescriptorPoolSize::default().descriptor_count(1).ty(vk::DescriptorType::STORAGE_BUFFER)]
    )?;

    let triangle_dset = triangle_dset_pool.allocate_descriptor_sets(&[&triangle_dset_layout])?.remove(0);

    let triangle_render_pass = vk_context
      .create_ad_render_pass_builder(vk::RenderPassCreateFlags::default())
      .add_attachment(
        vk::AttachmentDescription::default()
          .format(vk::Format::R8G8B8A8_UNORM)
          .samples(vk::SampleCountFlags::TYPE_1)
          .initial_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
          .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
      )
      .add_sub_pass(
        vk::SubpassDescription::default()
          .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
          .color_attachments(&[vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)]),
      )
      .add_sub_pass_dependency(
        vk::SubpassDependency::default()
          .src_subpass(vk::SUBPASS_EXTERNAL)
          .dst_subpass(0)
          .src_stage_mask(vk::PipelineStageFlags::TRANSFER)
          .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
          .src_access_mask(vk::AccessFlags::TRANSFER_READ)
          .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
      )
      .add_sub_pass_dependency(
        vk::SubpassDependency::default()
          .src_subpass(0)
          .dst_subpass(vk::SUBPASS_EXTERNAL)
          .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
          .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
          .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
          .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
      )
      .build()?;

    let mut vert_shader = vk_context
      .create_ad_shader_from_spv_file(&PathBuf::from("render-manager/shaders/triangle.vert.spv"))?;
    let mut frag_shader = vk_context
      .create_ad_shader_from_spv_file(&PathBuf::from("render-manager/shaders/triangle.frag.spv"))?;

    let triangle_rasterizer_info = vk::PipelineRasterizationStateCreateInfo::default()
      .cull_mode(vk::CullModeFlags::BACK)
      .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
      .polygon_mode(vk::PolygonMode::FILL)
      .line_width(1.0);
    
    let triangle_pipeline = triangle_render_pass.create_ad_g_pipeline(
      0,
      &[&triangle_dset_layout],
      HashMap::from([
        (vk::ShaderStageFlags::VERTEX, &vert_shader),
        (vk::ShaderStageFlags::FRAGMENT, &frag_shader)
      ]),
      triangle_rasterizer_info,
      &vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&[vk::PipelineColorBlendAttachmentState::default()
          .color_write_mask(vk::ColorComponentFlags::RGBA)
          .blend_enable(false)])
    )?;

    vert_shader.manual_destroy();
    frag_shader.manual_destroy();

    Ok(Self {
      vk_context,
      swapchain,
      image_acquire_fence,
      render_cmd_pool,
      render_cmd_buffers,
      render_semaphores,
      render_fences,
      transfer_cmd_pool,
      gen_allocator,
      bg_image,
      triangle_vb,
      triangle_out_images,
      triangle_render_pass,
      triangle_dset_layout,
      triangle_dset_pool,
      triangle_dset,
      triangle_pipeline,
    })
  }

  pub fn draw(&mut self) -> Result<bool, String> {
    let (image_idx, refresh_needed) = self
      .swapchain
      .acquire_next_image(None, Some(&self.image_acquire_fence))
      .map_err(|e| format!("at acquiring next image: {e}"))?;
    if refresh_needed {
      let _ = self
        .swapchain
        .refresh_resolution()
        .inspect_err(|e| eprintln!("at refreshing swapchain res: {e}"));
      return Ok(true)
    }
    self.image_acquire_fence.wait(999999999)?;
    self.image_acquire_fence.reset()?;

    self.render_fences[image_idx as usize].wait(999999999)?;
    self.render_fences[image_idx as usize].reset()?;

    if !self.swapchain.is_initialized() {
      self
        .swapchain
        .initialize(&self.render_cmd_buffers[image_idx as usize])
        .map_err(|e| format!("at adding init cmds:  {e}"))?;

      self
        .render_cmd_buffers[image_idx as usize]
        .submit(&[], &[], Some(&self.image_acquire_fence))
        .map_err(|e| format!("error submitting cmds: {e}"))?;

      self.image_acquire_fence.wait(999999999)?;
      self.image_acquire_fence.reset()?;
      self.swapchain.set_initialized();
    }

    self.render_cmd_buffers[image_idx as usize]
      .begin(vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::default()))
      .map_err(|e| format!("at beginning render cmd buffer:  {e}"))?;

    self.render_cmd_buffers[image_idx as usize].pipeline_barrier(
      vk::PipelineStageFlags::TRANSFER,
      vk::PipelineStageFlags::TRANSFER,
      vk::DependencyFlags::BY_REGION,
      &[],
      &[],
      &[vk::ImageMemoryBarrier::default()
        .image(self.swapchain.images[image_idx as usize])
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1)
            .base_array_layer(0)
            .level_count(1)
            .base_mip_level(0),
        )
        .src_queue_family_index(self.vk_context.queues[&GPUQueueType::Graphics].qf_idx)
        .dst_queue_family_index(self.vk_context.queues[&GPUQueueType::Graphics].qf_idx)
        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)],
    );

    self.render_cmd_buffers[image_idx as usize].blit_image(
      self.bg_image.inner,
      vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
      self.swapchain.images[image_idx as usize],
      vk::ImageLayout::TRANSFER_DST_OPTIMAL,
      &[vk::ImageBlit::default()
        .src_subresource(
          vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1),
        )
        .src_offsets(self.bg_image.full_range_offset_3d())
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
        .image(self.swapchain.images[image_idx as usize])
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1)
            .base_array_layer(0)
            .level_count(1)
            .base_mip_level(0),
        )
        .src_queue_family_index(self.vk_context.queues[&GPUQueueType::Graphics].qf_idx)
        .dst_queue_family_index(self.vk_context.queues[&GPUQueueType::Graphics].qf_idx)
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)],
    );

    self.render_cmd_buffers[image_idx as usize]
      .end()
      .map_err(|e| format!("at ending render cmd buffer: {e}"))?;

    self
      .render_cmd_buffers[image_idx as usize]
      .submit(
        &[&self.render_semaphores[image_idx as usize]],
        &[],
        Some(&self.render_fences[image_idx as usize])
      )
      .map_err(|e| format!("error submitting cmds: {e}"))?;

    if let Err(e) = self
      .swapchain
      .present_image(image_idx, vec![&self.render_semaphores[image_idx as usize]]) {
      if e.ends_with("ERROR_OUT_OF_DATE_KHR") {
        return Ok(true)
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
