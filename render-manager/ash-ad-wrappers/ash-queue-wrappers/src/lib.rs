use std::sync::Arc;

use ash_context::{ash::{self, vk}, getset, AdAshDevice};
use ash_sync_wrappers::{AdFence, AdSemaphore};

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdQueue {
  #[getset(get = "pub")]
  ash_device: Arc<AdAshDevice>,
  #[getset(get_copy = "pub")]
  family_index: u32,
  #[getset(get_copy = "pub")]
  queue_index: u32,
  #[getset(get_copy = "pub")]
  inner: vk::Queue,
}

impl AdQueue {
  pub fn new(ash_device: Arc<AdAshDevice>, qf_idx: u32, q_idx: u32) -> Self {
    let vk_queue = unsafe {
      ash_device.inner().get_device_queue(qf_idx, q_idx)
    };
    Self {
      ash_device,
      family_index: qf_idx,
      queue_index: q_idx,
      inner: vk_queue
    }
  }

  pub fn submit(
    &self,
    submits: &[vk::SubmitInfo],
    fence: Option<&AdFence>
  ) -> Result<(), String> {
    unsafe {
      self
        .ash_device
        .inner()
        .queue_submit(self.inner, submits, fence.map(|x| x.inner()).unwrap_or(vk::Fence::null()))
        .map_err(|e| format!("error submitting to queue: {e}"))
    }
  }

  pub fn wait(&self) -> Result<(), String> {
    unsafe {
      self
        .ash_device
        .inner()
        .queue_wait_idle(self.inner)
        .map_err(|e| format!("error waiting for queue idle: {e}"))
    }
  }
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdCommandPool {
  #[getset(get_copy = "pub")]
  inner: vk::CommandPool,
  #[getset(get = "pub")]
  queue: Arc<AdQueue>,
}

impl AdCommandPool {
  pub fn new(
    queue: Arc<AdQueue>,
    flags: vk::CommandPoolCreateFlags
  ) -> Result<Self, String> {
    unsafe {
      let cmd_pool = queue
        .ash_device()
        .inner()
        .create_command_pool(
          &vk::CommandPoolCreateInfo::default()
            .flags(flags)
            .queue_family_index(queue.family_index()),
          None,
        )
        .map_err(|e| format!("at vk cmd pool create: {e}"))?;
      Ok(Self { inner: cmd_pool, queue })
    }
  }
}

impl Drop for AdCommandPool {
  fn drop(&mut self) {
    unsafe {
      self
        .queue
        .ash_device()
        .inner()
        .destroy_command_pool(self.inner, None);
    }
  }
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdCommandBuffer {
  #[getset(get = "pub")]
  cmd_pool: Arc<AdCommandPool>,
  #[getset(get_copy = "pub")]
  inner: vk::CommandBuffer,
}

impl AdCommandBuffer {
  pub fn new(
    cmd_pool: Arc<AdCommandPool>,
    level: vk::CommandBufferLevel,
    count: u32,
  ) -> Result<Vec<Self>, String> {
    let cmd_buffers = unsafe {
      cmd_pool
        .queue()
        .ash_device()
        .inner()
        .allocate_command_buffers(
          &vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool.inner())
            .level(level)
            .command_buffer_count(count),
        )
        .map_err(|e| format!("at creating command buffer: {e}"))?
        .iter()
        .map(|&x| AdCommandBuffer {
          cmd_pool: cmd_pool.clone(),
          inner: x,
        })
        .collect::<Vec<_>>()
    };
    Ok(cmd_buffers)
  }

  fn get_ash_device(&self) -> &ash::Device {
    self.cmd_pool.queue().ash_device().inner()
  }

  pub fn begin(&self, flags: vk::CommandBufferUsageFlags) -> Result<(), String> {
    unsafe {
      self
        .get_ash_device()
        .begin_command_buffer(self.inner, &vk::CommandBufferBeginInfo::default().flags(flags))
        .map_err(|e| format!("at cmd buffer begin: {e}"))
    }
  }

  pub fn end(&self) -> Result<(), String> {
    unsafe {
      self
        .get_ash_device()
        .end_command_buffer(self.inner)
        .map_err(|e| format!("at cmd buffer end: {e}"))
    }
  }

  pub fn submit(
    &self,
    signal_semaphores: &[&AdSemaphore],
    wait_semaphores: &[(&AdSemaphore, vk::PipelineStageFlags)],
    fence: Option<&AdFence>
  ) -> Result<(), String> {
    unsafe {
      self
        .get_ash_device()
        .queue_submit(
          self.cmd_pool.queue().inner(),
          &[
            vk::SubmitInfo::default()
              .command_buffers(&[self.inner])
              .signal_semaphores(&signal_semaphores.iter().map(|x| x.inner()).collect::<Vec<_>>())
              .wait_semaphores(&wait_semaphores.iter().map(|x| x.0.inner()).collect::<Vec<_>>())
              .wait_dst_stage_mask(&wait_semaphores.iter().map(|x| x.1).collect::<Vec<_>>())
          ],
          fence.map_or(vk::Fence::null(), |x| x.inner())
        )
        .map_err(|e| format!("error submitting cmd buffer: {e}"))
    }
  }

  pub fn reset(&self) -> Result<(), String> {
    unsafe {
      self
        .get_ash_device()
        .reset_command_buffer(self.inner, vk::CommandBufferResetFlags::default())
        .map_err(|e| format!("at cmd buffer reset: {e}"))
    }
  }

  pub fn begin_render_pass(
    &self,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    render_area: vk::Rect2D,
    clear_values: &[vk::ClearValue],
    subpass_contents: vk::SubpassContents,
  ) {
    unsafe {
      self
        .get_ash_device()
        .cmd_begin_render_pass(
          self.inner,
          &vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(render_area)
            .clear_values(clear_values),
          subpass_contents
        );
    }
  }

  pub fn end_render_pass(&self) {
    unsafe {
      self.get_ash_device().cmd_end_render_pass(self.inner);
    }
  }

  pub fn bind_pipeline(&self, pipeline_bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline) {
    unsafe {
      self
        .get_ash_device()
        .cmd_bind_pipeline(self.inner, pipeline_bind_point, pipeline);
    }
  }

  pub fn bind_descriptor_sets(
    &self,
    pipeline_bind_point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    descriptor_sets: &[vk::DescriptorSet]
  ) {
    unsafe {
      self
        .get_ash_device()
        .cmd_bind_descriptor_sets(
          self.inner,
          pipeline_bind_point,
          layout,
          0,
          &descriptor_sets,
          &[]
        )
    }
  }

  pub fn set_view_port(&self, viewports: &[vk::Viewport]) {
    unsafe {
      self.get_ash_device().cmd_set_viewport(self.inner, 0, viewports);
    }
  }

  pub fn set_scissor(&self, scissors: &[vk::Rect2D]) {
    unsafe {
      self.get_ash_device().cmd_set_scissor(self.inner, 0, scissors);
    }
  }

  pub fn draw(&self, vert_count: u32) {
    unsafe {
      self.cmd_pool.queue().ash_device().inner().cmd_draw(self.inner, vert_count, 1, 0, 0);
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
      self.get_ash_device().cmd_pipeline_barrier(
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

  pub fn copy_buffer_to_buffer_cmd(
    &self,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    regions: &[vk::BufferCopy]
  ) {
    unsafe {
      self
        .get_ash_device()
        .cmd_copy_buffer(self.inner, src_buffer, dst_buffer, regions);
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
      self.get_ash_device().cmd_copy_buffer_to_image(
        self.inner,
        src_buffer,
        dst_image,
        dst_image_layout,
        regions,
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
      self.get_ash_device().cmd_blit_image(
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
}

impl Drop for AdCommandBuffer {
  fn drop(&mut self) {
    unsafe {
      self
        .get_ash_device()
        .free_command_buffers(self.cmd_pool.inner(), &[self.inner]);
    }
  }
}
