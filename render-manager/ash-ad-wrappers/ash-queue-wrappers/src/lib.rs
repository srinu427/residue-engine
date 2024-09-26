use std::sync::Arc;

use ash_context::{ash::vk, AdAshDevice, getset};
use ash_sync_wrappers::{AdFence, AdSemaphore};

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdQueue {
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

  pub fn create_ad_command_pool(
    &self,
    flags: vk::CommandPoolCreateFlags,
  ) -> Result<AdCommandPool, String> {
    unsafe {
      let cmd_pool = self
        .ash_device
        .inner()
        .create_command_pool(
          &vk::CommandPoolCreateInfo::default().flags(flags).queue_family_index(self.family_index),
          None,
        )
        .map_err(|e| format!("at vk cmd pool create: {e}"))?;
      Ok(AdCommandPool {
        ash_device: self.ash_device.clone(),
        inner: cmd_pool,
        queue: self.inner,
        queue_family_index: self.family_index,
      })
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
  ash_device: Arc<AdAshDevice>,
  #[getset(get_copy = "pub")]
  inner: vk::CommandPool,
  queue: vk::Queue,
  #[getset(get_copy = "pub")]
  queue_family_index: u32,
}

impl AdCommandPool {
  pub fn allocate_command_buffers(
    &self,
    level: vk::CommandBufferLevel,
    count: u32,
  ) -> Result<Vec<AdCommandBuffer>, String> {
    let cmd_buffers = unsafe {
      self
        .ash_device
        .inner()
        .allocate_command_buffers(
          &vk::CommandBufferAllocateInfo::default()
            .command_pool(self.inner)
            .level(level)
            .command_buffer_count(count),
        )
        .map_err(|e| format!("at creating command buffer: {e}"))?
        .iter()
        .map(|&x| AdCommandBuffer {
          ash_device: self.ash_device.clone(),
          pool: self.inner,
          inner: x,
          queue: self.queue,
          queue_family_index: self.queue_family_index,
        })
        .collect::<Vec<_>>()
    };
    Ok(cmd_buffers)
  }
}

impl Drop for AdCommandPool {
  fn drop(&mut self) {
    unsafe {
      self
        .ash_device
        .inner()
        .destroy_command_pool(self.inner, None);
    }
  }
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdCommandBuffer {
  #[getset(get = "pub")]
  ash_device: Arc<AdAshDevice>,
  pool: vk::CommandPool,
  #[getset(get_copy = "pub")]
  inner: vk::CommandBuffer,
  queue: vk::Queue,
  #[getset(get_copy = "pub")]
  queue_family_index: u32,
}

impl AdCommandBuffer {
  pub fn begin(&self, flags: vk::CommandBufferUsageFlags) -> Result<(), String> {
    unsafe {
      self
        .ash_device
        .inner()
        .begin_command_buffer(self.inner, &vk::CommandBufferBeginInfo::default().flags(flags))
        .map_err(|e| format!("at cmd buffer begin: {e}"))
    }
  }

  pub fn end(&self) -> Result<(), String> {
    unsafe {
      self.ash_device.inner().end_command_buffer(self.inner).map_err(|e| format!("at cmd buffer end: {e}"))
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
        .ash_device
        .inner()
        .queue_submit(
          self.queue,
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
        .ash_device
        .inner()
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
      self.ash_device.inner().cmd_begin_render_pass(self.inner, &render_pass_begin_info, subpass_contents);
    }
  }

  pub fn end_render_pass(&self) {
    unsafe {
      self.ash_device.inner().cmd_end_render_pass(self.inner);
    }
  }

  pub fn bind_pipeline(&self, pipeline_bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline) {
    unsafe {
      self.ash_device.inner().cmd_bind_pipeline(self.inner, pipeline_bind_point, pipeline);
    }
  }

  pub fn bind_vertex_buffer(
    &self,
    binding_count: u32,
    buffers: &[vk::Buffer],
    offsets: &[vk::DeviceSize],
  ) {
    unsafe {
      self.ash_device.inner().cmd_bind_vertex_buffers(self.inner, binding_count, buffers, offsets);
    }
  }

  pub fn bind_index_buffer(
    &self,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    index_type: vk::IndexType,
  ) {
    unsafe {
      self.ash_device.inner().cmd_bind_index_buffer(self.inner, buffer, offset, index_type);
    }
  }

  pub fn bind_descriptor_sets(
    &self,
    pipeline_bind_point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    descriptor_sets: &[vk::DescriptorSet]
  ) {
    // let vk_descriptor_sets = descriptor_sets.iter().map(|x| x.inner).collect::<Vec<_>>();
    unsafe {
      self.ash_device.inner().cmd_bind_descriptor_sets(self.inner, pipeline_bind_point, layout, 0, &descriptor_sets, &[])
    }
  }

  pub fn set_view_port(&self, viewports: &[vk::Viewport]) {
    unsafe {
      self.ash_device.inner().cmd_set_viewport(self.inner, 0, viewports);
    }
  }

  pub fn set_scissor(&self, scissors: &[vk::Rect2D]) {
    unsafe {
      self.ash_device.inner().cmd_set_scissor(self.inner, 0, scissors);
    }
  }

  pub fn draw(&self, vert_count: u32) {
    unsafe {
      self.ash_device.inner().cmd_draw(self.inner, vert_count, 1, 0, 0);
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
      self.ash_device.inner().cmd_pipeline_barrier(
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
      self.ash_device.inner().cmd_blit_image(
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
      self.ash_device.inner().cmd_copy_buffer_to_image(
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
      self.ash_device.inner().free_command_buffers(self.pool, &[self.inner]);
    }
  }
}
