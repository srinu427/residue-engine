use std::sync::Arc;

use ash_common_imports::ash::{self, vk};
use ash_data_wrappers::AdBuffer;
use ash_sync_wrappers::{AdFence, AdSemaphore};

pub struct AdCommandPool {
  pub(crate) vk_device: Arc<ash::Device>,
  pub inner: vk::CommandPool,
  pub(crate) queue: vk::Queue,
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
          vk_device: self.vk_device.clone(),
          pool: self.inner,
          inner: x,
          queue: self.queue,
          qf_idx: self.qf_idx,
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
  pub(crate) queue: vk::Queue,
  pub qf_idx: u32,
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

  pub fn submit(
    &self,
    signal_semaphores: &[&AdSemaphore],
    wait_semaphores: &[(&AdSemaphore, vk::PipelineStageFlags)],
    fence: Option<&AdFence>
  ) -> Result<(), String> {
    unsafe {
      self
        .vk_device
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

  pub fn bind_descriptor_sets(
    &self,
    pipeline_bind_point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    descriptor_sets: &[vk::DescriptorSet]
  ) {
    // let vk_descriptor_sets = descriptor_sets.iter().map(|x| x.inner).collect::<Vec<_>>();
    unsafe {
      self.vk_device.cmd_bind_descriptor_sets(self.inner, pipeline_bind_point, layout, 0, &descriptor_sets, &[])
    }
  }

  pub fn set_view_port(&self, viewports: &[vk::Viewport]) {
    unsafe {
      self.vk_device.cmd_set_viewport(self.inner, 0, viewports);
    }
  }

  pub fn set_scissor(&self, scissors: &[vk::Rect2D]) {
    unsafe {
      self.vk_device.cmd_set_scissor(self.inner, 0, scissors);
    }
  }

  pub fn draw(&self, vert_count: u32) {
    unsafe {
      self.vk_device.cmd_draw(self.inner, vert_count, 1, 0, 0);
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

  pub fn copy_buffer_to_buffer(
    &self,
    src_buffer: &AdBuffer,
    dst_buffer: &AdBuffer,
    regions: &[vk::BufferCopy]
  ) {
    unsafe {
      self.vk_device.cmd_copy_buffer(self.inner, src_buffer.inner(), dst_buffer.inner(), regions);
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

#[derive(Hash, PartialEq, Eq, Copy, Clone)]
pub enum GPUQueueType {
  Graphics,
  Compute,
  Transfer,
  Present,
}

pub struct AdQueue {
  vk_device: Arc<ash::Device>,
  qf_idx: u32,
  inner: vk::Queue,
}

impl AdQueue {
  pub fn new(vk_device: Arc<ash::Device>, qf_idx: u32, vk_queue: vk::Queue) -> Self {
    Self { vk_device, qf_idx, inner: vk_queue }
  }

  pub fn inner(&self) -> vk::Queue {
    self.inner
  }

  pub fn family_idx(&self) -> u32 {
    self.qf_idx
  }

  pub fn create_ad_command_pool(
    &self,
    flags: vk::CommandPoolCreateFlags,
  ) -> Result<AdCommandPool, String> {
    unsafe {
      let cmd_pool = self
        .vk_device
        .create_command_pool(
          &vk::CommandPoolCreateInfo::default().flags(flags).queue_family_index(self.qf_idx),
          None,
        )
        .map_err(|e| format!("at vk cmd pool create: {e}"))?;
      Ok(AdCommandPool {
        vk_device: self.vk_device.clone(),
        inner: cmd_pool,
        queue: self.inner,
        qf_idx: self.qf_idx,
      })
    }
  }

  pub fn submit(
    &self,
    submits: &[vk::SubmitInfo],
    fence: &AdFence
  ) -> Result<(), String> {
    unsafe {
      self
        .vk_device
        .queue_submit(self.inner, submits, fence.inner())
        .map_err(|e| format!("error submitting to queue: {e}"))
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