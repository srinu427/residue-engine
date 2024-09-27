use std::{collections::HashMap, path::PathBuf, sync::{Arc, Mutex}};

use ash_ad_wrappers::{
  ash_context::{ash::vk, gpu_allocator::vulkan::Allocator, AdAshDevice, GPUQueueType},
  ash_data_wrappers::{AdDescriptorBinding, AdDescriptorPool, AdDescriptorSet, AdDescriptorSetLayout},
  ash_queue_wrappers::{AdCommandBuffer, AdCommandPool, AdQueue}, ash_render_wrappers::{AdPipeline, AdRenderPass}
};

#[repr(C)]
pub struct TriMeshVertex {
  pub pos: [f32; 4],
  pub uv: [f32; 4],
}

pub struct TriMeshCPU {
  pub verts: Vec<TriMeshVertex>,
  pub triangles: Vec<[u32; 3]>,
}

impl TriMeshCPU {

}

pub struct TriMesh {
  indx_len: u32,
  dset: AdDescriptorSet,
}

pub struct TriMeshRenderer {
  meshes: Vec<TriMesh>,
  mesh_allocator: Arc<Mutex<Allocator>>,
  cmd_pool: AdCommandPool,
  pipeline: AdPipeline,
  pub render_pass: AdRenderPass,
  vert_dset_layout: AdDescriptorSetLayout,
  dset_pool: AdDescriptorPool,
  ash_device: Arc<AdAshDevice>,
}

impl TriMeshRenderer {
  pub fn new(
    ash_device: Arc<AdAshDevice>,
    transfer_queue: Arc<AdQueue>,
    cam_dset_layout: &AdDescriptorSetLayout
  ) -> Result<Self, String> {
    let mesh_allocator = Arc::new(Mutex::new(ash_device.create_allocator()?));
    let cmd_pool = ash_device.queues[&GPUQueueType::Transfer]
      .create_ad_command_pool(vk::CommandPoolCreateFlags::TRANSIENT)?;
    let vert_dset_layout = AdDescriptorSetLayout::new(
      ash_device.clone(),
      &[
        (vk::ShaderStageFlags::VERTEX, AdDescriptorBinding::StorageBuffer(vec![None])),
        (vk::ShaderStageFlags::VERTEX, AdDescriptorBinding::StorageBuffer(vec![None])),
      ]
    )?;
    let dset_pool = AdDescriptorPool::new(
      ash_device.clone(),
      vk::DescriptorPoolCreateFlags::default(),
      2000,
      &[vk::DescriptorPoolSize { descriptor_count: 2000, ty: vk::DescriptorType::STORAGE_BUFFER }],
    )?;
    let render_pass = ash_device
      .create_ad_render_pass_builder(vk::RenderPassCreateFlags::default())
      .add_attachment(
        vk::AttachmentDescription::default()
          .format(vk::Format::R8G8B8A8_UNORM)
          .samples(vk::SampleCountFlags::TYPE_1)
          .initial_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
          .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
          .load_op(vk::AttachmentLoadOp::CLEAR),
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

    let triangle_rasterizer_info = vk::PipelineRasterizationStateCreateInfo::default()
      .cull_mode(vk::CullModeFlags::NONE)
      .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
      .polygon_mode(vk::PolygonMode::FILL)
      .line_width(1.0);

    let pipeline = AdPipeline::new(
      render_pass.clone(),
      0,
      HashMap::from([
        (vk::ShaderStageFlags::VERTEX, PathBuf::from("render-manager/shaders/triangle.vert.spv")),
        (vk::ShaderStageFlags::FRAGMENT, PathBuf::from("render-manager/shaders/triangle.frag.spv")),
      ]),
      &[vert_dset_layout, cam_dset_layout],
      triangle_rasterizer_info,
      &vk::PipelineColorBlendStateCreateInfo::default().attachments(&[
        vk::PipelineColorBlendAttachmentState::default()
          .color_write_mask(vk::ColorComponentFlags::RGBA)
          .blend_enable(false),
      ]),
    )?;
    
    Ok(Self { meshes: vec![], mesh_allocator, render_pass, pipeline, vert_dset_layout, dset_pool, ash_device, cmd_pool })
  }

  pub fn add_mesh(&mut self, name: &str, cpu_mesh: &TriMeshCPU) -> Result<(), String>{
    let tmp_cmd_buffer =
      self.cmd_pool.allocate_command_buffers(vk::CommandBufferLevel::PRIMARY, 1)?.remove(0);
    let vb_size = std::mem::size_of::<TriMeshVertex>() * cpu_mesh.verts.len();
    let vert_buffer = self.ash_device.create_ad_buffer_from_data(
      self.mesh_allocator.clone(),
      ash_wrappers::MemoryLocation::GpuOnly,
      name,
      vk::BufferCreateFlags::empty(),
      vk::BufferUsageFlags::STORAGE_BUFFER,
      unsafe { std::slice::from_raw_parts(cpu_mesh.verts.as_ptr() as *const u8, vb_size) },
      &tmp_cmd_buffer
    )?;

    let tmp_cmd_buffer =
      self.cmd_pool.allocate_command_buffers(vk::CommandBufferLevel::PRIMARY, 1)?.remove(0);
    let indices = cpu_mesh.triangles.iter().flatten().cloned().collect::<Vec<_>>();
    let ib_size = std::mem::size_of::<u32>() * indices.len();
    let indx_buffer = self.ash_device.create_ad_buffer_from_data(
      self.mesh_allocator.clone(),
      MemoryLocation::GpuOnly,
      name,
      vk::BufferCreateFlags::empty(),
      vk::BufferUsageFlags::STORAGE_BUFFER,
      unsafe { std::slice::from_raw_parts(indices.as_ptr() as *const u8, ib_size) },
      &tmp_cmd_buffer
    )?;

    let dset = self.dset_pool.allocate_owned_dset(
      &self.vert_dset_layout,
      HashMap::from([
        (0, OwnedDSetBinding::StorageBuffer(vec![vert_buffer])),
        (1, OwnedDSetBinding::StorageBuffer(vec![indx_buffer])),
      ]))?;
    self.meshes.push(TriMesh { indx_len: indices.len() as u32, dset });
    Ok(())
  }

  pub fn render_meshes(&self, cmd_buffer: &AdCommandBuffer, frame_buffer: &AdFrameBuffer, camera_dset: vk::DescriptorSet) {
    cmd_buffer.begin_render_pass(
      vk::RenderPassBeginInfo::default()
        .render_pass(self.render_pass.inner())
        .render_area(vk::Rect2D {
          offset: vk::Offset2D { x: 0, y: 0 },
          extent: vk::Extent2D { width: 800, height: 600 },
        })
        .framebuffer(frame_buffer.inner)
        .clear_values(&[vk::ClearValue {
          color: vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 0.0] },
        }]),
      vk::SubpassContents::INLINE,
    );
    cmd_buffer
      .bind_pipeline(vk::PipelineBindPoint::GRAPHICS, self.pipeline.inner);

    cmd_buffer.set_view_port(&[vk::Viewport {
      x: 0.0,
      y: 0.0,
      width: 800.0,
      height: 600.0,
      min_depth: 0.0,
      max_depth: 1.0,
    }]);
    cmd_buffer.set_scissor(&[vk::Rect2D {
      offset: vk::Offset2D { x: 0, y: 0 },
      extent: vk::Extent2D { width: 800, height: 600 },
    }]);

    for mesh in self.meshes.iter() {
      cmd_buffer.bind_descriptor_sets(
        vk::PipelineBindPoint::GRAPHICS,
        self.pipeline.layout,
        &[mesh.dset.inner, camera_dset],
      );
      cmd_buffer.draw(mesh.indx_len);
    }
    cmd_buffer.end_render_pass();
  }
}
