use std::{collections::HashMap, sync::Arc};

use ash_ad_wrappers::{ash_context::{ash::vk, AdAshDevice}, ash_data_wrappers::{AdBuffer, AdDescriptorBinding}, ash_queue_wrappers::AdCommandBuffer, ash_render_wrappers::{AdFrameBuffer, AdPipeline, AdRenderPass}};
use renderables::{depth_texture::{DepthTextureGPU, DepthTextureGenerator}, triangle_mesh::{TriMeshGPU, TriMeshGenerator}, Camera3D};

use include_bytes_aligned::include_bytes_aligned;

static DEPTH_VERT_SHADER_CODE: &[u8] = include_bytes_aligned!(4, "shaders/triangle_depth.vert.spv");
static DEPTH_FRAG_SHADER_CODE: &[u8] = include_bytes_aligned!(4, "shaders/triangle_depth.frag.spv");

pub struct TriMeshDepthRenderer {
  pipelines: Vec<AdPipeline>,
  render_pass: Arc<AdRenderPass>,
}

impl TriMeshDepthRenderer {
  pub fn new(
    ash_device: Arc<AdAshDevice>,
    tri_mesh_gen: &TriMeshGenerator,
  ) -> Result<Self, String> {
    let render_pass = AdRenderPass::new(
      ash_device.clone(),
      vk::RenderPassCreateFlags::default(),
      &[vk::AttachmentDescription::default()
        .format(vk::Format::R32_SFLOAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::CLEAR)],
      &[vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&[vk::AttachmentReference::default()
          .attachment(0)
          .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)])],
      &[
        vk::SubpassDependency::default()
          .src_subpass(vk::SUBPASS_EXTERNAL)
          .dst_subpass(0)
          .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
          .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
          .src_access_mask(vk::AccessFlags::SHADER_READ)
          .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
        vk::SubpassDependency::default()
          .src_subpass(0)
          .dst_subpass(vk::SUBPASS_EXTERNAL)
          .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
          .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
          .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
          .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
      ],
    )?;
    let render_pass = Arc::new(render_pass);

    let triangle_rasterizer_info = vk::PipelineRasterizationStateCreateInfo::default()
      .cull_mode(vk::CullModeFlags::BACK)
      .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
      .polygon_mode(vk::PolygonMode::FILL)
      .line_width(1.0);

    let pipeline = AdPipeline::new(
      render_pass.clone(),
      0,
      HashMap::from([
        (vk::ShaderStageFlags::VERTEX, DEPTH_VERT_SHADER_CODE),
        (vk::ShaderStageFlags::FRAGMENT, DEPTH_FRAG_SHADER_CODE),
      ]),
      &[tri_mesh_gen.mesh_dset_layout()],
      (vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, std::mem::size_of::<Camera3D>() as u32),
      triangle_rasterizer_info,
      &vk::PipelineColorBlendStateCreateInfo::default().attachments(&[
        vk::PipelineColorBlendAttachmentState::default()
          .color_write_mask(vk::ColorComponentFlags::RGBA)
          .blend_enable(false),
      ]),
    )?;

    Ok(Self { pipelines: vec![pipeline], render_pass })
  }

  pub fn create_framebuffers(
    &self,
    depth_textures: &[Arc<DepthTextureGPU>],
  ) -> Result<Vec<Arc<AdFrameBuffer>>, String> {
    let depth_set_binding_infos = depth_textures.iter().map(|d_tex| {
      let binding = d_tex.dset().bindings()[0].clone();
      if let AdDescriptorBinding::Sampler2D(img) = binding {
          Ok(img)
      } else {
        Err("invalid bindings for depth texture".to_string())
      }
    })
    .collect::<Result<Vec<_>,_>>()?;

    let triangle_frame_buffers = depth_set_binding_infos
      .iter()
      .map(|(iview, _, _)| {
        AdFrameBuffer::new(
          self.render_pass.clone(),
          vec![iview.clone()],
          vk::Extent2D {
            width: iview.image().resolution().width,
            height: iview.image().resolution().height,
          },
          1,
        )
      })
      .collect::<Result<Vec<_>, String>>()?;
    Ok(triangle_frame_buffers)
  }

  pub fn render(
    &self,
    cmd_buffer: &AdCommandBuffer,
    frame_buffer: &AdFrameBuffer,
    camera: Camera3D,
    objs: &[Arc<TriMeshGPU>],
  ) {
    cmd_buffer.begin_render_pass(
      self.render_pass.inner(),
      frame_buffer.inner(),
      vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: frame_buffer.resolution() },
      &[vk::ClearValue { color: vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 0.0] } }],
      vk::SubpassContents::INLINE,
    );
    cmd_buffer.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, self.pipelines[0].inner());

    cmd_buffer.set_view_port(&[vk::Viewport {
      x: 0.0,
      y: 0.0,
      width: frame_buffer.resolution().width as f32,
      height: frame_buffer.resolution().height as f32,
      min_depth: 0.0,
      max_depth: 1.0,
    }]);
    cmd_buffer.set_scissor(&[vk::Rect2D {
      offset: vk::Offset2D { x: 0, y: 0 },
      extent: frame_buffer.resolution(),
    }]);

    for obj in objs.iter() {
      cmd_buffer.bind_descriptor_sets(
        vk::PipelineBindPoint::GRAPHICS,
        self.pipelines[0].layout(),
        &[obj.dset().inner()],
      );
      cmd_buffer.set_push_constant_data(
        self.pipelines[0].layout(),
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        AdBuffer::get_byte_slice(&[camera]),
      );
      cmd_buffer.draw(obj.indx_count() as _);
    }
    cmd_buffer.end_render_pass();
  }
}
