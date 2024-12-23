use std::{
  collections::HashMap,
  sync::{Arc, Mutex},
};

use ash_ad_wrappers::{
  ash_context::{
    ash::vk,
    gpu_allocator::{vulkan::Allocator, MemoryLocation},
    AdAshDevice,
  },
  ash_data_wrappers::{AdBuffer, AdImage, AdImageView},
  ash_queue_wrappers::AdCommandBuffer,
  ash_render_wrappers::{AdFrameBuffer, AdPipeline, AdRenderPass},
  ash_sync_wrappers::AdFence,
};
use include_bytes_aligned::include_bytes_aligned;
use renderables::{
  flat_texture::{FlatTextureGPU, FlatTextureGenerator}, triangle_mesh::{TriMeshGPU, TriMeshGenerator}, Camera3D
};

static FTEX_VERT_SHADER_CODE: &[u8] = include_bytes_aligned!(4, "shaders/triangle.vert.spv");
static FTEX_FRAG_SHADER_CODE: &[u8] = include_bytes_aligned!(4, "shaders/triangle_flat_tex.frag.spv");

pub struct TriMeshFlatTex {
  pub mesh: Arc<TriMeshGPU>,
  pub ftex: Arc<FlatTextureGPU>,
}

pub struct TriMeshTexRenderer {
  pipelines: Vec<AdPipeline>,
  render_pass: Arc<AdRenderPass>,
  depth_format: vk::Format,
}

impl TriMeshTexRenderer {
  pub fn new(
    ash_device: Arc<AdAshDevice>,
    tri_mesh_gen: &TriMeshGenerator,
    flat_tex_gen: &FlatTextureGenerator,
    depth_format: vk::Format,
  ) -> Result<Self, String> {
    let render_pass = AdRenderPass::new(
      ash_device.clone(),
      vk::RenderPassCreateFlags::default(),
      &[vk::AttachmentDescription::default()
          .format(vk::Format::R8G8B8A8_UNORM)
          .samples(vk::SampleCountFlags::TYPE_1)
          .initial_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
          .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
          .load_op(vk::AttachmentLoadOp::CLEAR),
        vk::AttachmentDescription::default()
          .format(depth_format)
          .samples(vk::SampleCountFlags::TYPE_1)
          .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
          .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
          .load_op(vk::AttachmentLoadOp::CLEAR)
          .store_op(vk::AttachmentStoreOp::DONT_CARE)],
      &[vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&[vk::AttachmentReference::default()
          .attachment(0)
          .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)])
        .depth_stencil_attachment(&vk::AttachmentReference::default()
          .attachment(1)
          .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL))],
      &[
        vk::SubpassDependency::default()
          .src_subpass(vk::SUBPASS_EXTERNAL)
          .dst_subpass(0)
          .src_stage_mask(vk::PipelineStageFlags::TRANSFER)
          .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
          .src_access_mask(vk::AccessFlags::TRANSFER_READ)
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
        (vk::ShaderStageFlags::VERTEX, FTEX_VERT_SHADER_CODE),
        (vk::ShaderStageFlags::FRAGMENT, FTEX_FRAG_SHADER_CODE),
      ]),
      &[tri_mesh_gen.mesh_dset_layout(), flat_tex_gen.tex_dset_layout()],
      (vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, std::mem::size_of::<Camera3D>() as u32),
      triangle_rasterizer_info,
      &vk::PipelineColorBlendStateCreateInfo::default().attachments(&[
        vk::PipelineColorBlendAttachmentState::default()
          .color_write_mask(vk::ColorComponentFlags::RGBA)
          .blend_enable(false),
      ]),
      &vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
    )?;

    Ok(Self { pipelines: vec![pipeline], render_pass, depth_format })
  }

  pub fn create_framebuffers(
    &self,
    cmd_buffer: &AdCommandBuffer,
    allocator: Arc<Mutex<Allocator>>,
    resolution: vk::Extent2D,
    count: usize,
  ) -> Result<Vec<Arc<AdFrameBuffer>>, String> {
    let triangle_out_images = (0..count)
      .map(|i| {
        let Ok(color_img) = AdImage::new_2d(
          self.render_pass.ash_device().clone(),
          allocator.clone(),
          MemoryLocation::GpuOnly,
          &format!("triangle_color_image_temp_{i}"),
          vk::Format::R8G8B8A8_UNORM,
          resolution,
          vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::COLOR_ATTACHMENT,
          vk::SampleCountFlags::TYPE_1,
          1,
        ) else {
          return Err(format!("failed to create color image {}", i));
        };
        let Ok(depth_img) = AdImage::new_2d(
          self.render_pass.ash_device().clone(),
          allocator.clone(),
          MemoryLocation::GpuOnly,
          &format!("triangle_depth_image_temp_{i}"),
          self.depth_format,
          resolution,
          vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
          vk::SampleCountFlags::TYPE_1,
          1,
        ) else {
          return Err(format!("failed to create depth image {}", i))
        };
        Ok((color_img, depth_img))
      })
      .collect::<Result<Vec<_>, _>>()?;

    cmd_buffer.begin(vk::CommandBufferUsageFlags::default())?;
    cmd_buffer.pipeline_barrier(
      vk::PipelineStageFlags::TRANSFER,
      vk::PipelineStageFlags::TRANSFER,
      vk::DependencyFlags::BY_REGION,
      &[],
      &[],
      &triangle_out_images
        .iter()
        .map(|(color_img, _)| {
          vk::ImageMemoryBarrier::default()
            .image(color_img.inner())
            .subresource_range(
              vk::ImageSubresourceRange::default()
                .aspect_mask(color_img.possible_image_aspect())
                .layer_count(1)
                .base_array_layer(0)
                .level_count(1)
                .base_mip_level(0),
            )
            .src_queue_family_index(cmd_buffer.cmd_pool().queue().family_index())
            .dst_queue_family_index(cmd_buffer.cmd_pool().queue().family_index())
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        })
        .collect::<Vec<_>>(),
    );
    cmd_buffer.pipeline_barrier(
      vk::PipelineStageFlags::TOP_OF_PIPE,
      vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
      vk::DependencyFlags::BY_REGION,
      &[],
      &[],
      &triangle_out_images
        .iter()
        .map(|(_, depth_img)| {
          vk::ImageMemoryBarrier::default()
            .image(depth_img.inner())
            .subresource_range(
              vk::ImageSubresourceRange::default()
                .aspect_mask(depth_img.possible_image_aspect())
                .layer_count(1)
                .base_array_layer(0)
                .level_count(1)
                .base_mip_level(0),
            )
            .src_queue_family_index(cmd_buffer.cmd_pool().queue().family_index())
            .dst_queue_family_index(cmd_buffer.cmd_pool().queue().family_index())
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        })
        .collect::<Vec<_>>(),
    );
    cmd_buffer.end()?;
    let fence = AdFence::new(self.render_pass.ash_device().clone(), vk::FenceCreateFlags::empty())?;
    cmd_buffer.submit(&[], &[], Some(&fence))?;
    fence.wait(999999999)?;
    fence.reset()?;

    let triangle_color_image_views = (0..3)
      .map(|i| {
        AdImageView::create_view(
          triangle_out_images[i].0.clone(),
          vk::ImageViewType::TYPE_2D,
          vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
          },
        )
      })
      .collect::<Result<Vec<_>, _>>()?;
    let triangle_depth_image_views = (0..3)
      .map(|i| {
        AdImageView::create_view(
          triangle_out_images[i].1.clone(),
          vk::ImageViewType::TYPE_2D,
          vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
          },
        )
      })
      .collect::<Result<Vec<_>, _>>()?;

    let triangle_frame_buffers = (0..3)
      .map(|i| {
        AdFrameBuffer::new(
          self.render_pass.clone(),
          vec![triangle_color_image_views[i].clone(), triangle_depth_image_views[i].clone()],
          resolution,
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
    objs: &[(Arc<TriMeshGPU>, Arc<FlatTextureGPU>)],
  ) {
    cmd_buffer.begin_render_pass(
      self.render_pass.inner(),
      frame_buffer.inner(),
      vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: frame_buffer.resolution() },
      &[
        vk::ClearValue { color: vk::ClearColorValue { float32: [0.1, 0.1, 0.1, 0.0] } },
        vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
      ],
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
        &[obj.0.dset().inner(), obj.1.dset().inner()],
      );
      cmd_buffer.set_push_constant_data(
        self.pipelines[0].layout(),
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        AdBuffer::get_byte_slice(&[camera]),
      );
      cmd_buffer.draw(obj.0.indx_count() as _);
    }
    cmd_buffer.end_render_pass();
  }
}
