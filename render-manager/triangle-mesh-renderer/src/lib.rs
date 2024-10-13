use std::{
  collections::HashMap,
  path::PathBuf,
  sync::{Arc, Mutex},
};

use ash_ad_wrappers::{
  ash_context::{
    ash::vk,
    gpu_allocator::{vulkan::Allocator, MemoryLocation},
    AdAshDevice,
  },
  ash_data_wrappers::{
    AdBuffer, AdDescriptorBinding, AdDescriptorPool, AdDescriptorSet, AdDescriptorSetLayout,
    AdImage, AdImageView, AdSampler,
  },
  ash_queue_wrappers::{AdCommandBuffer, AdCommandPool, AdQueue},
  ash_render_wrappers::{AdFrameBuffer, AdPipeline, AdRenderPass},
  ash_sync_wrappers::AdFence,
};

pub use glam;

pub fn g_vec4_from_vec3(v: glam::Vec3, w: f32) -> glam::Vec4 {
  glam::vec4(v.x, v.y, v.z, w)
}


#[repr(C)]
pub struct TriMeshVertex {
  pub pos: glam::Vec4,
  pub normal: glam::Vec4,
  pub uv: glam::Vec4,
}

pub struct TriMeshCPU {
  pub verts: Vec<TriMeshVertex>,
  pub triangles: Vec<[u32; 3]>,
}

impl TriMeshCPU {
  pub fn merge(&mut self, mut other: Self) {
    let curr_vert_len = self.verts.len() as u32;
    for t in other.triangles.iter_mut() {
      for idx in t {
        *idx += curr_vert_len;
      }
    }
    self.verts.append(&mut other.verts);
    self.triangles.append(&mut other.triangles);
  }

  pub fn make_rect(center: glam::Vec3, tangent: glam::Vec3, bitangent: glam::Vec3) -> Self {
    let normal = tangent.cross(bitangent).normalize();
    let verts = vec![
      TriMeshVertex {
        pos: g_vec4_from_vec3(center + tangent/2.0 - bitangent/2.0, 1.0),
        normal: g_vec4_from_vec3(normal, 1.0),
        uv: glam::vec4(0.0, 0.0, 0.0, 0.0)
      },
      TriMeshVertex {
        pos: g_vec4_from_vec3(center - tangent/2.0 - bitangent/2.0, 1.0),
        normal: g_vec4_from_vec3(normal, 1.0),
        uv: glam::vec4(0.0, bitangent.length(), 0.0, 0.0)
      },
      TriMeshVertex {
        pos: g_vec4_from_vec3(center - tangent/2.0 + bitangent/2.0, 1.0),
        normal: g_vec4_from_vec3(normal, 1.0),
        uv: glam::vec4(tangent.length(), bitangent.length(), 0.0, 0.0)
      },
      TriMeshVertex {
        pos: g_vec4_from_vec3(center + tangent/2.0 + bitangent/2.0, 1.0),
        normal: g_vec4_from_vec3(normal, 1.0),
        uv: glam::vec4(tangent.length(), 0.0, 0.0, 0.0)
      },
    ];
    let triangles = vec![[0, 1, 2], [2, 3, 0]];
    Self { verts, triangles }
  }
}

pub struct TriMesh {
  indx_len: u32,
  dset: AdDescriptorSet,
}

pub struct TriRenderable {
  mesh: TriMesh,
  texture: Arc<AdDescriptorSet>,
}

pub struct TriMeshRenderer {
  renderables: Vec<TriRenderable>,
  textures: HashMap<String, Arc<AdDescriptorSet>>,
  mesh_allocator: Arc<Mutex<Allocator>>,
  cmd_pool: Arc<AdCommandPool>,
  pipeline: AdPipeline,
  pub render_pass: Arc<AdRenderPass>,
  tex_sampler: Arc<AdSampler>,
  vert_dset_layout: Arc<AdDescriptorSetLayout>,
  tex_dset_layout: Arc<AdDescriptorSetLayout>,
  dset_pool: Arc<AdDescriptorPool>,
  ash_device: Arc<AdAshDevice>,
}

impl TriMeshRenderer {
  pub fn new(
    ash_device: Arc<AdAshDevice>,
    transfer_queue: Arc<AdQueue>,
    cam_dset_layout: &AdDescriptorSetLayout,
  ) -> Result<Self, String> {
    let mesh_allocator = Arc::new(Mutex::new(ash_device.create_allocator()?));
    let cmd_pool =
      Arc::new(AdCommandPool::new(transfer_queue, vk::CommandPoolCreateFlags::TRANSIENT)?);
    let vert_dset_layout = Arc::new(AdDescriptorSetLayout::new(
      ash_device.clone(),
      &[
        (vk::ShaderStageFlags::VERTEX, AdDescriptorBinding::StorageBuffer(vec![None])),
        (vk::ShaderStageFlags::VERTEX, AdDescriptorBinding::StorageBuffer(vec![None])),
      ],
    )?);
    let tex_dset_layout = Arc::new(AdDescriptorSetLayout::new(
      ash_device.clone(),
      &[(vk::ShaderStageFlags::FRAGMENT, AdDescriptorBinding::Sampler2D(vec![None]))],
    )?);
    let tex_sampler = Arc::new(AdSampler::new(ash_device.clone())?);
    let dset_pool = Arc::new(AdDescriptorPool::new(
      ash_device.clone(),
      vk::DescriptorPoolCreateFlags::default(),
      2000,
      &[
        vk::DescriptorPoolSize { descriptor_count: 2000, ty: vk::DescriptorType::STORAGE_BUFFER },
        vk::DescriptorPoolSize { descriptor_count: 2000, ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER },
      ],
    )?);
    let render_pass = Arc::new(AdRenderPass::new(
      ash_device.clone(),
      vk::RenderPassCreateFlags::default(),
      &[vk::AttachmentDescription::default()
        .format(vk::Format::R8G8B8A8_UNORM)
        .samples(vk::SampleCountFlags::TYPE_1)
        .initial_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
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
    )?);

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
      &[&vert_dset_layout, cam_dset_layout, &tex_dset_layout],
      triangle_rasterizer_info,
      &vk::PipelineColorBlendStateCreateInfo::default().attachments(&[
        vk::PipelineColorBlendAttachmentState::default()
          .color_write_mask(vk::ColorComponentFlags::RGBA)
          .blend_enable(false),
      ]),
    )?;

    Ok(Self {
      renderables: vec![],
      textures: HashMap::new(),
      mesh_allocator,
      render_pass,
      pipeline,
      tex_sampler,
      vert_dset_layout,
      tex_dset_layout,
      dset_pool,
      ash_device,
      cmd_pool,
    })
  }

  pub fn add_texture(&mut self, name: &str, path: &str, _replace: bool) -> Result<Arc<AdDescriptorSet>, String> {
    if self.textures.contains_key(name) {
      return self.textures.get(name).ok_or("can't get tex from memory".to_string()).map(|x| x.clone());
    }
    let cmd_buffer = AdCommandBuffer::new(self.cmd_pool.clone(), vk::CommandBufferLevel::PRIMARY, 1)?.remove(0);
    let albedo = AdImage::new_2d_from_file(
      self.ash_device.clone(),
      self.mesh_allocator.clone(),
      MemoryLocation::GpuOnly,
      name,
      vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
      path,
      &cmd_buffer,
      vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    )?;
    let albedo_local = AdImageView::create_view(
      albedo,
      vk::ImageViewType::TYPE_2D,
      vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .base_mip_level(0)
        .level_count(1)
    )?;
    let mut dset = AdDescriptorSet::new(self.dset_pool.clone(), &[&self.tex_dset_layout])?.remove(0);
    dset.set_binding(0, AdDescriptorBinding::Sampler2D(vec![Some((albedo_local, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, self.tex_sampler.clone()))]));
    let dset = Arc::new(dset);
    self.textures.insert(name.to_string(), dset.clone());
    Ok(dset)
  }

  pub fn add_renderable(&mut self, name: &str, cpu_mesh: &TriMeshCPU, texture: (&str, &str)) -> Result<(), String> {
    let tmp_cmd_buffer =
      AdCommandBuffer::new(self.cmd_pool.clone(), vk::CommandBufferLevel::PRIMARY, 1)?.remove(0);
    let vert_buffer = Arc::new(AdBuffer::from_data(
      self.ash_device.clone(),
      self.mesh_allocator.clone(),
      MemoryLocation::GpuOnly,
      name,
      vk::BufferCreateFlags::empty(),
      vk::BufferUsageFlags::STORAGE_BUFFER,
      &cpu_mesh.verts,
      &tmp_cmd_buffer,
    )?);

    let tmp_cmd_buffer =
      AdCommandBuffer::new(self.cmd_pool.clone(), vk::CommandBufferLevel::PRIMARY, 1)?.remove(0);
    let indices = cpu_mesh.triangles.iter().flatten().cloned().collect::<Vec<_>>();
    let indx_buffer = Arc::new(AdBuffer::from_data(
      self.ash_device.clone(),
      self.mesh_allocator.clone(),
      MemoryLocation::GpuOnly,
      name,
      vk::BufferCreateFlags::empty(),
      vk::BufferUsageFlags::STORAGE_BUFFER,
      &indices,
      &tmp_cmd_buffer,
    )?);

    let mut dset =
      AdDescriptorSet::new(self.dset_pool.clone(), &[&self.vert_dset_layout])?.remove(0);
    dset.set_binding(0, AdDescriptorBinding::StorageBuffer(vec![Some(vert_buffer)]));
    dset.set_binding(1, AdDescriptorBinding::StorageBuffer(vec![Some(indx_buffer)]));

    let texture = self.add_texture(texture.0, texture.1, false)?;

    self.renderables.push( TriRenderable { mesh: TriMesh { indx_len: indices.len() as u32, dset }, texture});
    Ok(())
  }

  pub fn render_meshes(
    &self,
    cmd_buffer: &AdCommandBuffer,
    frame_buffer: &AdFrameBuffer,
    camera_dset: vk::DescriptorSet,
  ) {
    cmd_buffer.begin_render_pass(
      self.render_pass.inner(),
      frame_buffer.inner(),
      vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: frame_buffer.resolution() },
      &[vk::ClearValue { color: vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 0.0] } }],
      vk::SubpassContents::INLINE,
    );
    cmd_buffer.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, self.pipeline.inner());

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

    for renderable in self.renderables.iter() {
      cmd_buffer.bind_descriptor_sets(
        vk::PipelineBindPoint::GRAPHICS,
        self.pipeline.layout(),
        &[renderable.mesh.dset.inner(), camera_dset, renderable.texture.inner()],
      );
      cmd_buffer.draw(renderable.mesh.indx_len);
    }
    cmd_buffer.end_render_pass();
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
        AdImage::new_2d(
          self.ash_device.clone(),
          allocator.clone(),
          MemoryLocation::GpuOnly,
          &format!("triangle_out_image_temp_{i}"),
          vk::Format::R8G8B8A8_UNORM,
          resolution,
          vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::COLOR_ATTACHMENT,
          vk::SampleCountFlags::TYPE_1,
          1,
        )
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
            .src_queue_family_index(cmd_buffer.cmd_pool().queue().family_index())
            .dst_queue_family_index(cmd_buffer.cmd_pool().queue().family_index())
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        })
        .collect::<Vec<_>>(),
    );
    cmd_buffer.end()?;
    let fence = AdFence::new(self.ash_device.clone(), vk::FenceCreateFlags::empty())?;
    cmd_buffer.submit(&[], &[], Some(&fence))?;
    fence.wait(999999999)?;
    fence.reset()?;

    let triangle_out_image_views = (0..3)
      .map(|i| {
        AdImageView::create_view(
          triangle_out_images[i].clone(),
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

    let triangle_frame_buffers = (0..3)
      .map(|i| {
        AdFrameBuffer::new(
          self.render_pass.clone(),
          vec![triangle_out_image_views[i].clone()],
          resolution,
          1,
        )
      })
      .collect::<Result<Vec<_>, String>>()?;
    Ok(triangle_frame_buffers)
  }
}
