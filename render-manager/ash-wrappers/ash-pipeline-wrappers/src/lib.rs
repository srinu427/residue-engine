use std::{collections::HashMap, sync::Arc};

use ash_common_imports::ash::{self, vk};
use ash_data_wrappers::AdImageView;

// Auto destroying wrapper for a Render Pass
pub struct AdRenderPass {
  vk_device: Arc<ash::Device>,
  inner: vk::RenderPass,
}

impl AdRenderPass {
  pub fn new(vk_device: Arc<ash::Device>, vk_render_pass: vk::RenderPass) -> Self {
    Self { vk_device, inner: vk_render_pass }
  }

  pub fn inner(&self) -> vk::RenderPass {
    self.inner
  }

  pub fn create_ad_g_pipeline(
    &self,
    subpass_id: u32,
    set_layouts: &[&AdDescriptorSetLayout],
    shaders: HashMap<vk::ShaderStageFlags, &AdShaderModule>,
    rasterizer_config: vk::PipelineRasterizationStateCreateInfo,
    blend_info: &vk::PipelineColorBlendStateCreateInfo,
  ) -> Result<AdPipeline, String> {
    let empty_vert_input_info = vk::PipelineVertexInputStateCreateInfo::default();
    let triangle_input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::default()
      .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let pipeline_dyn_state = vk::PipelineDynamicStateCreateInfo::default()
      .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
    let pipeline_vp_state = vk::PipelineViewportStateCreateInfo::default()
      .scissor_count(1)
      .viewport_count(1);
    let msaa_state = vk::PipelineMultisampleStateCreateInfo::default()
      .sample_shading_enable(false)
      .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let shader_stages = shaders.iter().map(|(stage, shader_mod)| {
      vk::PipelineShaderStageCreateInfo::default()
        .module(shader_mod.inner)
        .stage(*stage)
        .name(c"main")
    }).collect::<Vec<_>>();

    let pipeline_layout = unsafe {
      self.vk_device.create_pipeline_layout(
        &vk::PipelineLayoutCreateInfo::default()
          .set_layouts(&set_layouts.iter().map(|x| x.inner).collect::<Vec<_>>()),
        None
      )
      .map_err(|e| format!("at creating vk pipeline layout: {e}"))?
    };

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
      .render_pass(self.inner)
      .subpass(subpass_id)
      .layout(pipeline_layout)
      .stages(&shader_stages)
      .vertex_input_state(&empty_vert_input_info)
      .input_assembly_state(&triangle_input_assembly_info)
      .dynamic_state(&pipeline_dyn_state)
      .viewport_state(&pipeline_vp_state)
      .multisample_state(&msaa_state)
      .color_blend_state(&blend_info)
      .rasterization_state(&rasterizer_config);

    let pipeline = unsafe {
      self.vk_device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
        .map_err(|(_, e)| format!("at creating vk pipeline: {e}"))?
        .swap_remove(0)
    };
    Ok(AdPipeline {
      vk_device: self.vk_device.clone(),
      layout: pipeline_layout,
      inner: pipeline,
    })
  }

  pub fn create_frame_buffer(&self, attachment_views: &[&AdImageView], resolution: vk::Extent2D, layers: u32)
    -> Result<AdFrameBuffer, String> {
    let attachments = attachment_views.iter().map(|x| x.inner()).collect::<Vec<_>>();
    let frame_buffer_create_info = vk::FramebufferCreateInfo::default()
      .render_pass(self.inner)
      .attachments(&attachments)
      .width(resolution.width)
      .height(resolution.height)
      .layers(layers);
    let frame_buffer = unsafe {
      self.vk_device.create_framebuffer(&frame_buffer_create_info, None)
        .map_err(|e| format!("at creating vk frame buffer: {e}"))?
    };
    Ok(AdFrameBuffer { vk_device: self.vk_device.clone(), inner: frame_buffer })
  }
}

impl Drop for AdRenderPass {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.destroy_render_pass(self.inner, None);
    }
  }
}

pub struct AdFrameBuffer {
  pub(crate) vk_device: Arc<ash::Device>,
  pub inner: vk::Framebuffer
}

impl Drop for AdFrameBuffer {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.destroy_framebuffer(self.inner, None);
    }
  }
}

pub struct AdDescriptorSetLayout {
  vk_device: Arc<ash::Device>,
  inner: vk::DescriptorSetLayout,
}

impl AdDescriptorSetLayout {
  pub fn new(vk_device: Arc<ash::Device>, vk_dset_layout: vk::DescriptorSetLayout) -> Self {
    Self { vk_device, inner: vk_dset_layout }
  }
}

impl Drop for AdDescriptorSetLayout {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.destroy_descriptor_set_layout(self.inner, None);
    }
  }
}

pub struct AdDescriptorPool {
  vk_device: Arc<ash::Device>,
  free_sets_supported: bool,
  inner: vk::DescriptorPool,
}

impl AdDescriptorPool {
  pub fn new(vk_device: Arc<ash::Device>, vk_dpool: vk::DescriptorPool, free_dset_support: bool) -> Self {
    Self { vk_device, free_sets_supported: free_dset_support, inner: vk_dpool }
  }
}

impl AdDescriptorPool {
  pub fn allocate_descriptor_sets(
    &self,
    set_layouts: &[&AdDescriptorSetLayout]
  ) -> Result<Vec<AdDescriptorSet>, String> {
    unsafe {
      self.vk_device.allocate_descriptor_sets(
        &vk::DescriptorSetAllocateInfo::default()
          .descriptor_pool(self.inner)
          .set_layouts(&set_layouts.iter().map(|x| x.inner).collect::<Vec<_>>())
      )
        .map_err(|e| format!("at allocating vk descriptor sets: {e}"))
        .map(|dsets| {
          dsets.iter().map(|dset| {
            AdDescriptorSet {
              vk_device: self.vk_device.clone(),
              pool: self.inner,
              free_possible: self.free_sets_supported,
              inner: *dset,
            }
          }).collect()
        })
    }
  }
}

impl Drop for AdDescriptorPool {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.destroy_descriptor_pool(self.inner, None);
    }
  }
}

pub struct AdDescriptorSet {
  pub(crate) vk_device: Arc<ash::Device>,
  pub(crate) pool: vk::DescriptorPool,
  pub(crate) free_possible: bool,
  pub inner: vk::DescriptorSet,
}

impl AdDescriptorSet {
  pub fn write_and_update(
    &self,
    binding: u32,
    start_idx: u32,
    descriptor_type: vk::DescriptorType,
    image_infos: &[vk::DescriptorImageInfo],
    buffer_infos: &[vk::DescriptorBufferInfo],
  ) {
    if (descriptor_type == vk::DescriptorType::STORAGE_BUFFER ||
      descriptor_type == vk::DescriptorType::STORAGE_BUFFER_DYNAMIC ||
      descriptor_type == vk::DescriptorType::UNIFORM_BUFFER ||
      descriptor_type == vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC) &&
      buffer_infos.len() > 0 {
      let write_info = vk::WriteDescriptorSet::default()
        .dst_set(self.inner)
        .dst_binding(binding)
        .dst_array_element(start_idx)
        .descriptor_count(buffer_infos.len() as u32)
        .descriptor_type(descriptor_type)
        .buffer_info(&buffer_infos);
      unsafe {
        self.vk_device.update_descriptor_sets(&[write_info], &[]);
      }
    }
    if (descriptor_type == vk::DescriptorType::SAMPLED_IMAGE ||
      descriptor_type == vk::DescriptorType::STORAGE_IMAGE ||
      descriptor_type == vk::DescriptorType::COMBINED_IMAGE_SAMPLER) &&
      image_infos.len() > 0 {
      let write_info = vk::WriteDescriptorSet::default()
        .dst_set(self.inner)
        .dst_binding(binding)
        .dst_array_element(start_idx)
        .descriptor_count(image_infos.len() as u32)
        .descriptor_type(descriptor_type)
        .image_info(&image_infos);
      unsafe {
        self.vk_device.update_descriptor_sets(&[write_info], &[]);
      }
    }
  }
}

impl Drop for AdDescriptorSet {
  fn drop(&mut self) {
    unsafe {
      if self.free_possible {
        let _ = self.vk_device.free_descriptor_sets(self.pool, &[self.inner])
        .inspect_err(|e| eprintln!("error freeing descriptor set: {e}"));
      }
    }
  }
}

pub struct AdShaderModule {
  vk_device: Arc<ash::Device>,
  dropped: bool,
  inner: vk::ShaderModule,
}

impl AdShaderModule {
  pub fn new(vk_device: Arc<ash::Device>, vk_shader_module: vk::ShaderModule) -> Self {
    Self { vk_device, dropped: false, inner: vk_shader_module }
  }

  pub fn manual_destroy(&mut self) {
    unsafe {
      if !self.dropped {
        self.vk_device.destroy_shader_module(self.inner, None);
        self.dropped = true;
      }
    }
  }
}

impl Drop for AdShaderModule {
  fn drop(&mut self) {
    unsafe {
      if !self.dropped {
        self.vk_device.destroy_shader_module(self.inner, None);
      }
    }
  } 
}

pub struct AdPipeline {
  pub(crate) vk_device: Arc<ash::Device>,
  pub layout: vk::PipelineLayout,
  pub inner: vk::Pipeline,
}

impl Drop for AdPipeline {
  fn drop(&mut self) {
    unsafe {
      self.vk_device.destroy_pipeline(self.inner, None);
      self.vk_device.destroy_pipeline_layout(self.layout, None);
    }
  }
}
