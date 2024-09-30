use std::{collections::HashMap, fs, path::{Path, PathBuf}, sync::Arc};

use ash_context::{ash::{self, vk}, getset, AdAshDevice};
use ash_data_wrappers::{AdDescriptorSetLayout, AdImageView};

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdRenderPass {
  #[getset(get = "pub")]
  ash_device: Arc<AdAshDevice>,
  #[getset(get_copy = "pub")]
  inner: vk::RenderPass,
}

impl AdRenderPass {
  pub fn new(
    ash_device: Arc<AdAshDevice>,
    flags: vk::RenderPassCreateFlags,
    attachments: &[vk::AttachmentDescription],
    subpasses: &[vk::SubpassDescription],
    dependencies: &[vk::SubpassDependency],
  ) -> Result<Self, String> {
    let vk_render_pass = unsafe {
      ash_device
        .inner()
        .create_render_pass(
          &vk::RenderPassCreateInfo::default()
            .flags(flags)
            .attachments(attachments)
            .subpasses(subpasses)
            .dependencies(dependencies),
          None
        )
        .map_err(|e| format!("at vk render pass create: {e}"))?
    };
    Ok(AdRenderPass { ash_device, inner: vk_render_pass })
  }
}

impl Drop for AdRenderPass {
  fn drop(&mut self) {
    unsafe {
      self.ash_device.inner().destroy_render_pass(self.inner, None);
    }
  }
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdShaderModule {
  ash_device: Arc<AdAshDevice>,
  dropped: bool,
  #[getset(get_copy = "pub")]
  inner: vk::ShaderModule,
}

impl AdShaderModule {
  pub fn new(ash_device: Arc<AdAshDevice>, file_path: &Path) -> Result<Self, String> {
    let mut fr = fs::File::open(file_path)
      .map_err(|e| format!("error opening file {:?}: {e}", file_path))?;
    let shader_code = ash::util::read_spv(&mut fr)
      .map_err(|e| format!("error reading ords from spv file: {e}"))?;
    let create_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
    unsafe {
      ash_device
        .inner()
        .create_shader_module(&create_info, None)
        .map_err(|e| format!("error creating vk shader module: {e}"))
        .map(|vk_shader| AdShaderModule { ash_device, dropped: false, inner: vk_shader })
    }
  }

  pub fn manual_destroy(&mut self) {
    unsafe {
      if !self.dropped {
        self.ash_device.inner().destroy_shader_module(self.inner, None);
        self.dropped = true;
      }
    }
  }
}

impl Drop for AdShaderModule {
  fn drop(&mut self) {
    unsafe {
      if !self.dropped {
        self.ash_device.inner().destroy_shader_module(self.inner, None);
      }
    }
  } 
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdPipeline {
  render_pass: Arc<AdRenderPass>,
  #[getset(get_copy = "pub")]
  layout: vk::PipelineLayout,
  #[getset(get_copy = "pub")]
  inner: vk::Pipeline,
}

impl AdPipeline {
  pub fn new(
    render_pass: Arc<AdRenderPass>,
    subpass_id: u32,
    shaders: HashMap<vk::ShaderStageFlags, PathBuf>,
    set_layouts: &[&AdDescriptorSetLayout],
    rasterizer_config: vk::PipelineRasterizationStateCreateInfo,
    blend_info: &vk::PipelineColorBlendStateCreateInfo,
  ) -> Result<Self, String> {
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
    let mut shader_modules = shaders
      .iter()
      .map(|(_, path)| { AdShaderModule::new(render_pass.ash_device().clone(), path) })
      .collect::<Result<Vec<_>, String>>()?;
    let shader_stages = shaders
      .iter()
      .enumerate()
      .map(|(i, (stage, _))| {
        vk::PipelineShaderStageCreateInfo::default()
          .stage(*stage)
          .name(c"main")
          .module(shader_modules[i].inner())
      })
      .collect::<Vec<_>>();
    let pipeline_layout = unsafe {
      render_pass.ash_device().inner().create_pipeline_layout(
        &vk::PipelineLayoutCreateInfo::default()
          .set_layouts(&set_layouts.iter().map(|x| x.inner()).collect::<Vec<_>>()),
        None
      )
      .map_err(|e| format!("at creating vk pipeline layout: {e}"))?
    };

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
      .render_pass(render_pass.inner())
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
      render_pass
        .ash_device()
        .inner()
        .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
        .map_err(|(_, e)| format!("at creating vk pipeline: {e}"))?
        .swap_remove(0)
    };
    for mut shader_mod in shader_modules.drain(..) {
      shader_mod.manual_destroy();
    }
    Ok(AdPipeline {
      render_pass,
      layout: pipeline_layout,
      inner: pipeline,
    })
  }
}

impl Drop for AdPipeline {
  fn drop(&mut self) {
    unsafe {
      self.render_pass.ash_device().inner().destroy_pipeline(self.inner, None);
      self.render_pass.ash_device().inner().destroy_pipeline_layout(self.layout, None);
    }
  }
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdFrameBuffer {
  #[getset(get = "pub")]
  render_pass: Arc<AdRenderPass>,
  #[getset(get = "pub")]
  attachments: Vec<Arc<AdImageView>>,
  #[getset(get_copy = "pub")]
  resolution: vk::Extent2D,
  #[getset(get_copy = "pub")]
  layers: u32,
  #[getset(get_copy = "pub")]
  inner: vk::Framebuffer,
}

impl AdFrameBuffer {
  pub fn new(
    render_pass: Arc<AdRenderPass>,
    attachments: Vec<Arc<AdImageView>>,
    resolution: vk::Extent2D,
    layers: u32,
  ) -> Result<Arc<Self>, String> {
    let vk_framebuffer = unsafe {
      render_pass
      .ash_device()
      .inner().create_framebuffer(
        &vk::FramebufferCreateInfo::default()
          .render_pass(render_pass.inner())
          .attachments(
            &attachments
              .iter()
              .map(|x| x.inner())
              .collect::<Vec<_>>()
          )
          .width(resolution.width)
          .height(resolution.height)
          .layers(layers),
        None
      )
      .map_err(|e| format!("at creating vk frame buffer: {e}"))?
    };
    Ok(Arc::new(Self { render_pass, attachments, resolution, layers, inner: vk_framebuffer }))
  }
}

impl Drop for AdFrameBuffer {
  fn drop(&mut self) {
    unsafe {
      self.render_pass.ash_device().inner().destroy_framebuffer(self.inner, None);
    }
  }
}
