use std::sync::{Arc, Mutex};

use ash_ad_wrappers::{
  ash_context::{
    ash::vk,
    getset,
    gpu_allocator::{vulkan::Allocator, MemoryLocation},
  },
  ash_data_wrappers::{
    AdDescriptorBinding, AdDescriptorPool, AdDescriptorSet, AdDescriptorSetLayout, AdImage,
    AdImageView, AdSampler,
  },
  ash_queue_wrappers::{AdCommandBuffer, AdCommandPool, AdQueue}, ash_sync_wrappers::AdFence,
};

#[derive(getset::Getters, getset::CopyGetters)]
pub struct DepthTextureGPU {
  #[getset(get = "pub")]
  dset: Arc<AdDescriptorSet>,
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct DepthTextureGenerator {
  #[getset(get = "pub")]
  tex_dset_layout: Arc<AdDescriptorSetLayout>,
  tex_dset_pool: Arc<AdDescriptorPool>,
  sampler: Arc<AdSampler>,
  allocator: Arc<Mutex<Allocator>>,
  cmd_pool: Arc<AdCommandPool>,
}

impl DepthTextureGenerator {
  pub fn new(allocator: Arc<Mutex<Allocator>>, queue: Arc<AdQueue>) -> Result<Self, String> {
    let ash_device = queue.ash_device().clone();
    let dset_pool = AdDescriptorPool::new(
      ash_device.clone(),
      vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
      10,
      &[vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 10,
      }],
    )?;
    let dset_layout = AdDescriptorSetLayout::new(
      ash_device.clone(),
      &[(vk::ShaderStageFlags::FRAGMENT, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)],
    )?;
    let cmd_pool = AdCommandPool::new(queue, vk::CommandPoolCreateFlags::TRANSIENT)?;
    let sampler = AdSampler::new(ash_device.clone())?;
    Ok(Self {
      allocator,
      cmd_pool: Arc::new(cmd_pool),
      sampler: Arc::new(sampler),
      tex_dset_pool: Arc::new(dset_pool),
      tex_dset_layout: Arc::new(dset_layout),
    })
  }

  pub fn generate_with_res(&self, name: &str, res: vk::Extent2D) -> Result<DepthTextureGPU, String> {
    let ash_device = self.cmd_pool.queue().ash_device().clone();
    let tex_image = AdImage::new_2d(
      ash_device.clone(),
      self.allocator.clone(),
      MemoryLocation::GpuOnly,
      name,
      vk::Format::R32_SFLOAT,
      res,
      vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
      vk::SampleCountFlags::TYPE_1,
      1,
    )?;

    let cmd_buffer = AdCommandBuffer::new(self.cmd_pool.clone(), vk::CommandBufferLevel::PRIMARY, 1)
      .map_err(|e| format!("at creating cmd buffer: {e}"))?
      .remove(0);

    cmd_buffer.begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;

    cmd_buffer.pipeline_barrier(
      vk::PipelineStageFlags::ALL_COMMANDS,
      vk::PipelineStageFlags::ALL_COMMANDS,
      vk::DependencyFlags::BY_REGION,
      &[],
      &[],
      &[
        vk::ImageMemoryBarrier::default()
          .image(tex_image.inner())
          .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).base_array_layer(0).layer_count(1).base_mip_level(0).level_count(1))
          .src_queue_family_index(cmd_buffer.cmd_pool().queue().family_index())
          .dst_queue_family_index(cmd_buffer.cmd_pool().queue().family_index())
          .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
          .old_layout(vk::ImageLayout::UNDEFINED)
          .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
          .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
      ]
    );

    cmd_buffer.end()?;
    let fence = AdFence::new(ash_device.clone(), vk::FenceCreateFlags::empty())?;
    cmd_buffer.submit(&[], &[], Some(&fence))?;
    fence.wait(999999999)?;


    let tex_image_view = AdImageView::create_view(
      tex_image,
      vk::ImageViewType::TYPE_2D,
      vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
      },
    )?;

    let tex_dset = AdDescriptorSet::new(
      self.tex_dset_pool.clone(),
      &[(
        self.tex_dset_layout.clone(),
        vec![AdDescriptorBinding::Sampler2D((
          tex_image_view,
          vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
          self.sampler.clone(),
        ))],
      )],
    )?
    .remove(0);

    Ok(DepthTextureGPU { dset: Arc::new(tex_dset) })
  }
}
