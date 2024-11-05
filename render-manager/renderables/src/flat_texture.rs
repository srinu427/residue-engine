use std::sync::{Arc, Mutex};

use ash_ad_wrappers::{ash_context::{ash::vk, getset, gpu_allocator::{vulkan::Allocator, MemoryLocation}}, ash_data_wrappers::{AdDescriptorBinding, AdDescriptorPool, AdDescriptorSet, AdDescriptorSetLayout, AdImage, AdImageView, AdSampler}, ash_queue_wrappers::{AdCommandBuffer, AdCommandPool, AdQueue}};

#[derive(getset::Getters, getset::CopyGetters)]
pub struct FlatTextureGPU {
  #[getset(get = "pub")]
  dset: Arc<AdDescriptorSet>,
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct FlatTextureGenerator {
  #[getset(get = "pub")]
  tex_dset_layout: Arc<AdDescriptorSetLayout>,
  tex_dset_pool: Arc<AdDescriptorPool>,
  sampler: Arc<AdSampler>,
  allocator: Arc<Mutex<Allocator>>,
  cmd_pool: Arc<AdCommandPool>,
}

impl FlatTextureGenerator {
  pub fn new(allocator: Arc<Mutex<Allocator>>, queue: Arc<AdQueue>) -> Result<Self, String> {
    let ash_device = queue.ash_device().clone();
    let dset_pool = AdDescriptorPool::new(
      ash_device.clone(),
      vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
      1000,
      &[vk::DescriptorPoolSize{ ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 1000 },]
    )?;
    let dset_layout = AdDescriptorSetLayout::new(
      ash_device.clone(),
      &[(vk::ShaderStageFlags::FRAGMENT, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)]
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

  pub fn upload_flat_texture(&self, name: &str, path: &str) -> Result<FlatTextureGPU, String> {
    let ash_device = self.cmd_pool.queue().ash_device().clone();
    let cmd_buffer = AdCommandBuffer::new(self.cmd_pool.clone(), vk::CommandBufferLevel::PRIMARY, 1)?.remove(0);
    let tex_image = AdImage::new_2d_from_file(
      ash_device.clone(),
      self.allocator.clone(),
      MemoryLocation::GpuOnly,
      name,
      vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
      path,
      &cmd_buffer,
      vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    )?;
    let tex_image_view = AdImageView::create_view(
      tex_image,
      vk::ImageViewType::TYPE_2D,
      vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1
      }
    )?;

    let tex_dset = AdDescriptorSet::new(
      self.tex_dset_pool.clone(),
      &[(
          self.tex_dset_layout.clone(),
          vec![AdDescriptorBinding::Sampler2D((
            tex_image_view,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            self.sampler.clone()))]
        )]
    )?
    .remove(0);

    Ok(FlatTextureGPU { dset: Arc::new(tex_dset) })
  }
}
