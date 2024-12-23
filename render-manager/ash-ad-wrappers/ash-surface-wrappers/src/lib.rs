use std::{collections::HashSet, sync::Arc};

use ash_context::{
  ash::{khr, vk},
  getset, AdAshDevice, AdAshInstance,
};
use ash_queue_wrappers::{AdCommandBuffer, AdQueue};
use ash_sync_wrappers::{AdFence, AdSemaphore};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdSurfaceInstance {
  #[getset(get = "pub")]
  inner: khr::surface::Instance,
  #[getset(get = "pub")]
  ash_instance: Arc<AdAshInstance>, // To stop deleting ash instance befor deleting this
}

impl AdSurfaceInstance {
  pub fn new(ash_instance: Arc<AdAshInstance>) -> Self {
    Self {
      inner: khr::surface::Instance::new(ash_instance.ash_entry(), ash_instance.inner()),
      ash_instance,
    }
  }
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdSurface {
  #[getset(get_copy = "pub")]
  inner: vk::SurfaceKHR,
  #[getset(get = "pub")]
  surface_instance: Arc<AdSurfaceInstance>,
}

impl AdSurface {
  pub fn new(
    surface_instance: Arc<AdSurfaceInstance>,
    window: &(impl HasWindowHandle + HasDisplayHandle),
  ) -> Result<Self, String> {
    unsafe {
      let vk_surface = ash_window::create_surface(
        surface_instance.ash_instance.ash_entry(),
        surface_instance.ash_instance.inner(),
        window.display_handle().map_err(|_| "unsupported window".to_string())?.as_raw(),
        window.window_handle().map_err(|_| "unsupported window".to_string())?.as_raw(),
        None,
      )
      .map_err(|e| format!("at surface create: {e}"))?;
      Ok(Self { surface_instance, inner: vk_surface })
    }
  }

  pub fn get_gpu_formats(
    &self,
    gpu: vk::PhysicalDevice,
  ) -> Result<Vec<vk::SurfaceFormatKHR>, String> {
    unsafe {
      self
        .surface_instance
        .inner
        .get_physical_device_surface_formats(gpu, self.inner)
        .map_err(|e| format!("can't get surface formats: {e}"))
    }
  }

  pub fn get_gpu_capabilities(
    &self,
    gpu: vk::PhysicalDevice,
  ) -> Result<vk::SurfaceCapabilitiesKHR, String> {
    unsafe {
      self
        .surface_instance
        .inner
        .get_physical_device_surface_capabilities(gpu, self.inner)
        .map_err(|e| format!("can't get surface capabilities: {e}"))
    }
  }

  pub fn get_gpu_present_modes(
    &self,
    gpu: vk::PhysicalDevice,
  ) -> Result<Vec<vk::PresentModeKHR>, String> {
    unsafe {
      self
        .surface_instance
        .inner
        .get_physical_device_surface_present_modes(gpu, self.inner)
        .map_err(|e| format!("can't get surface present modes: {e}"))
    }
  }

  pub fn get_supported_queue_families(&self, gpu: vk::PhysicalDevice) -> HashSet<u32> {
    unsafe {
      self
        .surface_instance
        .ash_instance
        .inner()
        .get_physical_device_queue_family_properties(gpu)
        .iter()
        .enumerate()
        .filter_map(|(qf_idx, _qf_props)| {
          match self.surface_instance.inner.get_physical_device_surface_support(
            gpu,
            qf_idx as u32,
            self.inner,
          ) {
            Ok(supported) => {
              if supported {
                Some(qf_idx as u32)
              } else {
                None
              }
            }
            Err(_) => None,
          }
        })
        .collect()
    }
  }
}

impl Drop for AdSurface {
  fn drop(&mut self) {
    unsafe {
      self.surface_instance.inner.destroy_surface(self.inner, None);
    }
  }
}

pub struct AdSwapchainDevice {
  inner: khr::swapchain::Device,
  ash_device: Arc<AdAshDevice>,
}

impl AdSwapchainDevice {
  pub fn new(ash_device: Arc<AdAshDevice>) -> Self {
    let swapchain_device =
      khr::swapchain::Device::new(ash_device.ash_instance().inner(), ash_device.inner());
    Self { inner: swapchain_device, ash_device }
  }
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdSwapchain {
  swapchain_device: Arc<AdSwapchainDevice>,
  surface: Arc<AdSurface>,
  present_queue: Arc<AdQueue>,
  #[getset(get_copy = "pub")]
  inner: vk::SwapchainKHR,
  images: Vec<vk::Image>,
  image_count: u32,
  color_space: vk::ColorSpaceKHR,
  #[getset(get_copy = "pub")]
  format: vk::Format,
  #[getset(get_copy = "pub")]
  resolution: vk::Extent2D,
  usage: vk::ImageUsageFlags,
  pre_transform: vk::SurfaceTransformFlagsKHR,
  present_mode: vk::PresentModeKHR,
  #[getset(get_copy = "pub")]
  initialized: bool,
}

impl AdSwapchain {
  pub fn new(
    swapchain_device: Arc<AdSwapchainDevice>,
    surface: Arc<AdSurface>,
    present_queue: Arc<AdQueue>,
    image_count: u32,
    color_space: vk::ColorSpaceKHR,
    format: vk::Format,
    resolution: vk::Extent2D,
    usage: vk::ImageUsageFlags,
    pre_transform: vk::SurfaceTransformFlagsKHR,
    present_mode: vk::PresentModeKHR,
    old_swapchain: Option<AdSwapchain>,
  ) -> Result<Self, String> {
    let swapchain_info = vk::SwapchainCreateInfoKHR::default()
      .surface(surface.inner)
      .old_swapchain(old_swapchain.as_ref().map(|x| x.inner).unwrap_or(vk::SwapchainKHR::null()))
      .min_image_count(image_count)
      .image_color_space(color_space)
      .image_format(format)
      .image_extent(resolution)
      .image_usage(usage)
      .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
      .pre_transform(pre_transform)
      .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
      .present_mode(present_mode)
      .clipped(true)
      .image_array_layers(1);

    unsafe {
      let swapchain = swapchain_device
        .inner
        .create_swapchain(&swapchain_info, None)
        .map_err(|e| format!("at vk swapchain create: {e}"))?;
      let images = swapchain_device
        .inner
        .get_swapchain_images(swapchain)
        .map_err(|e| format!("at getting swapchain images: {e}"))?;
      Ok(Self {
        swapchain_device: swapchain_device.clone(),
        surface,
        present_queue,
        inner: swapchain,
        images,
        image_count,
        color_space,
        format,
        resolution,
        usage,
        pre_transform,
        present_mode,
        initialized: false,
      })
    }
  }

  pub fn get_image(&self, idx: usize) -> vk::Image {
    self.images[idx % self.images.len()]
  }

  pub fn full_range_offset_3d(&self) -> [vk::Offset3D; 2] {
    [
      vk::Offset3D::default(),
      vk::Offset3D::default().x(self.resolution.width as i32).y(self.resolution.height as i32).z(1),
    ]
  }

  pub fn set_initialized(&mut self) {
    self.initialized = true;
  }

  pub fn refresh_resolution(&mut self) -> Result<(), String> {
    let surface_caps = self.surface.get_gpu_capabilities(self.swapchain_device.ash_device.gpu())?;

    let swapchain_info = vk::SwapchainCreateInfoKHR::default()
      .surface(self.surface.inner)
      .old_swapchain(self.inner)
      .min_image_count(self.image_count)
      .image_color_space(self.color_space)
      .image_format(self.format)
      .image_extent(surface_caps.current_extent)
      .image_usage(self.usage)
      .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
      .pre_transform(self.pre_transform)
      .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
      .present_mode(self.present_mode)
      .clipped(true)
      .image_array_layers(1);
    unsafe {
      let new_swapchain = self
        .swapchain_device
        .inner
        .create_swapchain(&swapchain_info, None)
        .map_err(|e| format!("at refreshing vk swapchain: {e}"))?;
      let new_images = self
        .swapchain_device
        .inner
        .get_swapchain_images(new_swapchain)
        .map_err(|e| format!("at getting new swapchain images: {e}"))?;
      self.swapchain_device.inner.destroy_swapchain(self.inner, None);
      self.inner = new_swapchain;
      self.images = new_images;
      self.resolution = surface_caps.current_extent;
    }
    self.initialized = false;
    Ok(())
  }

  pub fn acquire_next_image(
    &mut self,
    semaphore: Option<&AdSemaphore>,
    fence: Option<&AdFence>,
  ) -> Result<(u32, bool), String> {
    unsafe {
      match self.swapchain_device.inner.acquire_next_image(
        self.inner,
        999999999,
        semaphore.map(|x| x.inner()).unwrap_or(vk::Semaphore::null()),
        fence.map(|x| x.inner()).unwrap_or(vk::Fence::null()),
      ) {
        Ok((idx, refresh_needed)) => Ok((idx, refresh_needed)),
        Err(e) => {
          if e == vk::Result::ERROR_OUT_OF_DATE_KHR {
            return Ok((0, true));
          }
          Err(format!("at vk acquire image: {e}"))
        }
      }
    }
  }

  pub fn present_image(
    &self,
    image_idx: u32,
    wait_semaphores: Vec<&AdSemaphore>,
  ) -> Result<(), String> {
    unsafe {
      self
        .swapchain_device
        .inner
        .queue_present(
          self.present_queue.inner(),
          &vk::PresentInfoKHR::default()
            .swapchains(&[self.inner])
            .wait_semaphores(&wait_semaphores.iter().map(|x| x.inner()).collect::<Vec<_>>())
            .image_indices(&[image_idx]),
        )
        .map_err(|e| format!("at vk present: {e}"))?;
    }
    Ok(())
  }

  pub fn initialize(&mut self, cmd_buffer: &AdCommandBuffer) -> Result<(), String> {
    if !self.initialized {
      cmd_buffer.begin(vk::CommandBufferUsageFlags::default())?;
      cmd_buffer.pipeline_barrier(
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::BY_REGION,
        &[],
        &[],
        &self
          .images
          .iter()
          .map(|x| {
            vk::ImageMemoryBarrier::default()
              .image(*x)
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
              .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
              .old_layout(vk::ImageLayout::UNDEFINED)
              .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
          })
          .collect::<Vec<_>>(),
      );
      cmd_buffer.end()?;
    }
    Ok(())
  }
}

impl Drop for AdSwapchain {
  fn drop(&mut self) {
    unsafe {
      self.swapchain_device.inner.destroy_swapchain(self.inner, None);
    }
  }
}
