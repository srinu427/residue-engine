use std::{collections::HashMap, ffi::c_char, sync::Arc};

pub use ash;
pub use gpu_allocator;
pub use getset;
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

mod init_helpers;

#[derive(getset::Getters)]
pub struct AdAshInstance {
  #[getset(get = "pub")]
  inner: ash::Instance,
  #[getset(get = "pub")]
  ash_entry: ash::Entry,
}

impl AdAshInstance {
  pub fn new() -> Result<Self, String> {
    unsafe {
      let ash_entry = ash::Entry::load().map_err(|e| format!("at VK load: {e}"))?;
      let ash_instance = init_helpers::init_instance(&ash_entry, vec![], vec![])?;
      Ok(Self { inner: ash_instance, ash_entry })
    }
  }
}

impl Drop for AdAshInstance {
  fn drop(&mut self) {
    unsafe { self.inner.destroy_instance(None); }
  }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone)]
pub enum GPUQueueType {
  Graphics,
  Compute,
  Transfer,
  Present,
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct AdAshDevice {
  // queue_family_info: HashMap<GPUQueueType, (u32, u32)>, // Queue Family idx and count
  #[getset(get = "pub")]
  inner: ash::Device,
  #[getset(get_copy = "pub")]
  gpu: vk::PhysicalDevice,
  #[getset(get = "pub")]
  ash_instance: Arc<AdAshInstance>, // To avoid destroying instance till device is destroyed
}

impl AdAshDevice {
  pub fn new(
    ash_instance: Arc<AdAshInstance>,
    gpu: vk::PhysicalDevice,
    extensions: Vec<*const c_char>,
    features: vk::PhysicalDeviceFeatures,
    queue_counts: HashMap<u32, u32>,
  ) -> Result<Self, String> {
    let queue_priorities = [1.0, 1.0, 1.0, 1.0];
    let q_create_infos = queue_counts
      .iter()
      .map(|(q_f_idx, q_count)| {
        vk::DeviceQueueCreateInfo::default()
        .queue_family_index(*q_f_idx)
        .queue_priorities(&queue_priorities[0..(*q_count as usize)])
      })
      .collect::<Vec<_>>();
    let device_create_info = vk::DeviceCreateInfo::default()
      .queue_create_infos(&q_create_infos)
      .enabled_extension_names(&extensions)
      .enabled_features(&features);
    let vk_device = unsafe {
      ash_instance
        .inner
        .create_device(gpu, &device_create_info, None)
        .map_err(|e| format!("at vk device create: {e}"))?
    };

    Ok(Self { inner: vk_device, gpu, ash_instance })
  }

  pub fn create_allocator(&self) -> Result<Allocator, String> {
    Allocator::new(&AllocatorCreateDesc {
      instance: self.ash_instance.inner.clone(),
      device: self.inner.clone(),
      physical_device: self.gpu,
      debug_settings: Default::default(),
      buffer_device_address: false,
      allocation_sizes: Default::default()
    })
      .map_err(|e| format!("at creating gpu allocator: {e}"))
  }
}

impl Drop for AdAshDevice {
  fn drop(&mut self) {
    unsafe {
      self.inner.destroy_device(None);
    }
  }
}
