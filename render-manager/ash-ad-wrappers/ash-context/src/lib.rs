use std::{collections::{HashMap, HashSet}, ffi::c_char, sync::Arc};

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

  pub fn list_gpus(&self) -> Result<Vec<vk::PhysicalDevice>, String> {
    unsafe {
      self.inner.enumerate_physical_devices().map_err(|e| format!("at getting gpus: {e}"))
    }
  }

  pub fn list_dedicated_gpus(&self) -> Result<Vec<vk::PhysicalDevice>, String> {
    unsafe {
      self
        .list_gpus()
        .map(|gpu_list| {
          gpu_list
            .iter()
            .filter(|x|
              self.inner.get_physical_device_properties(**x).device_type ==
               vk::PhysicalDeviceType::DISCRETE_GPU)
            .cloned()
            .collect::<Vec<_>>()
        })
    }
  }

  pub fn get_queue_family_props(&self, gpu: vk::PhysicalDevice) -> Vec<vk::QueueFamilyProperties> {
    unsafe {
      self.inner.get_physical_device_queue_family_properties(gpu)
    }
  }

  fn select_g_queue(qf_props: &[vk::QueueFamilyProperties]) -> Result<u32, String> {
    qf_props
      .iter()
      .enumerate()
      .filter(|(_, x)| x.queue_flags.contains(vk::QueueFlags::GRAPHICS))
      .max_by(|x, y| x.1.queue_count.cmp(&y.1.queue_count))
      .map(|(i, _)| i as u32)
      .ok_or("graphics queue not supported".to_string())
  }

  fn select_c_queue(qf_props: &[vk::QueueFamilyProperties]) -> Result<u32, String> {
    qf_props
      .iter()
      .enumerate()
      .filter(|(_, x)| x.queue_flags.contains(vk::QueueFlags::COMPUTE))
      .map(|(i, x)| {
        let mut weight = 0;
        if !x.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
          weight = 1;
        }
        (i, *x, weight)
      })
      .max_by(|x, y|
        if x.1.queue_count == y.1.queue_count {
          x.2.cmp(&y.2)
        } else {
          x.1.queue_count.cmp(&y.1.queue_count)
        }
      )
      .map(|(i, _, _)| i as u32)
      .ok_or("compute queue not supported".to_string())
  }

  fn select_t_queue(qf_props: &[vk::QueueFamilyProperties]) -> Result<u32, String> {
    qf_props
      .iter()
      .enumerate()
      .filter(|(_, x)| x.queue_flags.contains(vk::QueueFlags::TRANSFER))
      .map(|(i, x)| {
        let g_support = x.queue_flags.contains(vk::QueueFlags::GRAPHICS);
        let c_support = x.queue_flags.contains(vk::QueueFlags::COMPUTE);
        let mut weight = 0;
        if !g_support {
          weight += 2
        }
        if !c_support {
          weight += 1;
        }
        (i, *x, weight)
      })
      .max_by(|x, y|
        if x.1.queue_count == y.1.queue_count {
          x.2.cmp(&y.2)
        } else {
          x.1.queue_count.cmp(&y.1.queue_count)
        }
      )
      .map(|(i, _, _)| i as u32)
      .ok_or("transfer queue not supported".to_string())
  }

  pub fn select_gpu_queue_families(
    &self,
    gpu: vk::PhysicalDevice
  ) -> Result<HashMap<GPUQueueType, u32>, String> {
    let qf_props = self.get_queue_family_props(gpu);
    Ok(HashMap::from([
      (GPUQueueType::Graphics, Self::select_g_queue(&qf_props)?),
      (GPUQueueType::Compute, Self::select_c_queue(&qf_props)?),
      (GPUQueueType::Transfer, Self::select_t_queue(&qf_props)?),
    ]))
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
