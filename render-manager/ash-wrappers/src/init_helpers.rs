use ash_queue_wrappers::AdQueue;
use ash_common_imports::ash::{self, ext, khr, vk};
use std::borrow::Cow;
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, CStr};
use std::sync::Arc;

unsafe extern "system" fn vulkan_debug_callback(
  message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
  message_type: vk::DebugUtilsMessageTypeFlagsEXT,
  p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
  _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
  let callback_data = *p_callback_data;
  let message_id_number = callback_data.message_id_number;

  let message_id_name = if callback_data.p_message_id_name.is_null() {
    Cow::from("")
  } else {
    CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
  };

  let message = if callback_data.p_message.is_null() {
    Cow::from("")
  } else {
    CStr::from_ptr(callback_data.p_message).to_string_lossy()
  };

  println!(
    "{message_severity:?}:\n\
        {message_type:?} [{message_id_name} ({message_id_number})] : {message}",
  );

  vk::FALSE
}

pub fn make_debug_mgr_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
  vk::DebugUtilsMessengerCreateInfoEXT::default()
    .message_severity(
      vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
    )
    .message_type(
      vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
    )
    .pfn_user_callback(Some(vulkan_debug_callback))
}

pub unsafe fn init_instance(
  entry: &ash::Entry,
  layers: Vec<*const c_char>,
  extensions: Vec<*const c_char>,
) -> Result<ash::Instance, String> {
  let mandatory_layers = HashSet::from([
    #[cfg(debug_assertions)]
    c"VK_LAYER_KHRONOS_validation".as_ptr(),
  ]);
  let mandatory_extensions = HashSet::from([
    #[cfg(debug_assertions)]
    ext::debug_utils::NAME.as_ptr(),
    khr::get_physical_device_properties2::NAME.as_ptr(),
    khr::surface::NAME.as_ptr(),
    #[cfg(target_os = "windows")]
    khr::win32_surface::NAME.as_ptr(),
    #[cfg(target_os = "linux")]
    khr::xlib_surface::NAME.as_ptr(),
    #[cfg(target_os = "linux")]
    khr::wayland_surface::NAME.as_ptr(),
    #[cfg(target_os = "macos")]
    khr::portability_enumeration::NAME.as_ptr(),
    #[cfg(target_os = "macos")]
    ext::metal_surface::NAME.as_ptr(),
    #[cfg(target_os = "android")]
    khr::android_surface::NAME.as_ptr(),
  ]);

  let all_layers = layers
    .iter()
    .cloned()
    .collect::<HashSet<_>>()
    .union(&mandatory_layers)
    .cloned()
    .collect::<Vec<_>>();
  let all_extensions = extensions
    .iter()
    .cloned()
    .collect::<HashSet<_>>()
    .union(&mandatory_extensions)
    .cloned()
    .collect::<Vec<_>>();

  let app_info = vk::ApplicationInfo::default()
    .application_name(c"Residue VK App")
    .application_version(0)
    .engine_name(c"Residue Engine")
    .engine_version(0)
    .api_version(vk::API_VERSION_1_0);

  #[cfg(target_os = "macos")]
  let vk_instance_create_info = vk::InstanceCreateInfo::default()
    .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR)
    .application_info(&app_info)
    .enabled_layer_names(&all_layers)
    .enabled_extension_names(&all_extensions);

  #[cfg(not(target_os = "macos"))]
  let vk_instance_create_info = vk::InstanceCreateInfo::default()
    .application_info(&app_info)
    .enabled_layer_names(&all_layers)
    .enabled_extension_names(&all_extensions);

  entry
    .create_instance(&vk_instance_create_info, None)
    .map_err(|e| format!("at instance create: {e}"))
}

fn select_g_queue(gpu_queue_props: &[vk::QueueFamilyProperties]) -> Option<u32> {
  let mut selected_queue = None;
  let mut selected_queue_count = 0;
  for (queue_idx, queue_props) in gpu_queue_props.iter().enumerate() {
    let g_support = queue_props.queue_flags.contains(vk::QueueFlags::GRAPHICS);
    if g_support && selected_queue_count < queue_props.queue_count {
      selected_queue = Some(queue_idx as u32);
      selected_queue_count = queue_props.queue_count;
    }
  }
  selected_queue
}

fn select_c_queue(gpu_queue_props: &[vk::QueueFamilyProperties]) -> Option<u32> {
  let mut selected_queue = None;
  let mut selected_weight = 0;
  let mut selected_queue_count = 0;
  for (queue_idx, queue_props) in gpu_queue_props.iter().enumerate() {
    let g_support = queue_props.queue_flags.contains(vk::QueueFlags::GRAPHICS);
    let c_support = queue_props.queue_flags.contains(vk::QueueFlags::COMPUTE);

    if c_support {
      let mut weight = 0;
      if !g_support {
        weight += 1
      }
      if selected_weight < weight {
        selected_queue = Some(queue_idx as u32);
        selected_weight = weight;
        selected_queue_count = queue_props.queue_count;
      }
      if selected_weight == weight && selected_queue_count < queue_props.queue_count {
        selected_queue = Some(queue_idx as u32);
        selected_weight = weight;
        selected_queue_count = queue_props.queue_count;
      }
    }
  }
  selected_queue
}

fn select_t_queue(gpu_queue_props: &Vec<vk::QueueFamilyProperties>) -> Option<u32> {
  let mut selected_queue = None;
  let mut selected_weight = 0;
  let mut selected_queue_count = 0;
  for (queue_idx, queue_props) in gpu_queue_props.iter().enumerate() {
    let g_support = queue_props.queue_flags.contains(vk::QueueFlags::GRAPHICS);
    let t_support = queue_props.queue_flags.contains(vk::QueueFlags::TRANSFER);
    let c_support = queue_props.queue_flags.contains(vk::QueueFlags::COMPUTE);

    if t_support {
      let mut weight = 0;
      if !g_support {
        weight += 2
      }
      if !c_support {
        weight += 1;
      }
      if selected_weight < weight {
        selected_queue = Some(queue_idx as u32);
        selected_weight = weight;
        selected_queue_count = queue_props.queue_count;
      }
      if selected_weight == weight && selected_queue_count < queue_props.queue_count {
        selected_queue = Some(queue_idx as u32);
        selected_weight = weight;
        selected_queue_count = queue_props.queue_count;
      }
    }
  }
  selected_queue
}

fn select_p_queue(
  gpu_queue_props: &Vec<vk::QueueFamilyProperties>,
  surface_driver: &khr::surface::Instance,
  surface: vk::SurfaceKHR,
  gpu: vk::PhysicalDevice,
) -> Option<u32> {
  let mut selected_queue = None;
  let mut selected_queue_count = 0;
  unsafe {
    for (queue_idx, queue_props) in gpu_queue_props.iter().enumerate() {
      if let Ok(p_support) =
        surface_driver.get_physical_device_surface_support(gpu, queue_idx as u32, surface)
      {
        if p_support && selected_queue_count < queue_props.queue_count {
          selected_queue = Some(queue_idx as u32);
          selected_queue_count = queue_props.queue_count;
        }
      }
    }
  }
  selected_queue
}

pub unsafe fn select_gpu_queues(
  vk_instance: &ash::Instance,
  gpu: vk::PhysicalDevice,
  surface_instance: &khr::surface::Instance,
  surface: vk::SurfaceKHR,
) -> Option<[u32; 4]> {
  let gpu_q_props = vk_instance.get_physical_device_queue_family_properties(gpu);
  let graphics_q_idx = select_g_queue(&gpu_q_props)?;
  let compute_q_idx = select_c_queue(&gpu_q_props)?;
  let transfer_q_idx = select_t_queue(&gpu_q_props)?;
  let present_q_idx = select_p_queue(&gpu_q_props, surface_instance, surface, gpu)?;
  return Some([graphics_q_idx, compute_q_idx, transfer_q_idx, present_q_idx]);
}

pub unsafe fn create_device_and_queues(
  vk_instance: &ash::Instance,
  gpu: vk::PhysicalDevice,
  extensions: Vec<*const c_char>,
  features: vk::PhysicalDeviceFeatures,
  queue_indices: [u32; 4],
) -> Result<(Arc<ash::Device>, Vec<AdQueue>), String> {
  let queue_priorities: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
  let gpu_q_props = vk_instance.get_physical_device_queue_family_properties(gpu);

  let mut q_idx_map = HashMap::<u32, u32>::with_capacity(4);
  for x in queue_indices {
    if x >= gpu_q_props.len() as u32 {
      return Err("invalid queue ids requested".to_string());
    }
    if let Some(q_count) = q_idx_map.get_mut(&x) {
      *q_count = min(*q_count + 1, gpu_q_props[x as usize].queue_count);
    } else {
      q_idx_map.insert(x, 1);
    }
  }

  let queue_create_infos = q_idx_map
    .iter()
    .map(|(k, v)| {
      vk::DeviceQueueCreateInfo::default()
        .queue_family_index(*k)
        .queue_priorities(&queue_priorities[0..(*v as usize)])
    })
    .collect::<Vec<_>>();
  let device_create_info = vk::DeviceCreateInfo::default()
    .queue_create_infos(queue_create_infos.as_slice())
    .enabled_extension_names(&extensions)
    .enabled_features(&features);

  let device = Arc::new(
    vk_instance
      .create_device(gpu, &device_create_info, None)
      .map_err(|e| format!("at logic device init: {e}"))?,
  );

  let mut queues = Vec::with_capacity(4);
  for x in queue_indices {
    let cur_q_idx = q_idx_map.get_mut(&x).ok_or("invalid queue".to_string())?;
    queues.push(AdQueue::new(device.clone(), x, device.get_device_queue(x, *cur_q_idx - 1)));
    if *cur_q_idx != 1 {
      *cur_q_idx -= 1;
    }
  }

  Ok((device, queues))
}
