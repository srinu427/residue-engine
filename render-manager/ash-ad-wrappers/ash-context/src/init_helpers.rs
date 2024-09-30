use std::collections::HashSet;
use std::ffi::c_char;

use ash::{ext, khr, vk};

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
