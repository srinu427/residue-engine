use std::{borrow::Cow, ffi::CStr, sync::Arc};

use ash_context::{
  ash::{ext, vk},
  AdAshInstance,
};

pub struct AdDebugInstance {
  dbg_utils_instance: ext::debug_utils::Instance,
  _ash_instance: Arc<AdAshInstance>, // To stop deleting ash instance befor deleting this
}

impl AdDebugInstance {
  pub fn new(ash_instance: Arc<AdAshInstance>) -> Self {
    Self {
      dbg_utils_instance: ext::debug_utils::Instance::new(
        ash_instance.ash_entry(),
        ash_instance.inner(),
      ),
      _ash_instance: ash_instance,
    }
  }
}

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

fn make_debug_mgr_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
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

pub struct AdDebugMessenger {
  _dbg_utils_messenger: vk::DebugUtilsMessengerEXT,
  _dbg_instance: Arc<AdDebugInstance>,
}

impl AdDebugMessenger {
  pub fn new(dbg_instance: Arc<AdDebugInstance>) -> Result<Self, String> {
    unsafe {
      let dbg_utils_messenger = dbg_instance
        .dbg_utils_instance
        .create_debug_utils_messenger(&make_debug_mgr_create_info(), None)
        .map_err(|e| format!("at dbg messenger init: {e}"))?;
      Ok(Self { _dbg_utils_messenger: dbg_utils_messenger, _dbg_instance: dbg_instance })
    }
  }
}
