use std::{
  collections::HashMap, process::exit, sync::{mpsc, Arc, Mutex, RwLock}, time::Instant
};

use crossbeam_channel as cc;

use ash_ad_wrappers::{
  ash_context::{
    ash::{khr, vk},
    gpu_allocator::vulkan::Allocator,
    AdAshDevice, GPUQueueType,
  },
  ash_queue_wrappers::{AdCommandBuffer, AdCommandPool, AdQueue},
  ash_render_wrappers::AdFrameBuffer,
  ash_surface_wrappers::{AdSwapchain, AdSwapchainDevice},
  ash_sync_wrappers::{AdFence, AdSemaphore},
};
use triangle_mesh_renderer::{Camera3D, TriMeshRenderer, TriRenderable};
pub use triangle_mesh_renderer::{glam, TriMeshCPU};

pub use ash_ad_wrappers::ash_context::AdAshInstance;
pub use ash_ad_wrappers::ash_surface_wrappers::{AdSurface, AdSurfaceInstance};

pub enum RendererMessage {
  AddTriMesh(String, Arc<TriMeshCPU>, String),
  Draw,
  Stop,
}

pub struct RendererMessageBatch {
  ordered: bool,
  batch: Vec<RendererMessage>,
}

pub struct Renderer {
  thread: std::thread::JoinHandle<Result<(), String>>,
  ordered_cmds: Arc<Mutex<Vec<RendererMessage>>>,
}

impl Renderer {
  pub fn new(surface: Arc<AdSurface>) -> Result<Self, String> {
    let ordered_cmds = Arc::new(Mutex::new(vec![]));
    let renderer_ordered_cmds = ordered_cmds.clone();

    let thread = std::thread::spawn(move || {
      let mut render_mgr = RenderManager::new(surface)?;
      loop {
        let mut quit_renderer = false;
        let mut current_cmds = renderer_ordered_cmds.lock().map_err(|e| format!("memory poisoning: {e}"))?;
        for message in current_cmds.iter() {
          match message {
            RendererMessage::AddTriMesh(name, tri_mesh_cpu, tex_file) => {
              let _ = render_mgr.add_tri_mesh(name.to_string(), &tri_mesh_cpu, tex_file.to_string())
                .inspect_err(|e| eprintln!("error adding mesh: {e}"));
            },
            RendererMessage::Draw => {
              for _ in 0..3 {
                if let Ok(d_res) = render_mgr.draw().inspect_err(|e| eprintln!("{}", e)) {
                  if !d_res {
                    break;
                  }
                }
              };
            },
            RendererMessage::Stop => {
              quit_renderer = true;
            },
          }
        }
        current_cmds.clear();
        if quit_renderer {
          break;
        }
      }
      return Ok::<(), String>(())
    });
    Ok(Self { thread, ordered_cmds })
  }

  pub fn send_batch_sync(&mut self, mut batch: Vec<RendererMessage>) -> Result<bool, String> {
    loop {
      let mut current_cmds = self.ordered_cmds.lock().map_err(|e| format!("memory poisoning: {e}"))?;
      if current_cmds.len() == 0 {
        current_cmds.append(&mut batch);
        break;
      }
    }
    Ok(true)
  }
}

impl Drop for Renderer {
  fn drop(&mut self) {
    if !self.thread.is_finished() {
      let _ = self
        .send_batch_sync(vec![RendererMessage::Stop])
        .inspect_err(|e| eprintln!("at stopping renderer: {e}"));
    }
  }
}

pub enum RenderObject {
  TriMesh(Arc<TriRenderable>),
}

pub struct RenderManager {
  render_objects: Vec<Arc<RenderObject>>,
  camera: Camera3D,
  triangle_frame_buffers: Vec<Arc<AdFrameBuffer>>,
  triangle_mesh_renderer: TriMeshRenderer,
  gen_allocator: Arc<Mutex<Allocator>>,
  render_semaphores: Vec<AdSemaphore>,
  render_fences: Vec<AdFence>,
  render_cmd_buffers: Vec<AdCommandBuffer>,
  image_acquire_fence: AdFence,
  swapchain: AdSwapchain,
  queues: HashMap<GPUQueueType, Arc<AdQueue>>,
  ash_device: Arc<AdAshDevice>,
}

impl RenderManager {
  pub fn new(surface: Arc<AdSurface>) -> Result<Self, String> {
    let ash_instance = surface.surface_instance().ash_instance().clone();
    let gpu = ash_instance.list_dedicated_gpus()?.iter().next().cloned().unwrap_or(
      ash_instance.list_gpus()?.iter().next().cloned().ok_or("no supported gpus".to_string())?,
    );
    let mut q_f_idxs = ash_instance.select_gpu_queue_families(gpu)?;
    q_f_idxs.insert(
      GPUQueueType::Present,
      *surface
        .get_supported_queue_families(gpu)
        .iter()
        .next()
        .ok_or("no supported present queues".to_string())?,
    );
    let qf_info = ash_instance.get_queue_family_props(gpu);
    let mut queue_counts = HashMap::new();
    for (_, qf_idx) in q_f_idxs.iter() {
      let val_ptr = queue_counts.entry(*qf_idx).or_insert(0);
      if qf_info[*qf_idx as usize].queue_count > *val_ptr {
        *val_ptr += 1
      };
    }

    let device_extensions = vec![
      khr::swapchain::NAME.as_ptr(),
      #[cfg(target_os = "macos")]
      khr::portability_subset::NAME.as_ptr(),
    ];

    let ash_device = Arc::new(AdAshDevice::new(
      ash_instance,
      gpu,
      device_extensions,
      vk::PhysicalDeviceFeatures::default(),
      queue_counts.clone(),
    )?);

    let mut queues = HashMap::new();
    for (q_type, q_f_idx) in q_f_idxs {
      let queue_idx = queue_counts.entry(q_f_idx).or_default();
      if *queue_idx > 0 {
        *queue_idx -= 1
      };
      queues.insert(q_type, Arc::new(AdQueue::new(ash_device.clone(), q_f_idx, *queue_idx)));
    }

    let surface_formats = surface.get_gpu_formats(ash_device.gpu())?;
    let surface_caps = surface.get_gpu_capabilities(ash_device.gpu())?;
    let surface_present_modes = surface.get_gpu_present_modes(ash_device.gpu())?;

    let surface_format = surface_formats
      .iter()
      .find(|f| f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
      .cloned()
      .unwrap_or(surface_formats[0]);
    let present_mode = surface_present_modes
      .iter()
      .find(|m| **m == vk::PresentModeKHR::IMMEDIATE)
      .cloned()
      .unwrap_or(vk::PresentModeKHR::FIFO);

    let swapchain_resolution = match surface_caps.current_extent.width {
      u32::MAX => vk::Extent2D::default().width(640).height(480),
      _ => surface_caps.current_extent,
    };

    let swapchain_image_count = std::cmp::min(
      surface_caps.min_image_count + 1,
      std::cmp::max(surface_caps.max_image_count, std::cmp::min(surface_caps.min_image_count, 3)),
    );

    let swapchain = AdSwapchain::new(
      Arc::new(AdSwapchainDevice::new(ash_device.clone())),
      surface,
      queues.get(&GPUQueueType::Present).ok_or("no supported present queue")?.clone(),
      swapchain_image_count,
      surface_format.color_space,
      surface_format.format,
      swapchain_resolution,
      vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
      surface_caps.current_transform,
      present_mode,
      None,
    )?;

    let image_acquire_fence = AdFence::new(ash_device.clone(), vk::FenceCreateFlags::default())?;

    let render_cmd_pool = Arc::new(AdCommandPool::new(
      queues[&GPUQueueType::Graphics].clone(),
      vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
    )?);

    let render_cmd_buffers =
      AdCommandBuffer::new(render_cmd_pool.clone(), vk::CommandBufferLevel::PRIMARY, 3)?;

    let render_semaphores = (0..3)
      .map(|_| AdSemaphore::new(ash_device.clone(), vk::SemaphoreCreateFlags::default()))
      .collect::<Result<Vec<_>, _>>()?;

    let render_fences = (0..3)
      .map(|_| AdFence::new(ash_device.clone(), vk::FenceCreateFlags::SIGNALED))
      .collect::<Result<Vec<_>, _>>()?;

    let gen_allocator = Arc::new(Mutex::new(ash_device.create_allocator()?));

    let triangle_mesh_renderer = TriMeshRenderer::new(
      ash_device.clone(),
      queues[&GPUQueueType::Transfer].clone(),
    )?;

    let mut triangle_frame_buffers = triangle_mesh_renderer.create_framebuffers(
      &render_cmd_buffers[0],
      gen_allocator.clone(),
      swapchain_resolution,
      3,
    )?;
    for (i, fb) in triangle_frame_buffers.iter_mut().enumerate() {
      fb.attachments()[0]
        .image()
        .allocation()
        .lock()
        .map_err(|e| format!("at getting image mem lock: {e}"))?
        .rename(&format!("triangle_out_image_{i}"))?;
    }

    let camera = Camera3D {
      pos: glam::vec4(2.0, 2.0, 2.0, 0.0),
      look_dir: glam::vec4(-1.0, -1.0, -1.0, 0.0),
      view_proj_mat: glam::Mat4::IDENTITY,
    };

    Ok(Self {
      ash_device,
      queues,
      swapchain,
      image_acquire_fence,
      render_cmd_buffers,
      render_semaphores,
      render_fences,
      gen_allocator,
      triangle_mesh_renderer,
      triangle_frame_buffers,
      camera,
      render_objects: vec![],
    })
  }

  pub fn add_tri_mesh(&mut self, name: String, mesh: &TriMeshCPU, tex_path: String) -> Result<(), String> {
    self.triangle_mesh_renderer.add_renderable(&name, mesh, (&tex_path, &tex_path))?;
    Ok(())
  }

  pub fn draw(&mut self) -> Result<bool, String> {
    // Acquiring next image to draw
    let (image_idx, refresh_needed) = self
      .swapchain
      .acquire_next_image(None, Some(&self.image_acquire_fence))
      .map_err(|e| format!("at acquiring next image: {e}"))?;
    self.image_acquire_fence.wait(999999999)?;
    self.image_acquire_fence.reset()?;

    if refresh_needed {
      let _ = self
        .swapchain
        .refresh_resolution()
        .inspect_err(|e| eprintln!("at refreshing swapchain res: {e}"));
      return Ok(true);
    }

    self.render_fences[image_idx as usize].wait(999999999)?;
    self.render_fences[image_idx as usize].reset()?;

    if !self.swapchain.initialized() {
      self
        .swapchain
        .initialize(&self.render_cmd_buffers[image_idx as usize])
        .map_err(|e| format!("at adding init cmds:  {e}"))?;

      self.render_cmd_buffers[image_idx as usize]
        .submit(&[], &[], Some(&self.image_acquire_fence))
        .map_err(|e| format!("error submitting cmds: {e}"))?;

      self.image_acquire_fence.wait(999999999)?;
      self.image_acquire_fence.reset()?;
      self.swapchain.set_initialized();
    }

    let current_sc_res = self.swapchain.resolution();
    let triangle_out_image_res =
      self.triangle_frame_buffers[0].attachments()[0].image().resolution();
    if current_sc_res.height != triangle_out_image_res.height
      || current_sc_res.width != triangle_out_image_res.width
    {
      self.triangle_frame_buffers = self.triangle_mesh_renderer.create_framebuffers(
        &self.render_cmd_buffers[0],
        self.gen_allocator.clone(),
        current_sc_res,
        3,
      )?;
      for (i, fb) in self.triangle_frame_buffers.iter_mut().enumerate() {
        fb.attachments()[0]
          .image()
          .allocation()
          .lock()
          .map_err(|e| format!("at getting image mem lock: {e}"))?
          .rename(&format!("triangle_out_image_{i}"))?;
      }
    }

    // Camera update
    let current_aspect_ratio = self.triangle_frame_buffers[image_idx as usize].resolution().width as f32 / self.triangle_frame_buffers[image_idx as usize].resolution().height as f32;
    self.camera.refresh_vp_matrix(1.5, current_aspect_ratio);
    // if let AdDescriptorBinding::UniformBuffer(Some(cam_buffer)) = self.camera_dset.bindings()[0].clone() {
    //   cam_buffer.write_data(0, &[self.camera])?;
    // }

    self.render_cmd_buffers[image_idx as usize]
      .begin(vk::CommandBufferUsageFlags::default())
      .map_err(|e| format!("at beginning render cmd buffer:  {e}"))?;

    self.triangle_mesh_renderer.render_meshes(
      &self.render_cmd_buffers[image_idx as usize],
      &self.triangle_frame_buffers[image_idx as usize],
      self.camera,
    );

    self.render_cmd_buffers[image_idx as usize].pipeline_barrier(
      vk::PipelineStageFlags::TRANSFER,
      vk::PipelineStageFlags::TRANSFER,
      vk::DependencyFlags::BY_REGION,
      &[],
      &[],
      &[vk::ImageMemoryBarrier::default()
        .image(self.swapchain.get_image(image_idx as usize))
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1)
            .base_array_layer(0)
            .level_count(1)
            .base_mip_level(0),
        )
        .src_queue_family_index(self.queues[&GPUQueueType::Graphics].family_index())
        .dst_queue_family_index(self.queues[&GPUQueueType::Graphics].family_index())
        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)],
    );

    self.render_cmd_buffers[image_idx as usize].blit_image(
      self.triangle_frame_buffers[image_idx as usize].attachments()[0].image().inner(),
      vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
      self.swapchain.get_image(image_idx as usize),
      vk::ImageLayout::TRANSFER_DST_OPTIMAL,
      &[vk::ImageBlit::default()
        .src_subresource(
          vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1),
        )
        .src_offsets(
          self.triangle_frame_buffers[image_idx as usize].attachments()[0]
            .image()
            .full_range_offset_3d(),
        )
        .dst_subresource(
          vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1),
        )
        .dst_offsets(self.swapchain.full_range_offset_3d())],
      vk::Filter::NEAREST,
    );

    self.render_cmd_buffers[image_idx as usize].pipeline_barrier(
      vk::PipelineStageFlags::TRANSFER,
      vk::PipelineStageFlags::TRANSFER,
      vk::DependencyFlags::BY_REGION,
      &[],
      &[],
      &[vk::ImageMemoryBarrier::default()
        .image(self.swapchain.get_image(image_idx as usize))
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1)
            .base_array_layer(0)
            .level_count(1)
            .base_mip_level(0),
        )
        .src_queue_family_index(self.queues[&GPUQueueType::Graphics].family_index())
        .dst_queue_family_index(self.queues[&GPUQueueType::Graphics].family_index())
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)],
    );

    self.render_cmd_buffers[image_idx as usize]
      .end()
      .map_err(|e| format!("at ending render cmd buffer: {e}"))?;

    self.render_cmd_buffers[image_idx as usize]
      .submit(
        &[&self.render_semaphores[image_idx as usize]],
        &[],
        Some(&self.render_fences[image_idx as usize]),
      )
      .map_err(|e| format!("error submitting cmds: {e}"))?;

    if let Err(e) =
      self.swapchain.present_image(image_idx, vec![&self.render_semaphores[image_idx as usize]])
    {
      if e.ends_with("ERROR_OUT_OF_DATE_KHR") {
        let _ = self
          .swapchain
          .refresh_resolution()
          .inspect_err(|e| eprintln!("at refreshing swapchain res: {e}"));
        return Ok(true);
      }
    }
    Ok(false)
  }
}

impl Drop for RenderManager {
  fn drop(&mut self) {
    for fence in self.render_fences.iter() {
      let _ = fence.wait(999999999);
      let _ = fence.reset();
    }
  }
}
