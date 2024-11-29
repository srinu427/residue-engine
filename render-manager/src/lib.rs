use std::{
  collections::HashMap,
  sync::{Arc, Mutex, OnceLock},
};

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
use renderables::{
  flat_texture::FlatTextureGenerator, triangle_mesh::TriMeshGenerator
};
use renderers::triangle_mesh_renderers::TriMeshTexRenderer;

pub use ash_ad_wrappers::ash_context::AdAshInstance;
pub use ash_ad_wrappers::ash_surface_wrappers::{AdSurface, AdSurfaceInstance};
pub use renderables::{glam, Camera3D};
pub use renderables::triangle_mesh::{TriMeshCPU, TriMeshGPU, TriMeshTransform};
pub use renderables::flat_texture::FlatTextureGPU;

pub enum RendererMessage {
  UploadTriMesh(String, TriMeshCPU, Arc<OnceLock<Arc<TriMeshGPU>>>),
  UploadFlatTex(String, String, Arc<OnceLock<Arc<FlatTextureGPU>>>),
  SetCamera(Camera3D),
  Draw(Vec<(Arc<TriMeshGPU>, Arc<FlatTextureGPU>)>),
  Stop,
}

pub struct Renderer {
  thread: Option<std::thread::JoinHandle<Result<(), String>>>,
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
        let mut current_cmds = renderer_ordered_cmds
          .lock()
          .map_err(|e| format!("at getting lock for renderer work queue: {e}"))?;
        for message in current_cmds.drain(..) {
          match message {
            RendererMessage::UploadTriMesh(name, tri_mesh_cpu, tri_mesh_gpu) => {
              let _ = render_mgr
                .add_tri_mesh(name, &tri_mesh_cpu, tri_mesh_gpu)
                .inspect_err(|e| eprintln!("error adding mesh: {e}"));
            }
            RendererMessage::UploadFlatTex(name, flat_tex_path, flat_tex_gpu) => {
              let _ = render_mgr
                .add_flat_texture(name, flat_tex_path, flat_tex_gpu)
                .inspect_err(|e| eprintln!("error adding texture: {e}"));
            }
            RendererMessage::Draw(mesh_ftex_list) => {
              for _ in 0..3 {
                if let Ok(d_res) = render_mgr.draw(&mesh_ftex_list).inspect_err(|e| eprintln!("{}", e)) {
                  if !d_res {
                    break;
                  }
                }
              }
            }
            RendererMessage::Stop => {
              quit_renderer = true;
            }
            RendererMessage::SetCamera(camera3_d) =>{
              render_mgr.camera = camera3_d
            },
          }
        }
        current_cmds.clear();
        if quit_renderer {
          break;
        }
      }
      return Ok::<(), String>(());
    });
    Ok(Self { thread: Some(thread), ordered_cmds })
  }

  pub fn send_batch_sync(&mut self, mut batch: Vec<RendererMessage>) -> Result<bool, String> {
    loop {
      let mut current_cmds = self
        .ordered_cmds
        .lock()
        .map_err(|e| format!("at getting lock for renderer work queue: {e}"))?;
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
    let Some(thread) = self.thread.take() else { return; };

    if !thread.is_finished() {
      let _ = self
        .send_batch_sync(vec![RendererMessage::Stop])
        .inspect_err(|e| eprintln!("at stopping renderer: {e}"));
    }
    let _ = thread.join()
      .inspect_err(|_| eprintln!("at joining renderer thread"));
  }
}

const DEPTH_FORMAT_PREFERENCE: [vk::Format; 3] = [vk::Format::D24_UNORM_S8_UINT, vk::Format::D16_UNORM_S8_UINT, vk::Format::D32_SFLOAT];

pub struct RenderManager {
  triangle_frame_buffers: Vec<Arc<AdFrameBuffer>>,
  tri_mesh_tex_renderer: TriMeshTexRenderer,

  flat_texes: HashMap<String, Arc<FlatTextureGPU>>,
  flat_tex_gen: FlatTextureGenerator,
  tri_meshes: HashMap<String, Arc<TriMeshGPU>>,
  tri_mesh_gen: TriMeshGenerator,
  camera: Camera3D,

  gen_allocator: Arc<Mutex<Allocator>>,
  render_semaphores: Vec<AdSemaphore>,
  render_fences: Vec<AdFence>,
  render_cmd_buffers: Vec<AdCommandBuffer>,
  image_acquire_fence: AdFence,
  swapchain: AdSwapchain,
  depth_format: vk::Format,
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

    let mut depth_format = vk::Format::UNDEFINED;
    for format in DEPTH_FORMAT_PREFERENCE {
      let format_props = unsafe {
        ash_device.ash_instance().inner().get_physical_device_format_properties(gpu, format)
      };
      if format_props.optimal_tiling_features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT) {
        depth_format = format;
        break;
      }
    }
    if depth_format == vk::Format::UNDEFINED {
      return Err("preferred depth format not supported".to_string());
    }

    println!("depth format selected");

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
      .find(|m| **m == vk::PresentModeKHR::MAILBOX)
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
    let tri_mesh_allocator = Arc::new(Mutex::new(ash_device.create_allocator()?));
    let flat_tex_allocator = Arc::new(Mutex::new(ash_device.create_allocator()?));

    let tri_mesh_gen =
      TriMeshGenerator::new(tri_mesh_allocator, queues[&GPUQueueType::Transfer].clone())?;

    let flat_tex_gen =
      FlatTextureGenerator::new(flat_tex_allocator, queues[&GPUQueueType::Transfer].clone())?;

    let tri_mesh_tex_renderer =
      TriMeshTexRenderer::new(ash_device.clone(), &tri_mesh_gen, &flat_tex_gen, depth_format)?;

    let mut triangle_frame_buffers = tri_mesh_tex_renderer.create_framebuffers(
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
        .rename(&format!("triangle_color_image_{i}"))?;
      fb.attachments()[1]
        .image()
        .allocation()
        .lock()
        .map_err(|e| format!("at getting image mem lock: {e}"))?
        .rename(&format!("triangle_depth_image_{i}"))?;
    }

    let camera = Camera3D {
      pos: glam::vec4(2.0, 2.0, 2.0, 0.0),
      look_dir: glam::vec4(-1.0, -1.0, -1.0, 0.0),
      view_proj_mat: glam::Mat4::IDENTITY,
    };

    Ok(Self {
      ash_device,
      queues,
      depth_format,
      swapchain,
      image_acquire_fence,
      render_cmd_buffers,
      render_semaphores,
      render_fences,
      gen_allocator,
      triangle_frame_buffers,
      camera,
      tri_meshes: HashMap::new(),
      tri_mesh_gen,
      tri_mesh_tex_renderer,
      flat_texes: HashMap::new(),
      flat_tex_gen,
    })
  }

  pub fn add_tri_mesh(
    &mut self,
    name: String,
    mesh: &TriMeshCPU,
    output: Arc<OnceLock<Arc<TriMeshGPU>>>,
  ) -> Result<(), String> {
    let s_time = std::time::Instant::now();
    let tri_mesh_gpu = self
      .tri_meshes
      .entry(name.clone())
      .or_insert(Arc::new(self.tri_mesh_gen.upload_tri_mesh(&name, mesh)?));
    println!("mesh {} upload time: {}ms", &name, s_time.elapsed().as_millis());
    output
      .set(tri_mesh_gpu.clone())
      .map_err(|_| "at setting mesh output".to_string())?;
    Ok(())
  }

  pub fn add_flat_texture(
    &mut self,
    name: String,
    tex_path: String,
    output: Arc<OnceLock<Arc<FlatTextureGPU>>>
  ) -> Result<(), String> {
    let s_time = std::time::Instant::now();
    let flat_tex_gpu = self
      .flat_texes
      .entry(name.clone())
      .or_insert(Arc::new(self.flat_tex_gen.upload_flat_texture(&name, &tex_path)?));
    println!("tex {} upload time: {}ms", &name, s_time.elapsed().as_millis());
    output
      .set(flat_tex_gpu.clone())
      .map_err(|_| "at setting tex output".to_string())?;
    Ok(())
  }

  pub fn draw(
    &mut self,
    mesh_ftex_list: &[(Arc<TriMeshGPU>, Arc<FlatTextureGPU>)],
  ) -> Result<bool, String> {
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
      self.triangle_frame_buffers = self.tri_mesh_tex_renderer.create_framebuffers(
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
          .rename(&format!("triangle_color_image_{i}"))?;
        fb.attachments()[1]
          .image()
          .allocation()
          .lock()
          .map_err(|e| format!("at getting image mem lock: {e}"))?
          .rename(&format!("triangle_depth_image_{i}"))?;
      }
    }

    // Camera update
    let current_aspect_ratio = self.triangle_frame_buffers[image_idx as usize].resolution().width
      as f32
      / self.triangle_frame_buffers[image_idx as usize].resolution().height as f32;
    self.camera.refresh_vp_matrix(1.5, current_aspect_ratio);
    // if let AdDescriptorBinding::UniformBuffer(Some(cam_buffer)) = self.camera_dset.bindings()[0].clone() {
    //   cam_buffer.write_data(0, &[self.camera])?;
    // }

    self.render_cmd_buffers[image_idx as usize]
      .begin(vk::CommandBufferUsageFlags::default())
      .map_err(|e| format!("at beginning render cmd buffer:  {e}"))?;

    self.tri_mesh_tex_renderer.render(
      &self.render_cmd_buffers[image_idx as usize],
      &self.triangle_frame_buffers[image_idx as usize],
      self.camera,
      mesh_ftex_list,
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
