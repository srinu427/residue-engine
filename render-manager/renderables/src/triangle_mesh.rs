use std::sync::{Arc, Mutex};

use ash_ad_wrappers::{
  ash_context::{
    ash::vk,
    getset,
    gpu_allocator::{vulkan::Allocator, MemoryLocation},
  },
  ash_data_wrappers::{
    AdBuffer, AdDescriptorBinding, AdDescriptorPool, AdDescriptorSet, AdDescriptorSetLayout,
  },
  ash_queue_wrappers::{AdCommandBuffer, AdCommandPool, AdQueue},
  ash_sync_wrappers::AdFence,
};

pub fn g_vec4_from_vec3(v: glam::Vec3, w: f32) -> glam::Vec4 {
  glam::vec4(v.x, v.y, v.z, w)
}

#[derive(Debug)]
#[repr(C)]
pub struct TriMeshVertex {
  pub pos: glam::Vec4,
  pub normal: glam::Vec4,
  pub uv: glam::Vec4,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TriMeshTransform {
  pub transform: glam::Mat4,
}

pub struct TriMeshCPU {
  pub vertices: Vec<TriMeshVertex>,
  pub triangles: Vec<[u32; 3]>,
}

impl TriMeshCPU {
  pub fn merge(mut self, mut other: Self) -> Self {
    let curr_vert_len = self.vertices.len() as u32;
    for t in other.triangles.iter_mut() {
      for idx in t {
        *idx += curr_vert_len;
      }
    }
    self.vertices.append(&mut other.vertices);
    self.triangles.append(&mut other.triangles);
    self
  }

  pub fn make_rect(center: glam::Vec3, tangent: glam::Vec3, bitangent: glam::Vec3) -> Self {
    let normal = tangent.cross(bitangent).normalize();
    let verts = vec![
      TriMeshVertex {
        pos: g_vec4_from_vec3(center - tangent / 2.0 + bitangent / 2.0, 1.0),
        normal: g_vec4_from_vec3(normal, 1.0),
        uv: glam::vec4(0.0, 0.0, 0.0, 0.0),
      },
      TriMeshVertex {
        pos: g_vec4_from_vec3(center - tangent / 2.0 - bitangent / 2.0, 1.0),
        normal: g_vec4_from_vec3(normal, 1.0),
        uv: glam::vec4(0.0, bitangent.length() * 2.0, 0.0, 0.0),
      },
      TriMeshVertex {
        pos: g_vec4_from_vec3(center + tangent / 2.0 - bitangent / 2.0, 1.0),
        normal: g_vec4_from_vec3(normal, 1.0),
        uv: glam::vec4(tangent.length() * 2.0, bitangent.length() * 2.0, 0.0, 0.0),
      },
      TriMeshVertex {
        pos: g_vec4_from_vec3(center + tangent / 2.0 + bitangent / 2.0, 1.0),
        normal: g_vec4_from_vec3(normal, 1.0),
        uv: glam::vec4(tangent.length() * 2.0, 0.0, 0.0, 0.0),
      },
    ];
    let triangles = vec![[0, 1, 2], [2, 3, 0]];
    Self { vertices: verts, triangles }
  }

  pub fn make_cuboid(
    center: glam::Vec3,
    axis_x: glam::Vec3,
    axis_y: glam::Vec3,
    z_len: f32,
  ) -> Self {
    let axis_z = axis_x.cross(axis_y).normalize() * z_len;
    Self { vertices: vec![], triangles: vec![] }
      .merge(Self::make_rect(center + (axis_x / 2.0), axis_y, axis_z))
      .merge(Self::make_rect(center - (axis_x / 2.0), axis_z, axis_y))
      .merge(Self::make_rect(center + (axis_y / 2.0), axis_z, axis_x))
      .merge(Self::make_rect(center - (axis_y / 2.0), axis_x, axis_z))
      .merge(Self::make_rect(center + (axis_z / 2.0), axis_x, axis_y))
      .merge(Self::make_rect(center - (axis_z / 2.0), axis_y, axis_x))
  }

  pub fn make_planar_polygon(g_vertices: Vec<glam::Vec3>) -> Self {
    let normal = (g_vertices[1] - g_vertices[0]).cross(g_vertices[2] - g_vertices[1]).normalize();
    let tangent = (g_vertices[1] - g_vertices[0]).normalize();
    let bitangent = normal.cross(tangent).normalize();

    let vertices = g_vertices
      .iter()
      .map(|g_vertex| {
        TriMeshVertex {
          pos: g_vec4_from_vec3(*g_vertex, 1.0),
          normal: g_vec4_from_vec3(normal, 0.0),
          uv: glam::vec4(
            (g_vertex - g_vertices[0]).dot(tangent),
            (g_vertex - g_vertices[0]).dot(bitangent),
            0.0,
            0.0,
          ),
        }
      })
      .collect::<Vec<_>>();
    let triangles = (1..vertices.len() - 1)
      .map(|i| {
        let i = i as _;
        [0, i, i + 1]
      })
      .collect::<Vec<_>>();
    Self{vertices, triangles}
  }
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct TriMeshGPU {
  #[getset(get = "pub")]
  dset: Arc<AdDescriptorSet>,
  #[getset(get_copy = "pub")]
  indx_count: usize,
}

impl TriMeshGPU {
  pub fn update_transform(&self, t: TriMeshTransform) -> Result<(), String> {
    let AdDescriptorBinding::UniformBuffer(ob) = &self.dset.bindings()[2] else {
      return Err("Triangle mesh constructed with improper object data buffer".to_string())
    };
    ob.write_data(0, &[t])?;
    Ok(())
  }
}

#[derive(getset::Getters, getset::CopyGetters)]
pub struct TriMeshGenerator {
  allocator: Arc<Mutex<Allocator>>,
  cmd_pool: Arc<AdCommandPool>,
  mesh_dset_pool: Arc<AdDescriptorPool>,
  #[getset(get = "pub")]
  mesh_dset_layout: Arc<AdDescriptorSetLayout>,
}

impl TriMeshGenerator {
  pub fn new(allocator: Arc<Mutex<Allocator>>, queue: Arc<AdQueue>) -> Result<Self, String> {
    let ash_device = queue.ash_device().clone();
    let dset_pool = AdDescriptorPool::new(
      ash_device.clone(),
      vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
      3000,
      &[
        vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2000 },
        vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1000 },
      ],
    )?;
    let dset_layout = AdDescriptorSetLayout::new(
      ash_device.clone(),
      &[
        (vk::ShaderStageFlags::VERTEX, vk::DescriptorType::STORAGE_BUFFER),
        (vk::ShaderStageFlags::VERTEX, vk::DescriptorType::STORAGE_BUFFER),
        (vk::ShaderStageFlags::VERTEX, vk::DescriptorType::UNIFORM_BUFFER),
      ],
    )?;
    let cmd_pool = AdCommandPool::new(queue, vk::CommandPoolCreateFlags::TRANSIENT)?;
    Ok(Self {
      allocator,
      cmd_pool: Arc::new(cmd_pool),
      mesh_dset_pool: Arc::new(dset_pool),
      mesh_dset_layout: Arc::new(dset_layout),
    })
  }

  pub fn upload_tri_mesh(
    &self,
    name: &str,
    tri_mesh_cpu: &TriMeshCPU,
  ) -> Result<TriMeshGPU, String> {
    let ash_device = self.cmd_pool.queue().ash_device().clone();
    let cmd_buffer =
      AdCommandBuffer::new(self.cmd_pool.clone(), vk::CommandBufferLevel::PRIMARY, 1)?.remove(0);

    let vert_buffer_data = AdBuffer::get_byte_slice(&tri_mesh_cpu.vertices);
    let vert_buffer = AdBuffer::new(
      ash_device.clone(),
      self.allocator.clone(),
      MemoryLocation::GpuOnly,
      &format!("{name}_vb"),
      vk::BufferCreateFlags::empty(),
      vert_buffer_data.len() as _,
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
    )?;
    let vert_buffer_stage = AdBuffer::new(
      ash_device.clone(),
      self.allocator.clone(),
      MemoryLocation::CpuToGpu,
      &format!("{name}_vb_stage"),
      vk::BufferCreateFlags::empty(),
      vert_buffer_data.len() as _,
      vk::BufferUsageFlags::TRANSFER_SRC,
    )?;
    vert_buffer_stage.write_data(0, vert_buffer_data)?;

    let indx_buffer_data = AdBuffer::get_byte_slice(&tri_mesh_cpu.triangles);
    let indx_buffer = AdBuffer::new(
      ash_device.clone(),
      self.allocator.clone(),
      MemoryLocation::GpuOnly,
      &format!("{name}_ib"),
      vk::BufferCreateFlags::empty(),
      vert_buffer_data.len() as _,
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
    )?;
    let indx_buffer_stage = AdBuffer::new(
      ash_device.clone(),
      self.allocator.clone(),
      MemoryLocation::CpuToGpu,
      &format!("{name}_ib_stage"),
      vk::BufferCreateFlags::empty(),
      vert_buffer_data.len() as _,
      vk::BufferUsageFlags::TRANSFER_SRC,
    )?;
    indx_buffer_stage.write_data(0, indx_buffer_data)?;

    let obj_transform = vec![TriMeshTransform { transform: glam::Mat4::IDENTITY }];
    let objt_buffer_data = AdBuffer::get_byte_slice(&obj_transform);
    let objt_buffer = AdBuffer::new(
      ash_device.clone(),
      self.allocator.clone(),
      MemoryLocation::CpuToGpu,
      &format!("{name}_ob"),
      vk::BufferCreateFlags::empty(),
      vert_buffer_data.len() as _,
      vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
    )?;
    objt_buffer.write_data(0, objt_buffer_data)?;

    // Copy from stage buffers to gpu local
    cmd_buffer.begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
    cmd_buffer.copy_buffer_to_buffer_cmd(
      vert_buffer_stage.inner(),
      vert_buffer.inner(),
      &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: vert_buffer.size() }],
    );
    cmd_buffer.copy_buffer_to_buffer_cmd(
      indx_buffer_stage.inner(),
      indx_buffer.inner(),
      &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: indx_buffer.size() }],
    );
    cmd_buffer.end()?;

    let fence = AdFence::new(ash_device.clone(), vk::FenceCreateFlags::empty())?;
    cmd_buffer.submit(&[], &[], Some(&fence))?;
    fence.wait(999999999)?;

    let mesh_dset = AdDescriptorSet::new(
      self.mesh_dset_pool.clone(),
      &[(
        self.mesh_dset_layout.clone(),
        vec![
          AdDescriptorBinding::StorageBuffer(Arc::new(vert_buffer)),
          AdDescriptorBinding::StorageBuffer(Arc::new(indx_buffer)),
          AdDescriptorBinding::UniformBuffer(Arc::new(objt_buffer)),
        ],
      )],
    )?
    .remove(0);

    Ok(TriMeshGPU { dset: Arc::new(mesh_dset), indx_count: tri_mesh_cpu.triangles.len() * 3 })
  }
}
