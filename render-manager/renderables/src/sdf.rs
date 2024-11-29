pub fn g_vec4_from_vec3(v: glam::Vec3, w: f32) -> glam::Vec4 {
  glam::vec4(v.x, v.y, v.z, w)
}

#[repr(C)]
pub struct SDFSphere {
  pub pos: glam::Vec4,
  pub radius: f32,
}

#[repr(C)]
pub struct SDFBBVertex {
  pub pos: glam::Vec4,
  pub normal: glam::Vec4,
  pub uv: glam::Vec4,
}

pub struct SDFBBCPU {
  pub verts: Vec<SDFBBVertex>,
  pub triangles: Vec<[u32; 3]>,
}

impl SDFBBCPU {
  pub fn merge(mut self, mut other: Self) -> Self {
    let curr_vert_len = self.verts.len() as u32;
    for t in other.triangles.iter_mut() {
      for idx in t {
        *idx += curr_vert_len;
      }
    }
    self.verts.append(&mut other.verts);
    self.triangles.append(&mut other.triangles);
    self
  }

  pub fn make_rect(center: glam::Vec3, tangent: glam::Vec3, bitangent: glam::Vec3) -> Self {
    let normal = tangent.cross(bitangent).normalize();
    let verts = vec![
      SDFBBVertex {
        pos: g_vec4_from_vec3(center - tangent / 2.0 + bitangent / 2.0, 1.0),
        normal: g_vec4_from_vec3(normal, 1.0),
        uv: glam::vec4(0.0, 0.0, 0.0, 0.0),
      },
      SDFBBVertex {
        pos: g_vec4_from_vec3(center - tangent / 2.0 - bitangent / 2.0, 1.0),
        normal: g_vec4_from_vec3(normal, 1.0),
        uv: glam::vec4(0.0, bitangent.length() * 2.0, 0.0, 0.0),
      },
      SDFBBVertex {
        pos: g_vec4_from_vec3(center + tangent / 2.0 - bitangent / 2.0, 1.0),
        normal: g_vec4_from_vec3(normal, 1.0),
        uv: glam::vec4(tangent.length() * 2.0, bitangent.length() * 2.0, 0.0, 0.0),
      },
      SDFBBVertex {
        pos: g_vec4_from_vec3(center + tangent / 2.0 + bitangent / 2.0, 1.0),
        normal: g_vec4_from_vec3(normal, 1.0),
        uv: glam::vec4(tangent.length() * 2.0, 0.0, 0.0, 0.0),
      },
    ];
    let triangles = vec![[0, 1, 2], [2, 3, 0]];
    Self { verts, triangles }
  }

  pub fn make_cuboid(
    center: glam::Vec3,
    axis_x: glam::Vec3,
    axis_y: glam::Vec3,
    z_len: f32,
  ) -> Self {
    let axis_z = axis_x.cross(axis_y).normalize() * z_len;
    Self { verts: vec![], triangles: vec![] }
      .merge(Self::make_rect(center + (axis_x / 2.0), axis_y, axis_z))
      .merge(Self::make_rect(center - (axis_x / 2.0), axis_z, axis_y))
      .merge(Self::make_rect(center + (axis_y / 2.0), axis_z, axis_x))
      .merge(Self::make_rect(center - (axis_y / 2.0), axis_x, axis_z))
      .merge(Self::make_rect(center + (axis_z / 2.0), axis_x, axis_y))
      .merge(Self::make_rect(center - (axis_z / 2.0), axis_y, axis_x))
  }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SDFTransform {
  pub transform: glam::Mat4,
}

