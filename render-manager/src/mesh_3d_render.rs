use ash_wrappers::ash_data_wrappers::AdBuffer;

pub struct Mesh3D {
  name: String,
  vert_buffer: AdBuffer,
  indx_buffer: AdBuffer,
}
