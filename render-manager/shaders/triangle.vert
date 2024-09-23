#version 460
// #extension GL_KHR_vulkan_glsl: enable
// #extension GL_EXT_debug_printf : enable

struct VertexData {
  vec4 pos;
};

layout(std140, set = 0, binding = 0) readonly buffer VertexArray { VertexData verts[]; } vertex_buffer;
layout(std140, set = 0, binding = 1) readonly buffer IndexArray { uint inds[]; } index_buffer;

vec4 invert_y_axis(vec4 v) {
  return vec4(v.x, -v.y, v.z, v.w);
}


void main() {
  uint vert_id = index_buffer.inds[gl_VertexIndex];
  gl_Position = invert_y_axis(vertex_buffer.verts[vert_id].pos);
}