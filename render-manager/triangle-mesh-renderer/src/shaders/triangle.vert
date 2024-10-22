#version 460
// #extension GL_KHR_vulkan_glsl: enable
// #extension GL_EXT_debug_printf : enable

layout (location = 0) out vec4 outUV;

struct VertexData {
  vec4 position;
  vec4 normal;
  vec4 uv;
};

struct ObjectData {
  mat4 transform;
};

struct CamData {
  vec4 pos;
  vec4 look_at;
  mat4 view_proj_mat;
};

layout(std430, set = 0, binding = 0) readonly buffer VertexArray { VertexData verts[]; } vertex_buffer;
layout(std430, set = 0, binding = 1) readonly buffer IndexArray { uint inds[]; } index_buffer;

layout(std140, set = 1, binding = 0) uniform CamWrap { CamData data; } camera_buffer;

vec4 invert_y_axis(vec4 v) {
  return vec4(v.x, -v.y, v.z, v.w);
}

void main() {
  uint vert_id = index_buffer.inds[gl_VertexIndex];
  vec4 global_pos = vertex_buffer.verts[vert_id].position;
  gl_Position = invert_y_axis(camera_buffer.data.view_proj_mat * global_pos);
  //gl_Position = vec4(gl_Position.x, gl_Position.y, gl_Position.z, 1.0);
  outUV = vertex_buffer.verts[vert_id].uv;
  //debugPrintfEXT("%1.2v4f\n", gl_Position);
  //debugPrintfEXT("%1.2v4f\n", camera_buffer.data.look_at);
}