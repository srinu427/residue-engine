#version 460
// #extension GL_KHR_vulkan_glsl: enable
// #extension GL_EXT_debug_printf : enable

#include "common_structs.glsl"

layout (location = 0) out vec4 outGlobalPos;
layout (location = 1) out vec4 outUV;

layout(std430, set = 0, binding = 0) readonly buffer VertexArray { VertexData verts[]; } vertex_buffer;
layout(std430, set = 0, binding = 1) readonly buffer IndexArray { uint inds[]; } index_buffer;
layout(std140, set = 0, binding = 2) uniform ObjectWrap { ObjectData data; } object_transfer;

layout(push_constant) uniform CamWrap { CamData data; } camera_buffer;

vec4 invert_y_axis(vec4 v) {
  return vec4(v.x, -v.y, v.z, v.w);
}

void main() {
  uint vert_id = index_buffer.inds[gl_VertexIndex];
  vec4 global_pos = object_transfer.data.transform * vertex_buffer.verts[vert_id].position;
  gl_Position = invert_y_axis(camera_buffer.data.view_proj_mat * global_pos);
  outGlobalPos = global_pos;
  outUV = vertex_buffer.verts[vert_id].uv;
  //debugPrintfEXT("%1.2v4f\n", gl_Position);
}
