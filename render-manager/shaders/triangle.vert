#version 460
// #extension GL_KHR_vulkan_glsl: enable

struct VertexData {
  vec4 pos;
};

layout(std140, set = 0, binding = 0) readonly buffer VertexArray { VertexData verts[]; } vertex_buffer;

vec4 invert_y_axis(vec4 v) {
  return vec4(v.x, -v.y, v.z, v.w);
}


void main() {
	gl_Position = invert_y_axis(vertex_buffer.verts[gl_VertexIndex].pos);
}