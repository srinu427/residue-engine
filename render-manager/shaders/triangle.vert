#version 460
// #extension GL_KHR_vulkan_glsl: enable

struct VertexData {
  vec4 pos;
};

layout(std140, set = 0, binding = 0) readonly buffer VertexArray { VertexData verts[]; } vertex_buffer;

void main() {
	gl_Position = vertex_buffer.verts[gl_VertexIndex].pos;
}