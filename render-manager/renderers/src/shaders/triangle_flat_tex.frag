#version 460

layout (location = 0) in vec4 inUV;

layout (location = 0) out vec4 outFragColor;

layout(set = 1, binding = 0) uniform sampler2D albedo_texture;

void main() {
	outFragColor = texture(albedo_texture, inUV.xy);
}
