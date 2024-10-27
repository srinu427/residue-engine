#version 460

layout (location = 0) in vec4 inUV;

layout (location = 0) out vec4 outFragColor;

layout(set = 3, binding = 0) uniform sampler2D albedo_texture;

void main() {
	outFragColor = vec4(0.0f, 1.0f, 0.0f, 1.0f);
	outFragColor = texture(albedo_texture, inUV.xy);
}