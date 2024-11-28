#version 460

#include "common_structs.glsl"

layout (location = 0) in vec4 inGlobalPos;
layout (location = 1) in vec4 inUV;

layout (location = 0) out vec4 outFragColor;

layout(set = 1, binding = 0) uniform sampler2D albedo_texture;

layout(push_constant) uniform CamWrap { CamData data; } camera_buffer;

void main() {
	outFragColor = texture(albedo_texture, inUV.xy);
}
