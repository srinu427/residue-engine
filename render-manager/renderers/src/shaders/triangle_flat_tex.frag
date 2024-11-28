#version 460

#include "common_structs.glsl"

layout (location = 0) in vec4 inGlobalPos;
layout (location = 1) in vec4 inUV;

layout (location = 0) out vec4 outFragColor;

layout(set = 1, binding = 0) uniform sampler2D albedo_texture;
layout(set = 2, binding = 0) uniform sampler2D depth_texture;

layout(push_constant) uniform CamWrap { CamData data; } camera_buffer;

void main() {
	if (length(inGlobalPos - camera_buffer.data.pos) < texture(depth_texture, gl_FragCoord.xy).r) {
		outFragColor = texture(albedo_texture, inUV.xy);
	} else {
		discard;
	}
}
