#version 460

#include "../../shaders/common_structs.glsl"

layout (location = 0) in vec4 inGlobalPos;

layout (location = 0) out float outFragDepth;

layout(push_constant) uniform CamWrap { CamData data; } camera_buffer;

void main() {
	outFragDepth = length(inGlobalPos - camera_buffer.data.pos);
}
