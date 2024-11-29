#version 460

layout (location = 0) out vec4 outFragColor;

struct CamData {
  vec4 pos;
  vec4 dir;
  vec4 up;
  vec4 right;
  vec4 props;
};

layout(push_constant) uniform CamWrap { CamData data; } camera_buffer;

float circle_sdf(vec3 center, float radius, vec3 point) {
  return (point - center).length() - radius;
}

float raymarch(vec3 start, vec3 dir) {
  return 1.0;
}


void main() {
  vec2 uv = gl_FragCoord.xy / camera_buffer.data.props.xy;

  // Create camera basis vectors (viewing matrix)
  vec3 forward = normalize(camera_buffer.data.dir);
  vec3 right = normalize(camera_buffer.data.right);
  vec3 up = normalize(camera_buffer.data.up);
  
  // Calculate ray direction based on the pixel and the camera's FOV
  float aspect = camera_buffer.data.props.x / camera_buffer.data.props.y;
  vec2 p = (2.0 * uv - 1.0) * vec2(aspect, 1.0); // Map UV to [-1, 1]
  p *= tan(fov * 0.5);                          // Apply FOV scaling

  vec3 rayDir = normalize(p.x * right + p.y * up + forward);  // Ray direction

	outFragColor = texture(albedo_texture, inUV.xy);
}
