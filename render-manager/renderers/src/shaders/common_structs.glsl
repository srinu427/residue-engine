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