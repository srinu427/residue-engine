use std::sync::{Arc, OnceLock};
use render_manager::{FlatTextureGPU, Renderer, RendererMessage, TriMeshGPU};

pub enum Renderable {
  TriangleMeshFlatTexture{
    mesh: Arc<OnceLock<Arc<TriMeshGPU>>>,
    texture: Arc<OnceLock<Arc<FlatTextureGPU>>>,
  },
}

impl Renderable {

}