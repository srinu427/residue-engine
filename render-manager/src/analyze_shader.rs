use ash_wrappers::ash;
use spirv_cross::{spirv, glsl};
use std::path::Path;

pub fn get_spv_ast(path: &Path) -> Result<spirv::Ast<glsl::Target>, String> {
  let mut file = std::fs::File::open(path)
    .map_err(|e| format!("at opening spv file: {e}"))?;
  let words = ash::util::read_spv(&mut file)
    .map_err(|e| format!("at reading spv file: {e}"))?;
  let module = spirv::Module::from_words(&words);
  spirv::Ast::<glsl::Target>::parse(&module)
    .map_err(|e| format!("at parsing spv file: {e}"))
}

pub fn analyze_shader(path: &Path) {
  if let Ok(mut ast) = get_spv_ast(path) {
    if let Ok(resources) = ast.get_shader_resources() {
      println!("{:?}", resources.storage_buffers[0]);
      println!("{:?}", ast.get_decoration(resources.storage_buffers[0].id, spirv::Decoration::Binding));
      println!("{:?}", ast.get_decoration(resources.storage_buffers[0].id, spirv::Decoration::DescriptorSet));
    }
  }
}