use std::{ops::{Add, Mul}, rc::Rc};

use glam::Vec4Swizzles;

pub struct KeyFramed<T: Clone + Mul<f32, Output = T> + Add<Output = T>> {
  pub key_frames: Vec<(u128, T)>,
}

impl<T> KeyFramed<T> where T: Clone + Mul<f32, Output = T> + Add<Output = T> {
  pub fn search_key_frame_idx(&self, time_ms: u128) -> usize {
    let mut begin_idx = 0;
    let mut end_idx = self.key_frames.len() - 1;
    
    loop {
      if begin_idx == end_idx {
        return begin_idx;
      }
      let check_idx = (begin_idx + end_idx) / 2;
      if self.key_frames[check_idx].0 <= time_ms {
        if self.key_frames[check_idx + 1].0 > time_ms {
          return check_idx;
        } else {
          begin_idx = check_idx + 1;
        }
      } else {
        end_idx = check_idx;
      }
    }
  }

  pub fn value_at(&self, time_ms: u128) -> T {
    let kf_idx = self.search_key_frame_idx(time_ms);
    if kf_idx == self.key_frames.len() - 1 {
      return self.key_frames[kf_idx].1.clone();
    } else {
      let mix_factor = (time_ms - self.key_frames[kf_idx].0) as f32 / (self.key_frames[kf_idx + 1].0 - self.key_frames[kf_idx].0) as f32;
      return (self.key_frames[kf_idx].1.clone() * mix_factor) + (self.key_frames[kf_idx + 1].1.clone() * (1.0 - mix_factor));
    }
  }
}


pub struct RTSAnimation {
  pos: KeyFramed<glam::Vec4>,
  roatation: KeyFramed<glam::Vec4>,
  scale: KeyFramed<glam::Vec4>,
}

pub struct RTSAnimator {
  anim: Rc<RTSAnimation>,
  current_time: u128,
  repeat_after: Option<u128>,
}

impl RTSAnimator {
  pub fn forward(&mut self, time_ms: u128) {
    self.current_time += time_ms;
  }

  pub fn reset(&mut self) {
    self.current_time = 0;
  }

  pub fn get_transform(&self) -> glam::Mat4 {
    let cur_pos = self.anim.pos.value_at(self.current_time);
    let cur_rot = self.anim.roatation.value_at(self.current_time);
    let cur_scl = self.anim.scale.value_at(self.current_time);
    let rot_mat = glam::Mat4::from_axis_angle(cur_rot.xyz().normalize(), cur_rot.xyz().length());
    let pos_mat = glam::Mat4::from_translation(cur_pos.xyz());
    let scl_mat = glam::Mat4::from_scale(cur_scl.xyz());
    pos_mat * rot_mat * scl_mat
  }
}
