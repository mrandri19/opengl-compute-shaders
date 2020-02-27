use gl::types::*;

use crate::shader::Shader;
use std::ffi::CString;

pub struct Program {
    id: GLuint,
}

impl Program {
    pub fn new(shaders: Vec<(Shader, GLenum)>) -> Result<Self, String> {
        let program = unsafe { gl::CreateProgram() };
        unsafe {
            for (shader, _) in &shaders {
                gl::AttachShader(program, shader.id());
            }
            gl::LinkProgram(program);

            for (shader, _) in &shaders {
                gl::DetachShader(program, shader.id());
            }
        }
        let mut success: GLint = 1;
        unsafe {
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
        }
        if success == 0 {
            let mut len: GLint = 0;
            unsafe {
                gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
            }
            let mut buffer: Vec<u8> = Vec::with_capacity(len as usize + 1);
            buffer.extend([b' '].iter().cycle().take(len as usize));
            let error: CString = unsafe { CString::from_vec_unchecked(buffer) };
            unsafe {
                gl::GetProgramInfoLog(
                    program,
                    len,
                    std::ptr::null_mut(),
                    error.as_ptr() as *mut gl::types::GLchar,
                );
            }

            return Err(error.to_string_lossy().to_string());
        }

        Ok(Program { id: program })
    }
    pub fn use_(&self) {
        unsafe { gl::UseProgram(self.get_id()) };
    }
    pub fn get_id(&self) -> GLuint {
        self.id
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteProgram(self.id);
        }
    }
}
