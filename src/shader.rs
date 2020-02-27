use gl::types::*;
use std::ffi::CStr;
use std::ffi::CString;
pub struct Shader {
    id: GLuint,
}

impl Shader {
    pub fn from_source(source: &CStr, kind: GLenum) -> Result<Shader, String> {
        let id = unsafe { gl::CreateShader(kind) };
        unsafe {
            gl::ShaderSource(id, 1, &source.as_ptr(), std::ptr::null());
            gl::CompileShader(id);
        }
        let mut success: GLint = 1;
        unsafe {
            gl::GetShaderiv(id, gl::COMPILE_STATUS, &mut success);
        }
        if success == 0 {
            let mut len: GLint = 0;
            unsafe {
                gl::GetShaderiv(id, gl::INFO_LOG_LENGTH, &mut len);
            }
            let mut buffer: Vec<u8> = Vec::with_capacity(len as usize + 1);
            buffer.extend([b' '].iter().cycle().take(len as usize));
            let error: CString = unsafe { CString::from_vec_unchecked(buffer) };
            unsafe {
                gl::GetShaderInfoLog(
                    id,
                    len,
                    std::ptr::null_mut(),
                    error.as_ptr() as *mut gl::types::GLchar,
                );
            }

            return Err(error.to_string_lossy().to_string());
        }
        Ok(Shader { id })
    }
    pub fn id(&self) -> GLuint {
        self.id
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe { gl::DeleteShader(self.id) };
    }
}
