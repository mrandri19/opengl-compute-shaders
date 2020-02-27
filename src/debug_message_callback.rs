use gl::types::*;
use std::ffi::CStr;

pub extern "system" fn callback(
    source: GLenum,
    type_: GLenum,
    id: GLuint,
    severity: GLenum,
    _length: GLsizei,
    message: *const GLchar,
    _user_param: *mut GLvoid,
) {
    let mut _source = "";
    match source {
        gl::DEBUG_SOURCE_API => _source = "API",
        gl::DEBUG_SOURCE_WINDOW_SYSTEM => _source = "WINDOW SYSTEM",
        gl::DEBUG_SOURCE_SHADER_COMPILER => _source = "SHADER COMPILER",
        gl::DEBUG_SOURCE_THIRD_PARTY => _source = "THIRD PARTY",
        gl::DEBUG_SOURCE_APPLICATION => _source = "APPLICATION",
        gl::DEBUG_SOURCE_OTHER => _source = "UNKNOWN",
        _ => _source = "UNKNOWN",
    }

    let mut _type = "";
    match type_ {
        gl::DEBUG_TYPE_ERROR => _type = "ERROR",
        gl::DEBUG_TYPE_DEPRECATED_BEHAVIOR => _type = "DEPRECATED BEHAVIOR",
        gl::DEBUG_TYPE_UNDEFINED_BEHAVIOR => _type = "UDEFINED BEHAVIOR",
        gl::DEBUG_TYPE_PORTABILITY => _type = "PORTABILITY",
        gl::DEBUG_TYPE_PERFORMANCE => _type = "PERFORMANCE",
        gl::DEBUG_TYPE_OTHER => _type = "OTHER",
        gl::DEBUG_TYPE_MARKER => _type = "MARKER",
        _ => _type = "UNKNOWN",
    }

    let mut _severity = "";
    match severity {
        gl::DEBUG_SEVERITY_HIGH => _severity = "HIGH",
        gl::DEBUG_SEVERITY_MEDIUM => _severity = "MEDIUM",
        gl::DEBUG_SEVERITY_LOW => _severity = "LOW",
        gl::DEBUG_SEVERITY_NOTIFICATION => _severity = "NOTIFICATION",
        _ => _severity = "UNKNOWN",
    }

    if _severity == "NOTIFICATION" {
        return;
    }

    let msg = unsafe { CStr::from_ptr(message) };
    println!(
        "{}: {} of {} severity, raised from {}: {}",
        id,
        _type,
        _severity,
        _source,
        msg.to_owned().to_string_lossy()
    );

    if !(_severity == "LOW" || _severity == "NOTIFICATION") {
        // panic!();
    }
}
