mod debug_message_callback;
mod program;
mod shader;
mod vertex;

#[cfg(test)]
mod tests {
    use gl::types::*;

    use std::ffi::CString;

    use glfw::Context;

    use crate::debug_message_callback;
    use crate::program::Program;
    use crate::shader;

    const DATA_LEN: usize = 32;

    #[derive(Debug, Copy, Clone)]
    #[repr(C, packed)]
    struct InputData {
        data: [GLfloat; DATA_LEN],
    }

    #[derive(Debug, Copy, Clone)]
    #[repr(C, packed)]
    struct OutputData {
        data: [GLfloat; DATA_LEN],
    }

    fn make_opengl_window() -> glfw::Window {
        let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
        glfw.window_hint(glfw::WindowHint::Visible(false));
        glfw.window_hint(glfw::WindowHint::ContextVersion(4, 6));
        glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
        glfw.window_hint(glfw::WindowHint::OpenGlDebugContext(true));

        let (mut window, _) = glfw
            .create_window(300, 300, "Hello this is window", glfw::WindowMode::Windowed)
            .expect("Failed to create GLFW window.");
        window.make_current();
        gl::load_with(|s| window.get_proc_address(s));

        unsafe {
            gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
            gl::DebugMessageCallback(Some(debug_message_callback::callback), std::ptr::null())
        }

        return window;
    }

    fn make_compute_shader_program(source: &str) -> Program {
        let kernel =
            shader::Shader::from_source(&CString::new(source).unwrap(), gl::COMPUTE_SHADER)
                .unwrap();
        Program::new(vec![(kernel, gl::COMPUTE_SHADER)]).unwrap()
    }

    fn get_ssbo(buffer: GLuint) -> Vec<f32> {
        unsafe {
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, buffer);
            let input_data =
                gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *const InputData;

            gl::UnmapBuffer(gl::SHADER_STORAGE_BUFFER);
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);

            return (*input_data).data.to_vec();
        }
    }

    #[test]
    fn test_prefix_sum() {
        // *************************************************************************
        // Create OpenGL Context
        let mut window = make_opengl_window();
        println!("[prefix_sum] created window");

        // *************************************************************************
        // Load shader and create program
        let program = make_compute_shader_program(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/prefix_sum.comp.glsl"
        )));

        // *************************************************************************
        // Create random data
        let input_data = InputData {
            data: [1.0; DATA_LEN],
        };

        // *************************************************************************
        // Calculate expected result
        let mut expected = [0.0; DATA_LEN];
        for i in 1..DATA_LEN {
            expected[i] += expected[i - 1] + input_data.data[i];
        }

        // *************************************************************************
        // Create input and output SSBOs
        let mut input_ssbo = 0;
        let input_index_binding_point = 0;
        unsafe {
            gl::GenBuffers(1, &mut input_ssbo);
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, input_ssbo);
            gl::NamedBufferData(
                input_ssbo,
                std::mem::size_of::<InputData>() as GLsizeiptr,
                std::mem::transmute(&input_data),
                gl::DYNAMIC_READ,
            );

            gl::BindBufferBase(
                gl::SHADER_STORAGE_BUFFER,
                input_index_binding_point,
                input_ssbo,
            );

            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
        }

        // *************************************************************************
        // Run compute shader
        program.use_();

        unsafe {
            gl::DispatchCompute(1, 1, 1);
            gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
        };

        // *************************************************************************
        // Check expected result matches with output

        let result = get_ssbo(input_ssbo);
        for i in 0..DATA_LEN {
            assert_eq!(expected[i], result[i]);
        }

        // *************************************************************************
        // Cleanup
        unsafe { gl::DeleteBuffers(1, &input_ssbo) };
    }

    #[test]
    fn test_daw() {
        // *************************************************************************
        // Create OpenGL Context
        let window = make_opengl_window();
        println!("[daw] created window");

        // *************************************************************************
        // Load shader and create program
        let program = make_compute_shader_program(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/prefix_sum.comp.glsl"
        )));

        // *************************************************************************
        // Create random data
        let input_data = InputData {
            data: [1.0; DATA_LEN],
        };

        // *************************************************************************
        // Calculate expected result
        let mut expected = [0.0; DATA_LEN];
        for i in 1..DATA_LEN {
            expected[i] += expected[i - 1] + input_data.data[i];
        }

        // *************************************************************************
        // Create input and output SSBOs
        let mut input_ssbo = 0;
        let input_index_binding_point = 0;
        unsafe {
            gl::GenBuffers(1, &mut input_ssbo);
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, input_ssbo);
            gl::NamedBufferData(
                input_ssbo,
                std::mem::size_of::<InputData>() as GLsizeiptr,
                std::mem::transmute(&input_data),
                gl::DYNAMIC_READ,
            );

            gl::BindBufferBase(
                gl::SHADER_STORAGE_BUFFER,
                input_index_binding_point,
                input_ssbo,
            );

            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
        }

        // *************************************************************************
        // Run compute shader
        program.use_();

        unsafe {
            gl::DispatchCompute(1, 1, 1);
            gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
        };

        // *************************************************************************
        // Check expected result matches with output
        let result = get_ssbo(input_ssbo);
        for i in 0..DATA_LEN {
            assert_eq!(expected[i], result[i]);
        }

        // *************************************************************************
        // Cleanup
        unsafe { gl::DeleteBuffers(1, &input_ssbo) };
    }
}
