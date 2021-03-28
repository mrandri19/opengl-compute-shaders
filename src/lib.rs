mod debug_message_callback;
mod program;
mod shader;

#[cfg(test)]
mod tests {
    use gl::types::*;
    use glsl::parser::Parse;
    use glsl::syntax::PreprocessorDefine;
    use glsl::visitor::Visit;
    use glsl::visitor::{Host, Visitor};
    use std::collections::HashMap;

    use std::ffi::CString;

    use glfw::Context;

    use crate::debug_message_callback;
    use crate::program::Program;
    use crate::shader;
    use glsl::syntax::ShaderStage;

    const RELATIVE_TOLERANCE: f32 = 1e-8;

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

    fn make_shader_src<'a>(src: &str, substs: &HashMap<&'a str, usize>) -> String {
        let mut shader = ShaderStage::parse(src).unwrap();

        let mut transformed_source = String::new();
        struct MyVisitor<'a> {
            substs: &'a HashMap<&'a str, usize>,
        }
        impl<'a> Visitor for MyVisitor<'a> {
            fn visit_preprocessor_define(&mut self, define: &mut PreprocessorDefine) -> Visit {
                match define {
                    PreprocessorDefine::ObjectLike { ident, value } => {
                        if value == "-1337" {
                            if ident.as_str() == "N" {
                                *value = self.substs["N"].to_string();
                            }
                            if ident.as_str() == "B" {
                                *value = self.substs["B"].to_string();
                            }
                            if ident.as_str() == "DATA_LEN" {
                                *value = self.substs["DATA_LEN"].to_string();
                            }
                            if ident.as_str() == "WORK_GROUPS" {
                                *value = self.substs["WORK_GROUPS"].to_string();
                            }
                            if ident.as_str() == "CHUNK_SIZE" {
                                *value = self.substs["CHUNK_SIZE"].to_string();
                            }
                            if ident.as_str() == "CHUNK_X" {
                                *value = self.substs["CHUNK_Y"].to_string();
                            }
                            if ident.as_str() == "CHUNK_Y" {
                                *value = self.substs["CHUNK_Y"].to_string();
                            }
                            if ident.as_str() == "CHUNK_Z" {
                                *value = self.substs["CHUNK_Z"].to_string();
                            }
                        }
                    }
                    _ => (),
                };

                Visit::Parent
            }
        }

        let mut my_visitor = MyVisitor { substs };
        shader.visit(&mut my_visitor);

        glsl::transpiler::glsl::show_translation_unit(&mut transformed_source, &shader);

        transformed_source
    }

    fn make_compute_shader_program(source: &str, substs: &HashMap<&str, usize>) -> Program {
        let kernel = shader::Shader::from_source(
            &CString::new(make_shader_src(source, &substs)).unwrap(),
            gl::COMPUTE_SHADER,
        )
        .unwrap();
        Program::new(vec![(kernel, gl::COMPUTE_SHADER)]).unwrap()
    }

    fn get_ssbo<T: Clone>(buffer: GLuint) -> T {
        unsafe {
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, buffer);
            let input_data = gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *const T;

            gl::UnmapBuffer(gl::SHADER_STORAGE_BUFFER);
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);

            return (*input_data).clone();
        }
    }

    #[test]
    fn test_single_wg_prefix_sum() {
        // Maximum number of threads is 1024 and each thread processes 2 elements
        const DATA_LEN: usize = 2048;
        #[derive(Debug, Copy, Clone)]
        #[repr(C, packed)]
        struct InputData {
            data: [GLfloat; DATA_LEN],
        }

        #[derive(Debug, Copy, Clone)]
        #[repr(C, packed)]
        struct OutputData {
            length: GLuint,
            data: [GLfloat; DATA_LEN],
        }

        // *************************************************************************
        // Create OpenGL Context
        let _window = make_opengl_window();

        // *************************************************************************
        // Load shader and create program
        let mut substs = std::collections::HashMap::new();
        substs.insert("N", DATA_LEN);
        let program = make_compute_shader_program(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/single_wg_prefix_sum/prefix_sum.comp.glsl"
            )),
            &substs,
        );

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

        let result = get_ssbo::<InputData>(input_ssbo);
        for i in 0..DATA_LEN {
            let result_value = result.data[i];
            assert_eq!(expected[i], result_value);
        }

        // *************************************************************************
        // Cleanup
        unsafe { gl::DeleteBuffers(1, &input_ssbo) };
    }

    #[test]
    fn test_single_wg_compaction() {
        const DATA_LEN: usize = 2048;
        #[derive(Debug, Copy, Clone)]
        #[repr(C, packed)]
        struct InputData {
            data: [GLfloat; DATA_LEN],
        }

        #[derive(Debug, Copy, Clone)]
        #[repr(C, packed)]
        struct OutputData {
            length: GLuint,
            data: [GLfloat; DATA_LEN],
        }

        // *************************************************************************
        // Create OpenGL Context
        let _window = make_opengl_window();

        // *************************************************************************
        // Load shader and create program

        let mut substs = std::collections::HashMap::new();
        substs.insert("N", DATA_LEN);
        let program = make_compute_shader_program(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/single_wg_compaction/compaction.comp.glsl"
            )),
            &substs,
        );

        // *************************************************************************
        // Create random data
        let input_data = InputData {
            data: {
                let mut data = [0.0; DATA_LEN];
                for i in 0..DATA_LEN {
                    data[i] = i as GLfloat + 1.0;
                }
                data
            },
        };

        // *************************************************************************
        // Calculate expected result

        let mut expected = [0.0; DATA_LEN];
        for i in 0..(DATA_LEN / 2) {
            expected[i] = 2.0 * (i + 1) as GLfloat;
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

        let mut output_ssbo = 0;
        let output_index_binding_point = 1;
        unsafe {
            gl::GenBuffers(1, &mut output_ssbo);
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, output_ssbo);
            gl::BufferData(
                gl::SHADER_STORAGE_BUFFER,
                std::mem::size_of::<OutputData>() as GLsizeiptr,
                std::ptr::null() as *const GLvoid,
                gl::DYNAMIC_READ,
            );

            gl::BindBufferBase(
                gl::SHADER_STORAGE_BUFFER,
                output_index_binding_point,
                output_ssbo,
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
        let output_struct = get_ssbo::<OutputData>(output_ssbo);

        for i in 0..DATA_LEN {
            let output_value = output_struct.data[i];
            assert!((expected[i] - output_value).abs() <= (RELATIVE_TOLERANCE * output_value));
        }

        // *************************************************************************
        // Cleanup
        unsafe { gl::DeleteBuffers(1, &input_ssbo) };
    }

    #[test]
    fn test_multiple_wg_prefix_sum() {
        // TODO(Andrea): understand why it doesn't work for non powers of 2 =>
        // It needs to be padded to closest multiple of two
        const DATA_LEN: usize = 17;
        const WORK_GROUPS: usize = 4;

        // See https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
        // 39.2.4 Arrays of Arbitrary Size

        #[derive(Debug, Copy, Clone)]
        #[repr(C, packed)]
        struct InputData {
            data: [GLfloat; DATA_LEN],
        }

        #[derive(Debug, Copy, Clone)]
        #[repr(C, packed)]
        struct OutputData {
            sums: [GLfloat; WORK_GROUPS],
            data: [GLfloat; DATA_LEN],
        }

        // *************************************************************************
        // Create OpenGL Context
        let _window = make_opengl_window();

        // *************************************************************************
        // Load shader and create program

        let mut substs = std::collections::HashMap::new();
        substs.insert("DATA_LEN", DATA_LEN);
        substs.insert("WORK_GROUPS", WORK_GROUPS);
        let program1 = make_compute_shader_program(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/multi_wg_prefix_sum/multi_wg_prefix_sum1.comp.glsl"
            )),
            &substs,
        );
        let program2 = make_compute_shader_program(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/multi_wg_prefix_sum/multi_wg_prefix_sum2.comp.glsl"
            )),
            &substs,
        );

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

        let mut output_ssbo = 0;
        let output_index_binding_point = 1;
        unsafe {
            gl::GenBuffers(1, &mut output_ssbo);
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, output_ssbo);
            gl::BufferData(
                gl::SHADER_STORAGE_BUFFER,
                std::mem::size_of::<OutputData>() as GLsizeiptr,
                std::ptr::null() as *const GLvoid,
                gl::DYNAMIC_READ,
            );

            gl::BindBufferBase(
                gl::SHADER_STORAGE_BUFFER,
                output_index_binding_point,
                output_ssbo,
            );

            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
        }

        // *************************************************************************
        // Run compute shader
        program1.use_();
        unsafe {
            gl::DispatchCompute(WORK_GROUPS as GLuint, 1, 1);
            gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
        };

        fn inplace_exclusive_prefix_sum(a: &mut [GLfloat; WORK_GROUPS]) {
            let mut v = a.to_vec();
            v.insert(0, 0.);
            for i in 1..v.len() {
                v[i] += v[i - 1];
            }
            v.pop();
            for i in 0..v.len() {
                a[i] = v[i];
            }
        }

        // TODO(Andrea): should this be a kernel to avoid moving memory?
        unsafe {
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, output_ssbo);

            let ptr = gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *mut OutputData;

            inplace_exclusive_prefix_sum(&mut ((*ptr).sums));

            gl::UnmapBuffer(gl::SHADER_STORAGE_BUFFER);
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
        }

        program2.use_();
        unsafe {
            gl::DispatchCompute(WORK_GROUPS as GLuint, 1, 1);
            gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
        };

        // *************************************************************************
        // Check expected result matches with output
        let output_struct = get_ssbo::<OutputData>(output_ssbo);

        dbg!(&output_struct);

        for i in 0..DATA_LEN {
            let output_value = output_struct.data[i];
            assert!((expected[i] - output_value).abs() <= (RELATIVE_TOLERANCE * output_value));
        }

        // *************************************************************************
        // Cleanup
        unsafe { gl::DeleteBuffers(1, &input_ssbo) };
    }
}
