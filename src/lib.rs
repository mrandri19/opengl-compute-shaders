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
                    PreprocessorDefine::ObjectLike { ident, value } if value == "-1337" => {
                        *value = self.substs[ident.as_str()].to_string()
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
        match Program::new(vec![(kernel, gl::COMPUTE_SHADER)]) {
            Ok(res) => res,
            Err(err) => {
                println!("{}", err);
                std::process::exit(1);
            }
        }
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

    fn make_input_ssbo<T>(input_data: &T) -> GLuint {
        let mut ssbo = 0;
        let index_binding_point = 0;
        unsafe {
            gl::GenBuffers(1, &mut ssbo);
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, ssbo);
            gl::NamedBufferData(
                ssbo,
                std::mem::size_of::<T>() as GLsizeiptr,
                std::mem::transmute(input_data),
                gl::DYNAMIC_READ,
            );

            gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, index_binding_point, ssbo);

            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
        }
        return ssbo;
    }

    fn make_output_ssbo<T>() -> GLuint {
        let mut ssbo = 0;
        let index_binding_point = 1;
        unsafe {
            gl::GenBuffers(1, &mut ssbo);
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, ssbo);
            gl::BufferData(
                gl::SHADER_STORAGE_BUFFER,
                std::mem::size_of::<T>() as GLsizeiptr,
                std::mem::transmute(std::ptr::null::<T>()),
                gl::DYNAMIC_READ,
            );

            gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, index_binding_point, ssbo);

            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
        }
        return ssbo;
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
        let input_ssbo = make_input_ssbo(&input_data);

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
        let input_ssbo = make_input_ssbo(&input_data);
        let output_ssbo = make_output_ssbo::<OutputData>();

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

        assert_eq!(output_struct.length as usize, DATA_LEN / 2);
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
        // See https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
        // 39.2.4 Arrays of Arbitrary Size
        const DATA_LEN: usize = 262_144;
        const WORK_GROUPS: usize = 128;

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
        let input_ssbo = make_input_ssbo(&input_data);
        let output_ssbo = make_output_ssbo::<OutputData>();

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

        // TODO(Andrea): should this be a GPU kernel to avoid moving memory?
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

        for i in 0..DATA_LEN {
            let output_value = output_struct.data[i];
            assert!((expected[i] - output_value).abs() <= (RELATIVE_TOLERANCE * output_value));
        }

        // *************************************************************************
        // Cleanup
        unsafe { gl::DeleteBuffers(1, &input_ssbo) };
    }

    #[test]
    fn test_single_wg_raycasting() {
        const CHUNK_ROWS: usize = 8;
        const CHUNK_COLS: usize = 8;
        const CHUNK_SIZE: usize = CHUNK_ROWS * CHUNK_COLS;
        type GLvec4 = [GLfloat; 4];

        // See https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
        // 39.2.4 Arrays of Arbitrary Size

        #[derive(Debug, Copy, Clone)]
        #[repr(C, packed)]
        struct InputData {
            chunk: [GLuint; CHUNK_SIZE],
            ray_start: GLvec4,
            ray_direction: GLvec4,
        }

        #[derive(Debug, Copy, Clone)]
        #[repr(C, packed)]
        struct OutputData {
            hit: GLvec4,
            has_hit: GLboolean,
        }

        // *************************************************************************
        // Create OpenGL Context
        let _window = make_opengl_window();

        // *************************************************************************
        // Load shader and create program

        let mut substs = std::collections::HashMap::new();
        substs.insert("CHUNK_ROWS", CHUNK_ROWS);
        substs.insert("CHUNK_COLS", CHUNK_COLS);
        substs.insert("MAX_ITERS", 100);
        substs.insert("THREADS", 1);
        let program = make_compute_shader_program(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/single_wg_raycasting/single_wg_raycasting.comp.glsl"
            )),
            &substs,
        );

        // *************************************************************************
        // Create random data
        let chunk = [
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            // hits here ^
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ];

        let input_data = InputData {
            chunk: unsafe { std::mem::transmute(chunk) },
            ray_start: [0.0, 0.0, 0.0, 0.0],
            ray_direction: [1.0, 1.0, 0.0, 0.0],
        };

        // *************************************************************************
        // Calculate expected result

        // *************************************************************************
        // Create input and output SSBOs
        let input_ssbo = make_input_ssbo(&input_data);
        let output_ssbo = make_output_ssbo::<OutputData>();

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
        let hit = output_struct.hit;
        assert_eq!(hit, [4.0, 4.0, 0.0, 0.0]);
        assert_eq!(output_struct.has_hit, 1);

        // *************************************************************************
        // Cleanup
        unsafe { gl::DeleteBuffers(1, &input_ssbo) };
    }

    #[test]
    fn test_multiple_wg_compaction() {
        // TODO(Andrea): understand why it doesn't work for non powers of 2 =>
        // It needs to be padded to closest multiple of two

        // let N be the number of elements in the input array
        const N: usize = 131_072;
        // let B be the number of elements processed in a block
        const B: usize = 128;
        // then we need to allocate N/B thread blocks of B/2 threads each (since
        // each thread processes two elements)
        const N_OVER_B: usize = N / B;

        // See https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
        // 39.2.4 Arrays of Arbitrary Size

        #[derive(Debug, Copy, Clone)]
        #[repr(C, packed)]
        struct InputData {
            data: [GLuint; N],
        }

        #[derive(Debug, Copy, Clone)]
        #[repr(C, packed)]
        struct OutputData {
            sums: [GLuint; N_OVER_B],
            offsets: [GLuint; N],
            results: [GLuint; N],
            data: [GLuint; N],
        }

        // *************************************************************************
        // Create OpenGL Context
        let _window = make_opengl_window();

        // *************************************************************************
        // Load shader and create program

        let mut substs = std::collections::HashMap::new();
        substs.insert("N", N);
        substs.insert("B", B);
        let program1 = make_compute_shader_program(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/multi_wg_compaction/multi_wg_compaction1.comp.glsl"
            )),
            &substs,
        );
        let program2 = make_compute_shader_program(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/multi_wg_compaction/multi_wg_compaction2.comp.glsl"
            )),
            &substs,
        );

        // *************************************************************************
        // Create random data
        let data = {
            let mut arr = [0; N];
            for i in 0..N {
                arr[i] = i as GLuint + 1;
            }
            arr
        };
        let input_data = InputData { data };

        // *************************************************************************
        // Calculate expected result

        fn prefix_sum(data: Vec<GLuint>) -> Vec<GLuint> {
            let mut v = data.clone();
            v.insert(0, 0);
            for i in 1..v.len() {
                v[i] += v[i - 1];
            }
            v.pop();
            v
        }

        let expected_offsets = prefix_sum(
            data.to_vec()
                .iter()
                .map(|n| (n % 2 == 0) as GLuint)
                .collect::<Vec<GLuint>>(),
        );

        // *************************************************************************
        // Create input and output SSBOs
        let input_ssbo = make_input_ssbo(&input_data);
        let output_ssbo = make_output_ssbo::<OutputData>();

        // *************************************************************************
        // Run compute shader
        program1.use_();
        unsafe {
            gl::DispatchCompute(N_OVER_B as GLuint, 1, 1);
            gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
        };

        fn inplace_exclusive_prefix_sum(a: &mut [GLuint; N_OVER_B]) {
            let mut v = a.to_vec();
            v.insert(0, 0);
            for i in 1..v.len() {
                v[i] += v[i - 1];
            }
            v.pop();
            for i in 0..v.len() {
                a[i] = v[i];
            }
        }

        // TODO(Andrea): should this be a GPU kernel to avoid moving memory?
        unsafe {
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, output_ssbo);

            let ptr = gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *mut OutputData;

            inplace_exclusive_prefix_sum(&mut ((*ptr).sums));

            gl::UnmapBuffer(gl::SHADER_STORAGE_BUFFER);
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
        }

        program2.use_();
        unsafe {
            gl::DispatchCompute(N_OVER_B as GLuint, 1, 1);
            gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
        };

        // *************************************************************************
        // Check expected result matches with output
        let output_struct = get_ssbo::<OutputData>(output_ssbo);

        let offsets = output_struct.offsets;

        assert_eq!(
            offsets.to_vec(),
            expected_offsets,
            "The resulting offsets should match"
        );

        let computed_len = {
            let mut i = N - 1;
            let mut len = 0;
            loop {
                if output_struct.results[i] == 1 {
                    len = output_struct.offsets[i] + 1;
                    break;
                }
                if i > 0 {
                    i -= 1;
                } else {
                    break;
                }
            }
            len
        };
        assert_eq!(N / 2, computed_len as usize);

        // *************************************************************************
        // Cleanup
        unsafe { gl::DeleteBuffers(1, &input_ssbo) };
    }
}
