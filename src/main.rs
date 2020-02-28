use gl::types::*;

// use glutin::event::WindowEvent;
mod debug_message_callback;
mod program;
mod shader;
mod vertex;
// use shader::Shader;
use std::ffi::CString;

const DATA_LEN: usize = 8;

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

macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {{
        let eps = 1.0e-6;
        let (a, b) = (&$a, &$b);
        assert!(
            (*a - *b).abs() < eps,
            "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
            *a,
            *b,
            eps,
            (*a - *b).abs()
        );
    }};
    ($a:expr, $b:expr, $eps:expr) => {{
        let (a, b) = (&$a, &$b);
        let eps = $eps;
        assert!(
            (*a - *b).abs() < eps,
            "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
            *a,
            *b,
            eps,
            (*a - *b).abs()
        );
    }};
}

fn get_print_input_ssbo(msg: &str, buffer: GLuint) -> Vec<f32> {
    unsafe {
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, buffer);
        let input_data =
            gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *const InputData;

        println!("{} {:?}", msg, *input_data);

        gl::UnmapBuffer(gl::SHADER_STORAGE_BUFFER);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);

        return (*input_data).data.to_vec();
    }
}

fn get_print_output_ssbo(msg: &str, buffer: GLuint) -> Vec<f32> {
    unsafe {
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, buffer);
        let output_data =
            gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *const OutputData;

        println!("{} {:?}", msg, *output_data);

        gl::UnmapBuffer(gl::SHADER_STORAGE_BUFFER);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
        return (*output_data).data.to_vec();
    }
}

fn main() {
    // ************************************************************************
    // setup window and opengl context
    use glfw::Context;

    let glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    let (mut window, _) = glfw
        .create_window(300, 300, "Hello this is window", glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window.");
    window.make_current();

    gl::load_with(|s| window.get_proc_address(s) as *const _);
    unsafe {
        gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
        gl::DebugMessageCallback(Some(debug_message_callback::callback), std::ptr::null())
    }

    // ************************************************************************
    // load compute shader
    let prefix_sum_cs = {
        let basic_compute_shader = shader::Shader::from_source(
            &CString::new(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/compaction.comp"
            )))
            .unwrap(),
            gl::COMPUTE_SHADER,
        )
        .unwrap();

        program::Program::new(vec![(basic_compute_shader, gl::COMPUTE_SHADER)]).unwrap()
    };

    // Generate data and run algorithm on CPU to get expected value
    let data: [f32; DATA_LEN] = [
        0.7975555, 0.8064009, 0.3653794, 0.23632169, 0.5929925, 0.4024241, 0.20343924, 0.7010438,
    ];
    let expected = {
        let data_copy = data
            .to_vec()
            .iter()
            .cloned()
            .filter(|x| *x > 0.3)
            .collect::<Vec<_>>();
        data_copy
    };

    let input_data = InputData { data };

    // ************************************************************************
    // create input, output SSBOs and load input data into input SSBO
    {
        // https://github.com/rust-lang/rust/issues/46043
        let idc = input_data.data;
        assert_eq!(idc.len(), DATA_LEN);
    }

    let mut input_ssbo = 0;
    let input_index_binding_point = 0;
    unsafe {
        gl::GenBuffers(1, &mut input_ssbo);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, input_ssbo);
        gl::BufferData(
            gl::SHADER_STORAGE_BUFFER,
            (std::mem::size_of::<InputData>()) as GLsizeiptr,
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

    // ************************************************************************
    // Run compute shader
    get_print_input_ssbo("before:", input_ssbo);
    get_print_output_ssbo("before:", output_ssbo);

    // Perform reduce step on a single block
    prefix_sum_cs.use_();
    unsafe {
        gl::DispatchCompute(1, 1, 1);
        gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
    };

    // https://computergraphics.stackexchange.com/questions/400/synchronizing-successive-opengl-compute-shader-invocations
    // https://gamedev.stackexchange.com/questions/151563/synchronization-between-several-gldispatchcompute-with-same-ssbos

    get_print_input_ssbo("after:", input_ssbo);
    let res = get_print_output_ssbo("after:", output_ssbo);

    println!("expected: {:?}", expected);
    {
        for (l, r) in expected.iter().zip(res) {
            assert_approx_eq!(l, r);
        }
    }
}

// print_output_ssbo("after:", output_ssbo);
