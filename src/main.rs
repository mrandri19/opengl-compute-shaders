use gl::types::*;

use glsl::parser::Parse;
use glsl::syntax::PreprocessorDefine;
use glsl::syntax::ShaderStage;
use glsl::visitor::{Host, Visit, Visitor};

mod debug_message_callback;
mod program;
mod shader;
mod vertex;
use program::Program;

use std::ffi::CString;
use std::mem::size_of;

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

fn inplace_exclusive_prefix_sum(a: &mut [GLfloat]) {
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

unsafe fn get_ssbo_copy(buffer: GLuint, len: usize) -> Vec<GLfloat> {
    gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, buffer);

    let ptr = gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *mut GLfloat;

    let slice = std::slice::from_raw_parts(ptr, len);
    let copy = slice.clone().to_owned();

    gl::UnmapBuffer(gl::SHADER_STORAGE_BUFFER);
    gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);

    copy
}

fn make_shader_src(src: &str, data_len: usize, work_groups: usize) -> String {
    let mut shader = ShaderStage::parse(src).unwrap();

    let mut transformed_source = String::new();
    struct MyVisitor {
        data_len: usize,
        work_groups: usize,
    }
    impl Visitor for MyVisitor {
        fn visit_preprocessor_define(&mut self, define: &mut PreprocessorDefine) -> Visit {
            match define {
                PreprocessorDefine::ObjectLike { ident, value } => {
                    if ident.as_str() == "DATA_LEN" {
                        *value = self.data_len.to_string();
                    }
                    if ident.as_str() == "WORK_GROUPS" {
                        *value = self.work_groups.to_string();
                    }
                }
                _ => (),
            };

            Visit::Parent
        }
    }

    let mut my_visitor = MyVisitor {
        data_len,
        work_groups,
    };
    shader.visit(&mut my_visitor);

    glsl::transpiler::glsl::show_translation_unit(&mut transformed_source, &shader);

    transformed_source
}

fn load_shader(src: &str, data_len: usize, work_groups: usize) -> Program {
    {
        match shader::Shader::from_source(
            &CString::new(make_shader_src(src, data_len, work_groups)).unwrap(),
            gl::COMPUTE_SHADER,
        ) {
            Ok(cs) => Program::new(vec![(cs, gl::COMPUTE_SHADER)]).unwrap(),
            Err(msg) => {
                eprint!("Shader compilation error:\n{}", msg);
                std::process::exit(1)
            }
        }
    }
}

fn prefix_sum(data: Vec<GLfloat>) -> Vec<GLfloat> {
    let mut v = data.clone();
    v.insert(0, 0.0);
    for i in 1..v.len() {
        v[i] += v[i - 1];
    }
    v.pop();
    v
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
    // create input data
    let original_data_len: usize = 100_000;
    let work_groups = 128;

    let mut original_data = vec![0.; original_data_len];
    for i in 0..original_data.len() {
        original_data[i] = rand::random();
    }

    // ************************************************************************
    // calculate the necessary padding and pad the input data accordingly
    let padding_len = {
        if !original_data_len.is_power_of_two() {
            original_data_len.next_power_of_two() - original_data_len
        } else {
            0
        }
    };

    let data_len = original_data_len + padding_len;

    let mut input_data = original_data.clone();
    input_data.append(&mut vec![0.; padding_len]);

    // ************************************************************************
    // load the compute shaders, setting the length of input and number of work
    // groups
    let multi_wg_prefix_sum1_cs = load_shader(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/multi_wg_prefix_sum1.comp"
        )),
        data_len,
        work_groups,
    );

    let multi_wg_prefix_sum2_cs = load_shader(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/multi_wg_prefix_sum2.comp"
        )),
        data_len,
        work_groups,
    );

    // ************************************************************************
    // create input, output SSBOs and load input data into input SSBO
    let mut input_ssbo = 0;
    let input_index_binding_point = 0;
    unsafe {
        gl::GenBuffers(1, &mut input_ssbo);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, input_ssbo);
        gl::BufferData(
            gl::SHADER_STORAGE_BUFFER,
            (data_len * size_of::<GLfloat>()) as GLsizeiptr,
            input_data.as_ptr() as *const std::ffi::c_void,
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
            ((data_len + work_groups) * size_of::<GLfloat>()) as GLsizeiptr,
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
    // Run compute shaders
    // unsafe { println!("before: {:#?}", get_ssbo_copy(input_ssbo, data_len)) };

    multi_wg_prefix_sum1_cs.use_();
    unsafe {
        gl::DispatchCompute(work_groups as GLuint, 1, 1);
        gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
    };

    // TODO(Andrea): this moves the SSBO from device memory to device-host-shared
    // memory
    unsafe {
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, output_ssbo);

        let ptr = gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *mut GLfloat;
        let slice = std::slice::from_raw_parts_mut(ptr, work_groups + data_len);

        inplace_exclusive_prefix_sum(&mut slice[0..work_groups]);

        gl::UnmapBuffer(gl::SHADER_STORAGE_BUFFER);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
    }

    multi_wg_prefix_sum2_cs.use_();
    unsafe {
        gl::DispatchCompute(work_groups as GLuint, 1, 1);
        gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
    };

    unsafe {
        let output_data = get_ssbo_copy(output_ssbo, work_groups + data_len);
        // println!("\nafter: {:#?}", output_data);
        let s = prefix_sum(original_data);
        for i in 0..original_data_len {
            assert_approx_eq!(output_data[work_groups + i], s[i], 3e-1);
        }
    }
}
