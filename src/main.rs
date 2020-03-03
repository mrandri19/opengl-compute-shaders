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

const DATA_LEN: usize = 31;
const WORK_GROUPS: GLuint = 2;

#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
struct InputData {
    data: [GLfloat; DATA_LEN],
}
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
struct OutputData {
    sums: [GLfloat; WORK_GROUPS as usize],
    data: [GLfloat; DATA_LEN],
}

fn inplace_exclusive_prefix_sum(a: &mut [GLfloat; WORK_GROUPS as usize]) {
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

unsafe fn get_ssbo<T: Clone>(buffer: GLuint) -> T {
    gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, buffer);
    let data = gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *const T;

    let copy = (*data).clone();

    gl::UnmapBuffer(gl::SHADER_STORAGE_BUFFER);
    gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);

    copy
}

fn make_shader_src(src: &str) -> String {
    let mut shader = ShaderStage::parse(src).unwrap();

    let mut transformed_source = String::new();
    struct MyVisitor {}
    impl Visitor for MyVisitor {
        fn visit_preprocessor_define(&mut self, define: &mut PreprocessorDefine) -> Visit {
            match define {
                PreprocessorDefine::ObjectLike { ident, value } => {
                    if ident.as_str() == "DATA_LEN" {
                        *value = DATA_LEN.to_string();
                    }
                    if ident.as_str() == "WORK_GROUPS" {
                        *value = WORK_GROUPS.to_string();
                    }
                }
                _ => (),
            };

            Visit::Parent
        }
    }

    let mut my_visitor = MyVisitor {};
    shader.visit(&mut my_visitor);

    glsl::transpiler::glsl::show_translation_unit(&mut transformed_source, &shader);

    transformed_source
}

fn load_shader(src: &str) -> Program {
    {
        match shader::Shader::from_source(
            &CString::new(make_shader_src(src)).unwrap(),
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

fn prefix_sum(data: [GLfloat; DATA_LEN]) -> [GLfloat; DATA_LEN] {
    let mut a = [0.; DATA_LEN];
    let mut v = data.to_vec();
    v.insert(0, 0.0);
    for i in 1..v.len() {
        v[i] += v[i - 1];
    }
    v.pop();
    for i in 0..v.len() {
        a[i] = v[i];
    }
    a
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
    // load compute shaders
    let multi_wg_prefix_sum1_cs = load_shader(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shaders/multi_wg_prefix_sum1.comp"
    )));

    let multi_wg_prefix_sum2_cs = load_shader(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shaders/multi_wg_prefix_sum2.comp"
    )));

    // ************************************************************************
    // create input data
    let mut data = [0.; DATA_LEN];
    for i in 0..data.len() {
        data[i] = i as GLfloat;
    }

    // TODO(Andrea): finish
    // if !DATA_LEN.is_power_of_two() {
    //     let padding_len = DATA_LEN.next_power_of_two() - DATA_LEN;
    // }

    let input_data = InputData { data };

    // ************************************************************************
    // create input, output SSBOs and load input data into input SSBO

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
    // Run compute shaders
    unsafe { println!("before: {:#?}", get_ssbo::<InputData>(input_ssbo)) };

    multi_wg_prefix_sum1_cs.use_();
    unsafe {
        gl::DispatchCompute(WORK_GROUPS, 1, 1);
        gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
    };

    unsafe {
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, output_ssbo);
        let data = gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *mut OutputData;

        inplace_exclusive_prefix_sum(&mut (*data).sums);

        gl::UnmapBuffer(gl::SHADER_STORAGE_BUFFER);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
    }

    multi_wg_prefix_sum2_cs.use_();
    unsafe {
        gl::DispatchCompute(WORK_GROUPS, 1, 1);
        gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
    };

    unsafe {
        let output_data = get_ssbo::<OutputData>(output_ssbo);
        println!("\nafter: {:#?}", output_data);

        assert_eq!(output_data.data, prefix_sum(data));
    }
}
