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

fn inplace_exclusive_prefix_sum(a: &mut [GLuint]) {
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

unsafe fn get_ssbo_copy(buffer: GLuint, len: usize) -> Vec<GLuint> {
    gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, buffer);

    let ptr = gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *mut GLuint;

    let slice = std::slice::from_raw_parts(ptr, len);
    let copy = slice.clone().to_owned();

    gl::UnmapBuffer(gl::SHADER_STORAGE_BUFFER);
    gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);

    copy
}

fn make_shader_src(
    src: &str,
    n: usize,
    b: usize,
    chunk_size: usize,
    chunk_rows: usize,
    chunk_cols: usize,
) -> String {
    let mut shader = ShaderStage::parse(src).unwrap();

    let mut transformed_source = String::new();
    struct MyVisitor {
        n: usize,
        b: usize,
        chunk_size: usize,
        chunk_rows: usize,
        chunk_cols: usize,
    }
    impl Visitor for MyVisitor {
        fn visit_preprocessor_define(&mut self, define: &mut PreprocessorDefine) -> Visit {
            match define {
                PreprocessorDefine::ObjectLike { ident, value } => {
                    if ident.as_str() == "N" {
                        *value = self.n.to_string();
                    }
                    if ident.as_str() == "B" {
                        *value = self.b.to_string();
                    }
                    if ident.as_str() == "CHUNK_SIZE" {
                        *value = self.chunk_size.to_string();
                    }
                    if ident.as_str() == "CHUNK_ROWS" {
                        *value = self.chunk_rows.to_string();
                    }
                    if ident.as_str() == "CHUNK_COLS" {
                        *value = self.chunk_cols.to_string();
                    }
                }
                _ => (),
            };

            Visit::Parent
        }
    }

    let mut my_visitor = MyVisitor {
        n,
        b,
        chunk_size,
        chunk_rows,
        chunk_cols,
    };
    shader.visit(&mut my_visitor);

    glsl::transpiler::glsl::show_translation_unit(&mut transformed_source, &shader);

    transformed_source
}

fn load_shader(
    src: &str,
    n: usize,
    b: usize,
    chunk_size: usize,
    chunk_rows: usize,
    chunk_cols: usize,
) -> Program {
    {
        match shader::Shader::from_source(
            &CString::new(make_shader_src(
                src, n, b, chunk_size, chunk_rows, chunk_cols,
            ))
            .unwrap(),
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

fn _prefix_sum(data: Vec<GLuint>) -> Vec<GLuint> {
    let mut v = data.clone();
    v.insert(0, 0);
    for i in 1..v.len() {
        v[i] += v[i - 1];
    }
    v.pop();
    v
}

fn print_input_data(input_data: &Vec<GLuint>, _n: usize, chunk_size: usize) {
    println!("************* input_data ************* ");
    println!("uint chunk[CHUNK_SIZE]:");
    for y in 0..8 {
        for x in 0..8 {
            print!("{:2}", &input_data[y * 8 + x]);
        }
        println!();
    }
    println!(
        "uvec4 ray_start:\n{:?}",
        &input_data[chunk_size..(chunk_size + 4)]
    );
}

fn print_output_data(output_data: &Vec<GLuint>, n: usize, _b: usize, n_over_b: usize) {
    println!();
    println!("************* output_data ************* ");
    println!("uint sums[N_OVER_B]:\n{:?}", &output_data[0..n_over_b]);
    println!(
        "uint offsets[N]:\n{:?}",
        &output_data[n_over_b..(n_over_b + n)]
    );
    println!(
        "uint has_hit[N]:\n{:?}",
        &output_data[(n_over_b + n)..(n_over_b + n + n)]
    );
    let length = output_data[(n_over_b + n + n)..(n_over_b + n + n + 1)][0] as usize;
    println!("uint length:\n{:?}", length);
    println!(
        "uvec4 hits[N]:\n{:?}",
        (&output_data[(n_over_b + n + n + 1 * 4)..(n_over_b + n + n + 1 * 4 + 4 * n)])
            .chunks(4)
            .collect::<Vec<_>>()
    );
    println!(
        "uvec4 compact_hits[N]:\n{:?}",
        (&output_data
            [(n_over_b + n + n + 1 * 4 + 4 * n)..(n_over_b + n + n + 1 * 4 + 4 * n + 4 * n)])
            .chunks(4)
            .collect::<Vec<_>>()
    );
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
    // Parameters
    let n: usize = 8; // total number of rays
    let b = 2; // how many rays per workgroup
    let n_over_b = n / b; // how many workgroups

    let chunk_rows = 8;
    let chunk_cols = 8;
    let chunk_size = chunk_rows * chunk_cols;

    // ************************************************************************
    // Create chunk data
    let mut input_data = vec![0; chunk_size];
    for y in 0..4 {
        for x in 0..chunk_cols {
            if x == 7 {
                input_data[y * chunk_cols + x] = 1;
            }
        }
    }

    // ************************************************************************
    // Add ray_start
    input_data.append(&mut vec![0; 4]);

    // ************************************************************************
    // load the compute shaders, setting the length of input and number of work
    // groups
    let kernel_1 = load_shader(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/multi_wg_raycasting/multi_wg_raycasting1.comp"
        )),
        n,
        b,
        chunk_size,
        chunk_rows,
        chunk_cols,
    );

    let kernel_2 = load_shader(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/multi_wg_raycasting/multi_wg_raycasting2.comp"
        )),
        n,
        b,
        chunk_size,
        chunk_rows,
        chunk_cols,
    );

    // ************************************************************************
    // create input, output SSBOs and load input data into input SSBO
    let mut input_ssbo = 0;
    let input_index_binding_point = 0;
    let input_data_size = chunk_size + 4;
    unsafe {
        gl::GenBuffers(1, &mut input_ssbo);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, input_ssbo);

        // layout(std430, binding = 0) coherent buffer InputData {
        //   uint chunk[CHUNK_SIZE];
        //   uvec4 ray_start;
        // }
        // input_data;
        gl::BufferData(
            gl::SHADER_STORAGE_BUFFER,
            (input_data_size * size_of::<GLuint>()) as GLsizeiptr,
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
    // 1 (uint) * 4 because of alignment to the biggest element which is a uvec4 (16B)
    let output_data_size = n_over_b + n + n + 1 * 4 + 4 * n + 4 * n;
    unsafe {
        gl::GenBuffers(1, &mut output_ssbo);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, output_ssbo);

        // layout(std430, binding = 1) coherent buffer OutputData {
        //   uint sums[N_OVER_B];
        //   uint offsets[N];
        //   uint results[N];
        //   uint length;
        //   vec4 hits[N];
        //   bool has_hit[N];
        // }
        // output_data;
        gl::BufferData(
            gl::SHADER_STORAGE_BUFFER,
            (output_data_size * size_of::<GLuint>()) as GLsizeiptr,
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
    unsafe { print_input_data(&get_ssbo_copy(input_ssbo, input_data_size), n, chunk_size) };

    kernel_1.use_();
    unsafe {
        gl::DispatchCompute(n_over_b as GLuint, 1, 1);
        gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
    };

    // TODO: this moves the SSBO from device memory to device-host-shared
    // memory

    // TODO: What about std430 alignment??
    // https://www.khronos.org/registry/OpenGL/specs/gl/glspec45.core.pdf#page=159
    unsafe {
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, output_ssbo);

        let ptr = gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *mut GLuint;
        let slice = std::slice::from_raw_parts_mut(ptr, output_data_size);

        inplace_exclusive_prefix_sum(&mut slice[0..n_over_b]);

        gl::UnmapBuffer(gl::SHADER_STORAGE_BUFFER);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
    }

    kernel_2.use_();
    unsafe {
        gl::DispatchCompute(n_over_b as GLuint, 1, 1);
        gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
    };

    unsafe {
        let output_data = get_ssbo_copy(output_ssbo, output_data_size);
        print_output_data(&output_data, n, b, n_over_b);
    }
}
