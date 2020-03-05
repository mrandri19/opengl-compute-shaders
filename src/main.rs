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

fn make_shader_src(src: &str, n: usize, b: usize) -> String {
    let mut shader = ShaderStage::parse(src).unwrap();

    let mut transformed_source = String::new();
    struct MyVisitor {
        n: usize,
        b: usize,
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
                }
                _ => (),
            };

            Visit::Parent
        }
    }

    let mut my_visitor = MyVisitor { n, b };
    shader.visit(&mut my_visitor);

    glsl::transpiler::glsl::show_translation_unit(&mut transformed_source, &shader);

    transformed_source
}

fn load_shader(src: &str, n: usize, b: usize) -> Program {
    {
        match shader::Shader::from_source(
            &CString::new(make_shader_src(src, n, b)).unwrap(),
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

fn prefix_sum(data: Vec<GLuint>) -> Vec<GLuint> {
    let mut v = data.clone();
    v.insert(0, 0);
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
    let original_n: usize = 100_000;
    let b = 128;

    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng: StdRng = SeedableRng::from_seed([0; 32]);

    let mut original_data: Vec<GLuint> = vec![0; original_n];
    for i in 0..original_data.len() {
        original_data[i] = rng.gen_range(0, 20);
        // original_data[i] = i as GLuint;
    }

    // ************************************************************************
    // calculate the necessary padding and pad the input data accordingly
    let padding_len = {
        if !original_n.is_power_of_two() {
            original_n.next_power_of_two() - original_n
        } else {
            0
        }
    };

    let n = original_n + padding_len;
    let n_over_b = n / b;

    let mut input_data = original_data.clone();
    input_data.append(&mut vec![0; padding_len]);

    // ************************************************************************
    // load the compute shaders, setting the length of input and number of work
    // groups
    let multi_wg_prefix_sum1_cs = load_shader(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/multi_wg_prefix_sum_uint/multi_wg_prefix_sum1.comp"
        )),
        n,
        b,
    );

    let multi_wg_prefix_sum2_cs = load_shader(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/multi_wg_prefix_sum_uint/multi_wg_prefix_sum2.comp"
        )),
        n,
        b,
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
            (n * size_of::<GLuint>()) as GLsizeiptr,
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
    let output_data_size = n_over_b + n + n + 1 + n;
    unsafe {
        gl::GenBuffers(1, &mut output_ssbo);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, output_ssbo);

        // layout(std430, binding = 1) coherent buffer OutputData {
        //   uint sums[N_OVER_B];
        //   uint offsets[N];
        //   uint results[N];
        //   uint length;
        //   uint data[N];
        // }
        // output_data
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
    // unsafe { println!("before: {:?}", get_ssbo_copy(input_ssbo, n)) };

    multi_wg_prefix_sum1_cs.use_();
    unsafe {
        gl::DispatchCompute(n_over_b as GLuint, 1, 1);
        gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
    };

    // TODO(Andrea): this moves the SSBO from device memory to device-host-shared
    // memory

    // TODO(Andrea): What about std430 alignment??
    // https://www.khronos.org/registry/OpenGL/specs/gl/glspec45.core.pdf#page=159
    unsafe {
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, output_ssbo);

        let ptr = gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *mut GLuint;
        let slice = std::slice::from_raw_parts_mut(ptr, output_data_size);

        inplace_exclusive_prefix_sum(&mut slice[0..n_over_b]);

        gl::UnmapBuffer(gl::SHADER_STORAGE_BUFFER);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
    }

    multi_wg_prefix_sum2_cs.use_();
    unsafe {
        gl::DispatchCompute(n_over_b as GLuint, 1, 1);
        gl::MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
    };

    unsafe {
        let output_data = get_ssbo_copy(output_ssbo, output_data_size);

        // println!("\nafter\nsums: {:?}", &output_data[0..n_over_b]);
        // println!("offsets: {:?}", &output_data[n_over_b..(n_over_b + n)]);
        // println!(
        //     "results: {:?}",
        //     &output_data[(n_over_b + n)..(n_over_b + n + n)]
        // );
        let length = output_data[(n_over_b + n + n)..(n_over_b + n + n + 1)][0] as usize;
        // println!("length: {:?}", length);
        // println!(
        //     "data: {:?}",
        //     &output_data[(n_over_b + n + n + 1)..(n_over_b + n + n + 1 + n)]
        // );
        println!(
            "data: {:?}",
            &output_data[(n_over_b + n + n + 1)..(n_over_b + n + n + 1 + length)]
        );

        let s = prefix_sum(
            original_data
                .iter()
                .map(|n| (n % 2 == 0) as GLuint)
                .collect(),
        );

        for i in 0..original_n {
            assert_eq!(output_data[n_over_b + i], s[i]);
        }
    }
}
