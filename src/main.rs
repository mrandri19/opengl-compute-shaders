use gl::types::*;

// use glutin::event::WindowEvent;
mod debug_message_callback;
mod program;
mod shader;
mod vertex;
// use shader::Shader;
use std::ffi::CString;

const CHUNK_ROWS: usize = 8;
const CHUNK_COLS: usize = 8;
const CHUNK_SIZE: usize = CHUNK_ROWS * CHUNK_COLS;

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct InputData {
    // These need to be 32bit aligned
    chunk: [GLuint; CHUNK_SIZE],
    ray_start: [GLfloat; 4],
    ray_direction: [GLfloat; 4],
}
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
struct OutputData {
    // These need to be 32bit aligned
    hit: [GLfloat; 4],
    has_hit: GLboolean,
}

fn get_print_input_ssbo(msg: &str, buffer: GLuint) {
    unsafe {
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, buffer);
        let input_data =
            gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *const InputData;

        let grid = (*input_data).chunk.to_vec();
        let mut grid_str = String::new();
        for y in 0..8 {
            for x in 0..8 {
                grid_str += &format!("{:2}", grid[CHUNK_COLS * y + x]);
            }
            grid_str += "\n";
        }
        println!(
            "{} InputData\n{}{:?} {:?}",
            msg,
            grid_str,
            (*input_data).ray_start,
            (*input_data).ray_direction
        );

        gl::UnmapBuffer(gl::SHADER_STORAGE_BUFFER);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
    }
}

fn get_print_output_ssbo(msg: &str, buffer: GLuint) {
    unsafe {
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, buffer);
        let output_data =
            gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *const OutputData;

        println!("{} {:?}", msg, *output_data);

        gl::UnmapBuffer(gl::SHADER_STORAGE_BUFFER);
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
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
        match shader::Shader::from_source(
            &CString::new(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/raycasting.comp"
            )))
            .unwrap(),
            gl::COMPUTE_SHADER,
        ) {
            Ok(cs) => program::Program::new(vec![(cs, gl::COMPUTE_SHADER)]).unwrap(),
            Err(msg) => {
                eprint!("Shader compilation error:\n{}", msg);
                std::process::exit(1);
            }
        }
    };

    // Generate data
    let mut chunk = [0; CHUNK_SIZE];
    for y in 0..CHUNK_COLS {
        chunk[CHUNK_COLS * y + CHUNK_COLS - 1] = 1;
        chunk[CHUNK_COLS * y + CHUNK_COLS - 2] = 1;
    }
    let ray_start = [0.5, 1.5, 0., 0.];
    let ray_direction = [2., 1., 0., 0.];
    let input_data = InputData {
        chunk,
        ray_start,
        ray_direction,
    };

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
    println!();

    get_print_input_ssbo("after:", input_ssbo);
    get_print_output_ssbo("after:", output_ssbo);
}
