use gl::types::*;

#[derive(Debug, Copy, Clone)]
#[repr(C, packed)]
pub struct Vertex {
    position: [GLfloat; 3],
}
impl Vertex {
    // pub fn new(position: [GLfloat; 3]) -> Self {
    //     Self { position }
    // }
}
