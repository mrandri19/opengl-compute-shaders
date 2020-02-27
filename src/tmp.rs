// el.run(move |event, _, control_flow| {
//     *control_flow = ControlFlow::Poll;

//     match event {
//         Event::LoopDestroyed => return,
//         Event::WindowEvent { ref event, .. } => match event {
//             WindowEvent::Resized(logical_size) => {
//                 let dpi_factor = ctx.window().hidpi_factor();
//                 ctx.resize(logical_size.to_physical(dpi_factor));
//             }
//             WindowEvent::RedrawRequested => {

//                 // // ********************************************************

//                 // drawing_program.use_();

//                 // unsafe {
//                 //     gl::ClearColor(0.0, 0.5, 0.7, 1.0);
//                 //     gl::Clear(gl::COLOR_BUFFER_BIT);
//                 // };

//                 // unsafe {
//                 //     let binding_index = 0;
//                 //     let offset = 0;

//                 //     gl::VertexArrayVertexBuffer(
//                 //         triangle_vertices_vao,                  // the name of the vertex array object
//                 //         binding_index,                          // binding index
//                 //         input_ssbo,                             // buffer
//                 //         offset,                                 // offset
//                 //         std::mem::size_of::<Vertex>() as GLint, // stride
//                 //     );

//                 //     // layout (location = 0) in vec3 in_position;
//                 //     let offset = 0;
//                 //     let location = 0;
//                 //     gl::EnableVertexArrayAttrib(triangle_vertices_vao, location);
//                 //     gl::VertexArrayAttribFormat(
//                 //         triangle_vertices_vao,
//                 //         location,
//                 //         3,
//                 //         gl::FLOAT,
//                 //         gl::FALSE,
//                 //         offset,
//                 //     );
//                 //     gl::VertexArrayAttribBinding(
//                 //         triangle_vertices_vao,
//                 //         location,
//                 //         binding_index,
//                 //     );
//                 // }
//                 // unsafe {
//                 //     gl::DrawArrays(gl::TRIANGLES, 0, 3 as GLsizei);
//                 // }

//                 // ctx.swap_buffers().unwrap();
//                 *control_flow = ControlFlow::Exit
//             }
//             WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
//             _ => (),
//         },
//         _ => (),
//     }
// });
