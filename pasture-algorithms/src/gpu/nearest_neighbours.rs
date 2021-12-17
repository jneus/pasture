use anyhow::Result;
use pasture_core::{
    containers::{
        InterleavedPointBuffer, InterleavedPointBufferMut, InterleavedVecPointStorage,
        PerAttributePointBuffer, PerAttributeVecPointStorage, PointBuffer, PointBufferExt,
    },
    gpu,
    layout::{attributes, PointAttributeDataType, PointAttributeDefinition, PointType},
    nalgebra::Vector3,
};
use pasture_derive::PointType;
use pasture_io::las::LasPointFormat0;
use pasture_io::{base::PointReader, las::LASReader};

pub async fn find_nearest_neighbour(
    buf: &PointBuffer,
    query_point_buffer: &PointBuffer,
    attribs: &[PointAttributeDefinition],
    return_buffer: &mut InterleavedVecPointStorage,
) -> Result<()> {
    let count = buf.len();

    let device = gpu::Device::new(gpu::DeviceOptions {
        device_power: gpu::DevicePower::High,
        device_backend: gpu::DeviceBackend::Vulkan,
        use_adapter_features: true,
        use_adapter_limits: true,
    })
    .await;
    let mut device = match device {
        Ok(d) => d,
        Err(_) => {
            println!("Failed to request device. Abort...");
            return Ok(());
        }
    };
    device.print_device_info();
    device.print_active_features();
    device.print_active_limits();
    println!("\n");

    let buffer_info_interleaved = gpu::BufferInfoInterleaved {
        attributes: attribs,
        binding: 0,
    };
    // let query_point_buffer_info = gpu::BufferInfoInterleaved {
    //     attributes: &[attributes::POSITION_3D],
    //     binding: 1,
    // };
    // let return_buffer_info = gpu::BufferInfoInterleaved {
    //     attributes: attribs,
    //     binding: 0,
    // };

    let mut gpu_point_buffer = gpu::GpuPointBufferInterleaved::new();
    //let mut gpu_query_point_buffer = gpu::GpuPointBufferInterleaved::new();
    //let mut gpu_return_buffer = gpu::GpuPointBufferInterleaved::new();

    gpu_point_buffer.malloc(
        count as u64,
        &buffer_info_interleaved,
        &mut device.wgpu_device,
    );
    //gpu_query_point_buffer.malloc(1, &query_point_buffer_info, &mut device.wgpu_device);
    //gpu_return_buffer.malloc(1, &return_buffer_info, &mut device.wgpu_device);

    gpu_point_buffer.upload(
        buf,
        0..count,
        &buffer_info_interleaved,
        &mut device.wgpu_device,
        &device.wgpu_queue,
    );
    // gpu_query_point_buffer.upload(
    //     query_point_buffer,
    //     0..1,
    //     &query_point_buffer_info,
    //     &mut device.wgpu_device,
    //     &device.wgpu_queue,
    // );
    // gpu_return_buffer.upload(
    //     return_buffer,
    //     0..1,
    //     &return_buffer_info,
    //     &mut device.wgpu_device,
    //     &device.wgpu_queue,
    // );

    device.set_bind_group(
        0,
        gpu_point_buffer.bind_group_layout.as_ref().unwrap(),
        gpu_point_buffer.bind_group.as_ref().unwrap(),
    );
    // device.set_bind_group(
    //     1,
    //     gpu_query_point_buffer.bind_group_layout.as_ref().unwrap(),
    //     gpu_query_point_buffer.bind_group.as_ref().unwrap(),
    // );

    let mut query_as_bytes: &mut [u8] = &mut [0; 24];
    query_point_buffer.get_raw_point(0, &mut query_as_bytes);
    let (query_point_layout, query_point_bind_group) =
        device.create_uniform_bind_group(&query_as_bytes, 0);

    device.set_bind_group(1, &query_point_layout, &query_point_bind_group);

    device.set_compute_shader_glsl(include_str!("shaders/hello_shader.comp"));
    println!("YES");
    device.compute((count as u32) / 128, 1, 1);

    gpu_point_buffer
        .download_into_interleaved(
            &mut return_buffer,
            0..1,
            &buffer_info_interleaved,
            &device.wgpu_device,
        )
        .await;
    Ok(())
}
