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

pub fn find_nearest_neighbour(
    buf: &PointBuffer,
    attribs: &[PointAttributeDefinition],
    layout: &PointLayout,
) {
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
    let mut gpu_point_buffer = gpu::GpuPointBufferInterleaved::new();
    gpu_point_buffer.malloc(
        count as u64,
        &buffer_info_interleaved,
        &mut device.wgpu_device,
    );
    gpu_point_buffer.upload(
        &buffer,
        0..count,
        &buffer_info_interleaved,
        &mut device.wgpu_device,
        &device.wgpu_queue,
    );

    device.set_bind_group(
        0,
        gpu_point_buffer.bind_group_layout.as_ref().unwrap(),
        gpu_point_buffer.bind_group.as_ref().unwrap(),
    );
    device.set_compute_shader_glsl(include_str!("gpu/shaders/hello_shader.comp"));
    device.compute((count as u32) / 4, 1, 1);

    let mut new_buffer = InterleavedVecPointStorage::new();
    gpu_point_buffer
        .download_into_interleaved(
            &mut buffer,
            0..count,
            &buffer_info_interleaved,
            &device.wgpu_device,
        )
        .await;
}
