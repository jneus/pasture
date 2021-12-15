use pasture_core::containers::{InterleavedVecPointStorage, PointBuffer, PointBufferExt, InterleavedPointBufferMut};
use pasture_core::gpu;
use pasture_core::gpu::GpuPointBufferInterleaved;
use pasture_core::layout::PointType;
use pasture_core::layout::{attributes, PointAttributeDataType, PointAttributeDefinition};
use pasture_core::nalgebra::Vector3;
use pasture_derive::PointType;
use pasture_io::base::PointReader;
use pasture_io::las::LASReader;

use anyhow::Result;

#[derive(PointType, Debug)]
#[repr(C)]
struct MyPoint {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}

fn main(){
    futures::executor::block_on(run());
}

async fn run() -> Result<()>  {

    let mut reader = LASReader::from_path("/home/jnoice/Downloads/interesting.las")?;

    let count = reader.remaining_points();
    let mut buffer = InterleavedPointBufferMut::with_capacity(count, MyPoint::layout());
    reader.read_into(&mut buffer, count)?;

    for point in buffer.iter_point::<MyPoint>().take(5) {
        println!("{:?}", point);
    }

    let layout = MyPoint::layout();

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

    let attribs = &[attributes::POSITION_3D, attributes::INTENSITY];

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
    device.compute((count as u32)/4, 1, 1);

    println!("Before:");
    for point in buffer.iter_point::<MyPoint>().take(10) {
        println!("{:?}", point);
    }
    println!();

    gpu_point_buffer
        .download_into_interleaved(
            &mut buffer,
            0..count,
            &buffer_info_interleaved,
            &device.wgpu_device,
        )
        .await;

    println!("After:");
    for point in buffer.iter_point::<MyPoint>().take(10) {
        println!("{:?}", point);
    }

    Ok(())
}
