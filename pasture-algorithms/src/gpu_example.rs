#[cfg(feature = "gpu")]
mod ex {

    use pasture_core::containers::{InterleavedVecPointStorage, PointBuffer, PointBufferExt};
    use pasture_core::gpu;
    use pasture_core::gpu::GpuPointBufferInterleaved;
    use pasture_core::layout::PointType;
    use pasture_core::layout::{attributes, PointAttributeDataType, PointAttributeDefinition};
    use pasture_core::nalgebra::Vector3;
    use pasture_derive::PointType;
    use anyhow::Result;
    use pasture_io::base::PointReader;
    use pasture_io::las::LASReader;

    #[repr(C)]
    #[derive(PointType, Debug)]
    struct MyPointType {
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
        #[pasture(BUILTIN_COLOR_RGB)]
        pub icolor: Vector3<u16>,
        // #[pasture(attribute = "MyColorF32")]
        // pub fcolor: Vector3<f32>,
        // #[pasture(attribute = "MyVec3U8")]
        // pub byte_vec: Vector3<u8>,
        // #[pasture(BUILTIN_CLASSIFICATION)]
        // pub classification: u8,
        #[pasture(BUILTIN_INTENSITY)]
        pub intensity: u16,
        // #[pasture(BUILTIN_SCAN_ANGLE)]
        // pub scan_angle: i16,
        // #[pasture(BUILTIN_SCAN_DIRECTION_FLAG)]
        // pub scan_dir_flag: bool,
        #[pasture(attribute = "MyInt32")]
        pub my_int: i32,
        // #[pasture(BUILTIN_WAVEFORM_PACKET_SIZE)]
        // pub packet_size: u32,
        // #[pasture(BUILTIN_RETURN_POINT_WAVEFORM_LOCATION)]
        // pub ret_point_loc: f32,
        // #[pasture(BUILTIN_GPS_TIME)]
        // pub gps_time: f64,
    }

    pub fn main() {
        futures::executor::block_on(run());
    }

    async fn run() -> Result<()> {
        // == Init point buffer ======================================================================
        let mut reader = LASReader::from_path("/home/jnoice/Downloads/interesting.las")?;

        let count = reader.remaining_points();

        let layout = MyPointType::layout();
        let mut point_buffer = InterleavedVecPointStorage::with_capacity(count, MyPointType::layout());
        reader.read_into(&mut point_buffer, count)?;

        // let custom_color_attrib =
        //     PointAttributeDefinition::custom("MyColorF32", PointAttributeDataType::Vec3f32);

        // let custom_byte_vec_attrib =
        //     PointAttributeDefinition::custom("MyVec3U8", PointAttributeDataType::Vec3u8);

        let custom_int_attrib =
            PointAttributeDefinition::custom("MyInt32", PointAttributeDataType::I32);

        // == GPU ====================================================================================
        println!("MOARV");
        // Create a device with defaults...
        let device = gpu::Device::default().await;
        let device = match device {
            Ok(d) => d,
            Err(_) => {
                println!("Failed to request device. Aborting.");
                return Ok(());
            }
        };
        device.print_device_info();

        // ... or custom options
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
                println!("Failed to request device. Aborting.");
                return Ok(());
            }
        };

        device.print_device_info();
        device.print_active_features();
        device.print_active_limits();
        println!("\n");

        let attribs = &[
            attributes::POSITION_3D,
            attributes::COLOR_RGB,
            // custom_color_attrib,
            // custom_byte_vec_attrib,
            // attributes::CLASSIFICATION,
            attributes::INTENSITY,
            // attributes::SCAN_ANGLE,
            // attributes::SCAN_DIRECTION_FLAG,
            custom_int_attrib,
            // attributes::WAVEFORM_PACKET_SIZE,
            // attributes::RETURN_POINT_WAVEFORM_LOCATION,
            // attributes::GPS_TIME,
        ];

        let buffer_info_interleaved = gpu::BufferInfoInterleaved {
            attributes: attribs,
            binding: 0,
        };

        let mut gpu_point_buffer = GpuPointBufferInterleaved::new();
        gpu_point_buffer.malloc(count as u64, &buffer_info_interleaved, &mut device.wgpu_device);
        gpu_point_buffer.upload(
            &point_buffer,
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
        device.set_compute_shader_glsl(include_str!("gpu/shaders/interleaved.comp"));
        device.compute(1, 1, 1);

        println!("\n===== COMPUTE =====\n");

        println!("Before:");
        for point in point_buffer.iter_point::<MyPointType>().take(5) {
            println!("{:?}", point);
        }
        println!();

        gpu_point_buffer
            .download_into_interleaved(
                &mut point_buffer,
                0..3,
                &buffer_info_interleaved,
                &device.wgpu_device,
            )
            .await;

        println!("After:");
        for point in point_buffer.iter_point::<MyPointType>().take(5) {
            println!("{:?}", point);
        }
        Ok(())
    }
}

#[cfg(feature = "gpu")]
fn main() {
    ex::main();
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("MOARVINATOR");
}
