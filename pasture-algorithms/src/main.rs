use pasture_core::containers::{InterleavedVecPointStorage, PointBufferExt};
use pasture_core::gpu;
use pasture_core::layout::{
    attributes, PointAttributeDataType, PointAttributeDefinition, PointType,
};
use pasture_core::nalgebra::Vector3;
use pasture_derive::PointType;
use pasture_io::base::PointReader;
use pasture_io::las::LASReader;
use pasture_io::las::LasPointFormat0;

use pasture_algorithms::gpu::gpu_kdtree::{GpuKdTreeInterleaved, IndexedPointType};

use anyhow::Result;

pub fn main() {
    futures::executor::block_on(run());
}

async fn run() -> Result<()> {
    let mut reader = LASReader::from_path(
        //"/home/jnoice/dev/pasture/pasture-io/examples/in/10_points_format_1.las",
        "/home/jnoice/Downloads/WSV_Pointcloud_Tile-3-1.laz",
    )?;
    let count = reader.remaining_points();
    let mut buffer = InterleavedVecPointStorage::with_capacity(count, LasPointFormat0::layout());
    reader.read_into(&mut buffer, count)?;

    for point in buffer.iter_point::<LasPointFormat0>().take(5) {
        println!("{:?}", point.position);
    }

    let mut kdtree = GpuKdTreeInterleaved::new();
    kdtree.initialize(&buffer);

    println!("HIHIIH");

    for point in kdtree
        .point_buffer
        .unwrap()
        .iter_point::<IndexedPointType>()
    {
        println!("{:?}", point);
    }

    Ok(())
}
