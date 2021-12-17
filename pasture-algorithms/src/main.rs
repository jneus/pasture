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

use pasture_algorithms::gpu::find_nearest_neighbour;
use pasture_algorithms::gpu::gpu_kdtree::{GpuKdTreeInterleaved, IndexedPointType};

use anyhow::Result;

pub fn main() {
    futures::executor::block_on(run());
}

#[repr(C)]
#[derive(PointType, Debug)]
pub struct SimplePoint {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
}

async fn run() -> Result<()> {
    let mut reader = LASReader::from_path(
        "/home/jnoice/dev/pasture/pasture-io/examples/in/10_points_format_1.las",
        //"/home/jnoice/Downloads/WSV_Pointcloud_Tile-3-1.laz",
    )?;
    let count = reader.remaining_points();
    let mut buffer = InterleavedVecPointStorage::with_capacity(count, LasPointFormat0::layout());
    reader.read_into(&mut buffer, count)?;

    for point in buffer.iter_point::<LasPointFormat0>().take(5) {
        println!("{:?}", point.position);
    }

    let mut query_point = InterleavedVecPointStorage::with_capacity(1, SimplePoint::layout());
    let pos: Vector3<f64> = Vector3::new(63., 186., 92.);
    query_point.push_point(SimplePoint { position: pos });

    let mut found_point = InterleavedVecPointStorage::new(LasPointFormat0::layout());
    //found_point.push_point(buffer.get_point::<LasPointFormat0>(0));
    let attribs = &[attributes::POSITION_3D, attributes::INTENSITY];

    find_nearest_neighbour(&buffer, &query_point, attribs, &mut found_point).await;

    for point in found_point.iter_point::<LasPointFormat0>() {
        println!("{:?}", point);
    }

    Ok(())
}
