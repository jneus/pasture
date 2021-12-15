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
use pasture_io::{base::PointReader, las::LASReader};

#[repr(C)]
#[derive(PointType, Debug)]
pub struct IndexedPointType {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(attribute = "PointIndex")]
    pub index: u32,
}

struct GpuKdTreeNode {
    pub left_child: Option<Box<GpuKdTreeNode>>,
    pub right_child: Option<Box<GpuKdTreeNode>>,
    pub id: u32,
    pub dim: u32,
    pub point: IndexedPointType,
    size: u32,
}

struct GpuKdTree {
    root_node: GpuKdTreeNode,
}

pub struct GpuKdTreeInterleaved {
    root_node: Option<GpuKdTreeNode>,
    pub point_buffer: Option<InterleavedVecPointStorage>,
    x_buffer: Vec<(u32, f64)>,
    y_buffer: Vec<(u32, f64)>,
    z_buffer: Vec<(u32, f64)>,
    indices: Vec<u32>,
    size: u32,
}

impl GpuKdTreeInterleaved {
    pub fn new() -> Self {
        GpuKdTreeInterleaved {
            root_node: None,
            point_buffer: None,
            x_buffer: Vec::new(),
            y_buffer: Vec::new(),
            z_buffer: Vec::new(),
            indices: Vec::new(),
            size: 0,
        }
    }

    pub fn initialize(&mut self, point_buffer: &dyn PointBuffer) {
        let indexed_buffer = GpuKdTreeInterleaved::create_indexed_buffer(point_buffer);
        self.create_dim_buffers(&indexed_buffer);

        self.size = indexed_buffer.len() as u32;
        self.point_buffer = Some(indexed_buffer);

        let tree_depth = (self.size as f32).log2().ceil() as u32;
        let mut tree_node = &self.root_node;

        for current_level in 0..tree_depth {
            let mut dim = current_level % 3;
            match dim {
                0 => self.x_buffer.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()),
                1 => self.y_buffer.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()),
                2 => self.z_buffer.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()),
                _ => break,
            }

            self.reorganize_points(dim);
            //self.create_nodes(dim);
        }
    }

    fn reorganize_points(&mut self, dim: u32) {}
    fn create_indexed_buffer(point_buffer: &dyn PointBuffer) -> InterleavedVecPointStorage {
        let point_count = point_buffer.len();
        let mut indexed_buffer = InterleavedVecPointStorage::new(IndexedPointType::layout());

        for index in 0..point_count {
            let x = point_buffer.get_attribute(&attributes::POSITION_3D, index);
            let indexed_point = IndexedPointType {
                position: x,
                index: index as u32,
            };
            indexed_buffer.push_point(indexed_point);
        }
        indexed_buffer
    }

    fn create_dim_buffers(&mut self, point_buffer: &InterleavedVecPointStorage) {
        for point in point_buffer.iter_point::<IndexedPointType>() {
            self.x_buffer.push((point.index, point.position.x));
            self.y_buffer.push((point.index, point.position.y));
            self.z_buffer.push((point.index, point.position.z));
            self.indices.push(point.index);
        }
    }
}
