use std::{fs::File, io::BufWriter, io::Seek, io::Write, path::Path};

use anyhow::Result;
use pasture_core::{containers::PointBuffer, layout::PointLayout};

use crate::base::PointWriter;

use super::{path_is_compressed_las_file, RawLASWriter, RawLAZWriter};

enum WriterVariant<T: Write + Seek + Send + 'static> {
    LAS(RawLASWriter<T>),
    LAZ(RawLAZWriter<T>),
}

/// `PointWriter` implementation for LAS/LAZ files.
///
/// *NOTE*: Due to the nature of the LAS file format, this file
/// writer requires manual `flush` calls in order to actually write the LAS/LAZ data. Once you are done
/// writing points, make sure to call `flush` so that the LAS header is updated correctly.
pub struct LASWriter<T: Write + Seek + Send + 'static> {
    writer: WriterVariant<T>,
}

impl<T: Write + Seek + Send + 'static> LASWriter<T> {
    /// Creates a new 'LASWriter` from the given writer and LAS header
    pub fn from_writer_and_header(
        writer: T,
        header: las::Header,
        is_compressed: bool,
    ) -> Result<Self> {
        let raw_writer: WriterVariant<T> = if is_compressed {
            WriterVariant::LAZ(RawLAZWriter::from_write_and_header(writer, header)?)
        } else {
            WriterVariant::LAS(RawLASWriter::from_write_and_header(writer, header)?)
        };
        Ok(Self { writer: raw_writer })
    }

    /// Unwraps with LASWriter, returning the underlying write type `T`. All internal data is flushed before returning
    /// the writer
    pub fn into_inner(self) -> Result<T> {
        match self.writer {
            WriterVariant::LAS(writer) => writer.into_inner(),
            WriterVariant::LAZ(writer) => writer.into_inner(),
        }
    }
}

impl LASWriter<BufWriter<File>> {
    /// Creates a new 'LASWriter` from the given path and LAS header
    pub fn from_path_and_header<P: AsRef<Path>>(path: P, header: las::Header) -> Result<Self> {
        let is_compressed = path_is_compressed_las_file(path.as_ref())?;
        let writer = BufWriter::new(File::create(path)?);
        Self::from_writer_and_header(writer, header, is_compressed)
    }
}

impl<T: Write + Seek + Send + 'static> PointWriter for LASWriter<T> {
    fn write(&mut self, points: &dyn PointBuffer) -> Result<()> {
        match &mut self.writer {
            WriterVariant::LAS(writer) => writer.write(points),
            WriterVariant::LAZ(writer) => writer.write(points),
        }
    }

    fn flush(&mut self) -> Result<()> {
        match &mut self.writer {
            WriterVariant::LAS(writer) => writer.flush(),
            WriterVariant::LAZ(writer) => writer.flush(),
        }
    }

    fn get_default_point_layout(&self) -> &PointLayout {
        match &self.writer {
            WriterVariant::LAS(writer) => writer.get_default_point_layout(),
            WriterVariant::LAZ(writer) => writer.get_default_point_layout(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{io::Cursor, path::PathBuf};

    use las::{point::Format, Builder};
    use pasture_core::{
        containers::InterleavedVecPointStorage, containers::PointBufferExt, layout::PointType,
        nalgebra::Vector3,
    };
    use scopeguard::defer;

    use crate::{
        base::PointReader,
        las::{
            LASReader, LasPointFormat0, LasPointFormat1, LasPointFormat2, LasPointFormat3,
            LasPointFormat4, LasPointFormat5,
        },
    };
    use pasture_derive::PointType;

    use super::*;

    #[repr(C, packed)]
    #[derive(Debug, Clone, Copy, PointType)]
    struct TestPoint {
        #[pasture(BUILTIN_POSITION_3D)]
        pub position: Vector3<f64>,
        #[pasture(BUILTIN_COLOR_RGB)]
        pub color: Vector3<u16>,
    }

    fn get_test_points_custom_format() -> Vec<TestPoint> {
        vec![
            TestPoint {
                position: Vector3::new(1.0, 2.0, 3.0),
                color: Vector3::new(128, 129, 130),
            },
            TestPoint {
                position: Vector3::new(4.0, 5.0, 6.0),
                color: Vector3::new(1024, 1025, 1026),
            },
        ]
    }

    fn get_test_points_las_format_0() -> Vec<LasPointFormat0> {
        vec![
            LasPointFormat0 {
                classification: 1,
                edge_of_flight_line: false,
                intensity: 1,
                number_of_returns: 1,
                point_source_id: 1,
                position: Vector3::new(1.0, 1.0, 1.0),
                return_number: 1,
                scan_angle_rank: 1,
                scan_direction_flag: false,
                user_data: 1,
            },
            LasPointFormat0 {
                classification: 2,
                edge_of_flight_line: true,
                intensity: 2,
                number_of_returns: 2,
                point_source_id: 2,
                position: Vector3::new(2.0, 2.0, 2.0),
                return_number: 2,
                scan_angle_rank: 2,
                scan_direction_flag: true,
                user_data: 2,
            },
        ]
    }

    fn get_test_points_las_format_1() -> Vec<LasPointFormat1> {
        vec![
            LasPointFormat1 {
                classification: 1,
                edge_of_flight_line: false,
                intensity: 1,
                number_of_returns: 1,
                point_source_id: 1,
                position: Vector3::new(1.0, 1.0, 1.0),
                return_number: 1,
                scan_angle_rank: 1,
                scan_direction_flag: false,
                user_data: 1,
                gps_time: 1234.0,
            },
            LasPointFormat1 {
                classification: 2,
                edge_of_flight_line: true,
                intensity: 2,
                number_of_returns: 2,
                point_source_id: 2,
                position: Vector3::new(2.0, 2.0, 2.0),
                return_number: 2,
                scan_angle_rank: 2,
                scan_direction_flag: true,
                user_data: 2,
                gps_time: 5678.0,
            },
        ]
    }

    fn get_test_points_las_format_2() -> Vec<LasPointFormat2> {
        vec![
            LasPointFormat2 {
                classification: 1,
                edge_of_flight_line: false,
                intensity: 1,
                number_of_returns: 1,
                point_source_id: 1,
                position: Vector3::new(1.0, 1.0, 1.0),
                return_number: 1,
                scan_angle_rank: 1,
                scan_direction_flag: false,
                user_data: 1,
                color_rgb: Vector3::new(128, 129, 130),
            },
            LasPointFormat2 {
                classification: 2,
                edge_of_flight_line: true,
                intensity: 2,
                number_of_returns: 2,
                point_source_id: 2,
                position: Vector3::new(2.0, 2.0, 2.0),
                return_number: 2,
                scan_angle_rank: 2,
                scan_direction_flag: true,
                user_data: 2,
                color_rgb: Vector3::new(1024, 1025, 1026),
            },
        ]
    }

    fn get_test_points_las_format_3() -> Vec<LasPointFormat3> {
        vec![
            LasPointFormat3 {
                classification: 1,
                edge_of_flight_line: false,
                intensity: 1,
                number_of_returns: 1,
                point_source_id: 1,
                position: Vector3::new(1.0, 1.0, 1.0),
                return_number: 1,
                scan_angle_rank: 1,
                scan_direction_flag: false,
                user_data: 1,
                color_rgb: Vector3::new(128, 129, 130),
                gps_time: 1234.0,
            },
            LasPointFormat3 {
                classification: 2,
                edge_of_flight_line: true,
                intensity: 2,
                number_of_returns: 2,
                point_source_id: 2,
                position: Vector3::new(2.0, 2.0, 2.0),
                return_number: 2,
                scan_angle_rank: 2,
                scan_direction_flag: true,
                user_data: 2,
                color_rgb: Vector3::new(1024, 1025, 1026),
                gps_time: 5678.0,
            },
        ]
    }

    fn get_test_points_las_format_4() -> Vec<LasPointFormat4> {
        vec![
            LasPointFormat4 {
                classification: 1,
                edge_of_flight_line: false,
                intensity: 1,
                number_of_returns: 1,
                point_source_id: 1,
                position: Vector3::new(1.0, 1.0, 1.0),
                return_number: 1,
                scan_angle_rank: 1,
                scan_direction_flag: false,
                user_data: 1,
                gps_time: 1234.0,
                byte_offset_to_waveform_data: 10,
                return_point_waveform_location: 20.0,
                wave_packet_descriptor_index: 30,
                waveform_packet_size: 40,
                waveform_parameters: Vector3::new(1.0, 2.0, 3.0),
            },
            LasPointFormat4 {
                classification: 2,
                edge_of_flight_line: true,
                intensity: 2,
                number_of_returns: 2,
                point_source_id: 2,
                position: Vector3::new(2.0, 2.0, 2.0),
                return_number: 2,
                scan_angle_rank: 2,
                scan_direction_flag: true,
                user_data: 2,
                gps_time: 5678.0,
                byte_offset_to_waveform_data: 11,
                return_point_waveform_location: 22.0,
                wave_packet_descriptor_index: 33,
                waveform_packet_size: 44,
                waveform_parameters: Vector3::new(4.0, 5.0, 6.0),
            },
        ]
    }

    fn get_test_points_las_format_5() -> Vec<LasPointFormat5> {
        vec![
            LasPointFormat5 {
                classification: 1,
                edge_of_flight_line: false,
                intensity: 1,
                number_of_returns: 1,
                point_source_id: 1,
                position: Vector3::new(1.0, 1.0, 1.0),
                return_number: 1,
                scan_angle_rank: 1,
                scan_direction_flag: false,
                user_data: 1,
                gps_time: 1234.0,
                color_rgb: Vector3::new(128, 129, 130),
                byte_offset_to_waveform_data: 10,
                return_point_waveform_location: 20.0,
                wave_packet_descriptor_index: 30,
                waveform_packet_size: 40,
                waveform_parameters: Vector3::new(1.0, 2.0, 3.0),
            },
            LasPointFormat5 {
                classification: 2,
                edge_of_flight_line: true,
                intensity: 2,
                number_of_returns: 2,
                point_source_id: 2,
                position: Vector3::new(2.0, 2.0, 2.0),
                return_number: 2,
                scan_angle_rank: 2,
                scan_direction_flag: true,
                user_data: 2,
                gps_time: 5678.0,
                color_rgb: Vector3::new(1024, 1025, 1026),
                byte_offset_to_waveform_data: 11,
                return_point_waveform_location: 22.0,
                wave_packet_descriptor_index: 33,
                waveform_packet_size: 44,
                waveform_parameters: Vector3::new(4.0, 5.0, 6.0),
            },
        ]
    }

    fn prepare_point_buffer<T: PointType + Clone>(test_points: &[T]) -> InterleavedVecPointStorage {
        let layout = T::layout();
        let mut source_point_buffer =
            InterleavedVecPointStorage::with_capacity(test_points.len(), layout);

        for point in test_points.iter().cloned() {
            source_point_buffer.push_point(point);
        }

        source_point_buffer
    }

    #[test]
    fn test_write_las_format_0() -> Result<()> {
        let source_points = get_test_points_las_format_0();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_0.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(0)?;

        {
            let mut writer = LASWriter::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
            writer.flush()?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points: Vec<LasPointFormat0> = read_points_buffer.iter_point().collect();

            assert_eq!(read_points, source_points);
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_0_different_layout() -> Result<()> {
        let source_points = get_test_points_custom_format();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_0_different_layout.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(0)?;

        {
            let mut writer = LASWriter::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
            writer.flush()?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points: Vec<LasPointFormat0> = read_points_buffer.iter_point().collect();

            for (source, read) in source_points.iter().zip(read_points.iter()) {
                assert_eq!(
                    { source.position },
                    { read.position },
                    "Position of read point is different than of source point!"
                );
                assert_eq!(
                    0, read.classification,
                    "Classification of read point was not zero!"
                );
                assert_eq!(
                    false, read.edge_of_flight_line,
                    "Edge of flight line of read point was not false!"
                );
                assert_eq!(
                    0,
                    { read.intensity },
                    "Intensity of read point was not zero!"
                );
                assert_eq!(
                    0, read.number_of_returns,
                    "Number of returns of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.point_source_id },
                    "Point source ID of read point was not zero!"
                );
                assert_eq!(
                    0, read.return_number,
                    "Return number of read point was not zero!"
                );
                assert_eq!(
                    0, read.scan_angle_rank,
                    "Scan angle rank of read point was not zero!"
                );
                assert_eq!(
                    false, read.scan_direction_flag,
                    "Scan direction flag of read point was not false!"
                );
                assert_eq!(0, read.user_data, "User data of read point was not zero!");
            }
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_1() -> Result<()> {
        let source_points = get_test_points_las_format_1();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_1.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(1)?;

        {
            let mut writer = LASWriter::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
            writer.flush()?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points: Vec<LasPointFormat1> = read_points_buffer.iter_point().collect();

            assert_eq!(read_points, source_points);
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_1_different_layout() -> Result<()> {
        let source_points = get_test_points_custom_format();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_1_different_layout.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(1)?;

        {
            let mut writer = LASWriter::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
            writer.flush()?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points: Vec<LasPointFormat1> = read_points_buffer.iter_point().collect();

            for (source, read) in source_points.iter().zip(read_points.iter()) {
                assert_eq!(
                    { source.position },
                    { read.position },
                    "Position of read point is different than of source point!"
                );
                assert_eq!(
                    0, read.classification,
                    "Classification of read point was not zero!"
                );
                assert_eq!(
                    false, read.edge_of_flight_line,
                    "Edge of flight line of read point was not false!"
                );
                assert_eq!(
                    0,
                    { read.intensity },
                    "Intensity of read point was not zero!"
                );
                assert_eq!(
                    0, read.number_of_returns,
                    "Number of returns of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.point_source_id },
                    "Point source ID of read point was not zero!"
                );
                assert_eq!(
                    0, read.return_number,
                    "Return number of read point was not zero!"
                );
                assert_eq!(
                    0, read.scan_angle_rank,
                    "Scan angle rank of read point was not zero!"
                );
                assert_eq!(
                    false, read.scan_direction_flag,
                    "Scan direction flag of read point was not false!"
                );
                assert_eq!(0, read.user_data, "User data of read point was not zero!");
                assert_eq!(
                    0.0,
                    { read.gps_time },
                    "GPS time of read point was not zero!"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_2() -> Result<()> {
        let source_points = get_test_points_las_format_2();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_2.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(2)?;

        {
            let mut writer = LASWriter::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
            writer.flush()?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points: Vec<LasPointFormat2> = read_points_buffer.iter_point().collect();

            assert_eq!(read_points, source_points);
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_2_different_layout() -> Result<()> {
        let source_points = get_test_points_custom_format();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_2_different_layout.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(2)?;

        {
            let mut writer = LASWriter::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
            writer.flush()?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points: Vec<LasPointFormat2> = read_points_buffer.iter_point().collect();

            for (source, read) in source_points.iter().zip(read_points.iter()) {
                assert_eq!(
                    { source.position },
                    { read.position },
                    "Position of read point is different than of source point!"
                );
                assert_eq!(
                    { source.color },
                    { read.color_rgb },
                    "Color of read point is different than of source point!"
                );
                assert_eq!(
                    0, read.classification,
                    "Classification of read point was not zero!"
                );
                assert_eq!(
                    false, read.edge_of_flight_line,
                    "Edge of flight line of read point was not false!"
                );
                assert_eq!(
                    0,
                    { read.intensity },
                    "Intensity of read point was not zero!"
                );
                assert_eq!(
                    0, read.number_of_returns,
                    "Number of returns of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.point_source_id },
                    "Point source ID of read point was not zero!"
                );
                assert_eq!(
                    0, read.return_number,
                    "Return number of read point was not zero!"
                );
                assert_eq!(
                    0, read.scan_angle_rank,
                    "Scan angle rank of read point was not zero!"
                );
                assert_eq!(
                    false, read.scan_direction_flag,
                    "Scan direction flag of read point was not false!"
                );
                assert_eq!(0, read.user_data, "User data of read point was not zero!");
            }
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_3() -> Result<()> {
        let source_points = get_test_points_las_format_3();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_3.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(3)?;

        {
            let mut writer = LASWriter::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
            writer.flush()?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points: Vec<LasPointFormat3> = read_points_buffer.iter_point().collect();

            assert_eq!(read_points, source_points);
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_3_different_layout() -> Result<()> {
        let source_points = get_test_points_custom_format();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_3_different_layout.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(3)?;

        {
            let mut writer = LASWriter::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
            writer.flush()?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points: Vec<LasPointFormat3> = read_points_buffer.iter_point().collect();

            for (source, read) in source_points.iter().zip(read_points.iter()) {
                assert_eq!(
                    { source.position },
                    { read.position },
                    "Position of read point is not equal to position of source point!"
                );
                assert_eq!(
                    { source.color },
                    { read.color_rgb },
                    "Color of read point is not equal to color of source point!"
                );
                assert_eq!(
                    0, read.classification,
                    "Classification of read point was not zero!"
                );
                assert_eq!(
                    false, read.edge_of_flight_line,
                    "Edge of flight line of read point was not false!"
                );
                assert_eq!(
                    0,
                    { read.intensity },
                    "Intensity of read point was not zero!"
                );
                assert_eq!(
                    0, read.number_of_returns,
                    "Number of returns of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.point_source_id },
                    "Point source ID of read point was not zero!"
                );
                assert_eq!(
                    0, read.return_number,
                    "Return number of read point was not zero!"
                );
                assert_eq!(
                    0, read.scan_angle_rank,
                    "Scan angle rank of read point was not zero!"
                );
                assert_eq!(
                    false, read.scan_direction_flag,
                    "Scan direction flag of read point was not false!"
                );
                assert_eq!(0, read.user_data, "User data of read point was not zero!");
                assert_eq!(
                    0.0,
                    { read.gps_time },
                    "GPS time of read point was not zero!"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_4() -> Result<()> {
        let source_points = get_test_points_las_format_4();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_4.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(4)?;

        {
            let mut writer = LASWriter::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
            writer.flush()?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points: Vec<LasPointFormat4> = read_points_buffer.iter_point().collect();

            assert_eq!(read_points, source_points);
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_4_different_layout() -> Result<()> {
        let source_points = get_test_points_custom_format();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_4_different_layout.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(4)?;

        {
            let mut writer = LASWriter::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
            writer.flush()?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points: Vec<LasPointFormat4> = read_points_buffer.iter_point().collect();

            for (source, read) in source_points.iter().zip(read_points.iter()) {
                assert_eq!(
                    { source.position },
                    { read.position },
                    "Position of read point is different than of source point!"
                );
                assert_eq!(
                    0, read.classification,
                    "Classification of read point was not zero!"
                );
                assert_eq!(
                    false, read.edge_of_flight_line,
                    "Edge of flight line of read point was not false!"
                );
                assert_eq!(
                    0,
                    { read.intensity },
                    "Intensity of read point was not zero!"
                );
                assert_eq!(
                    0, read.number_of_returns,
                    "Number of returns of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.point_source_id },
                    "Point source ID of read point was not zero!"
                );
                assert_eq!(
                    0, read.return_number,
                    "Return number of read point was not zero!"
                );
                assert_eq!(
                    0, read.scan_angle_rank,
                    "Scan angle rank of read point was not zero!"
                );
                assert_eq!(
                    false, read.scan_direction_flag,
                    "Scan direction flag of read point was not false!"
                );
                assert_eq!(0, read.user_data, "User data of read point was not zero!");
                assert_eq!(
                    0,
                    { read.byte_offset_to_waveform_data },
                    "Byte offset to waveform data of read point was not zero!"
                );
                assert_eq!(
                    0.0,
                    { read.return_point_waveform_location },
                    "Return point waveform location of read point was not zero!"
                );
                assert_eq!(
                    0, read.wave_packet_descriptor_index,
                    "Wave packet descriptor index of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.waveform_packet_size },
                    "Waveform packet size of read point was not zero!"
                );
                assert_eq!(
                    { Vector3::<f32>::new(0.0, 0.0, 0.0) },
                    { read.waveform_parameters },
                    "Waveform parameters of read point were not zero!"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_5() -> Result<()> {
        let source_points = get_test_points_las_format_5();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_5.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(5)?;

        {
            let mut writer = LASWriter::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
            writer.flush()?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points: Vec<LasPointFormat5> = read_points_buffer.iter_point().collect();

            assert_eq!(read_points, source_points);
        }

        Ok(())
    }

    #[test]
    fn test_write_las_format_5_different_layout() -> Result<()> {
        let source_points = get_test_points_custom_format();
        let source_point_buffer = prepare_point_buffer(&source_points);

        //Write, then read, then check for equality

        let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        test_file_path.push("test_write_las_format_5_different_layout.las");

        defer! {
            std::fs::remove_file(&test_file_path).expect("Removing test file failed!");
        }

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(5)?;

        {
            let mut writer = LASWriter::from_path_and_header(
                &test_file_path,
                las_header_builder.into_header().unwrap(),
            )?;
            writer.write(&source_point_buffer)?;
            writer.flush()?;
        }

        {
            let mut reader = LASReader::from_path(&test_file_path)?;
            let read_points_buffer = reader.read(source_points.len())?;
            let read_points: Vec<LasPointFormat5> = read_points_buffer.iter_point().collect();

            for (source, read) in source_points.iter().zip(read_points.iter()) {
                assert_eq!(
                    { source.position },
                    { read.position },
                    "Position of read point is different than of source point!"
                );
                assert_eq!(
                    { source.color },
                    { read.color_rgb },
                    "Colors of read point are different from colors in source point!"
                );
                assert_eq!(
                    0, read.classification,
                    "Classification of read point was not zero!"
                );
                assert_eq!(
                    false, read.edge_of_flight_line,
                    "Edge of flight line of read point was not false!"
                );
                assert_eq!(
                    0,
                    { read.intensity },
                    "Intensity of read point was not zero!"
                );
                assert_eq!(
                    0, read.number_of_returns,
                    "Number of returns of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.point_source_id },
                    "Point source ID of read point was not zero!"
                );
                assert_eq!(
                    0, read.return_number,
                    "Return number of read point was not zero!"
                );
                assert_eq!(
                    0, read.scan_angle_rank,
                    "Scan angle rank of read point was not zero!"
                );
                assert_eq!(
                    false, read.scan_direction_flag,
                    "Scan direction flag of read point was not false!"
                );
                assert_eq!(0, read.user_data, "User data of read point was not zero!");
                assert_eq!(
                    0.0,
                    { read.gps_time },
                    "GPS time of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.byte_offset_to_waveform_data },
                    "Byte offset to waveform data of read point was not zero!"
                );
                assert_eq!(
                    0.0,
                    { read.return_point_waveform_location },
                    "Return point waveform location of read point was not zero!"
                );
                assert_eq!(
                    0, read.wave_packet_descriptor_index,
                    "Wave packet descriptor index of read point was not zero!"
                );
                assert_eq!(
                    0,
                    { read.waveform_packet_size },
                    "Waveform packet size of read point was not zero!"
                );
                assert_eq!(
                    { Vector3::<f32>::new(0.0, 0.0, 0.0) },
                    { read.waveform_parameters },
                    "Waveform parameters of read point were not zero!"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_las_writer_into_inner() -> Result<()> {
        let source_points = get_test_points_las_format_0();
        let source_point_buffer = prepare_point_buffer(&source_points);

        let mut cursor = Cursor::new(Vec::<u8>::new());

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(0)?;

        {
            let mut writer = LASWriter::from_writer_and_header(
                cursor,
                las_header_builder.into_header().unwrap(),
                false,
            )?;
            writer.write(&source_point_buffer)?;

            cursor = writer.into_inner()?;
        }

        // Assert that some bytes have been written. We could assert the exact number, but that might depend on implementation details
        // like padding that we don't really care about
        let vec = cursor.into_inner();
        assert!(vec.len() > 0);

        Ok(())
    }

    #[test]
    fn test_laz_writer_into_inner() -> Result<()> {
        let source_points = get_test_points_las_format_0();
        let source_point_buffer = prepare_point_buffer(&source_points);

        let mut cursor = Cursor::new(Vec::<u8>::new());

        let mut las_header_builder = Builder::from((1, 4));
        las_header_builder.point_format = Format::new(0)?;

        {
            let mut writer = LASWriter::from_writer_and_header(
                cursor,
                las_header_builder.into_header().unwrap(),
                true,
            )?;
            writer.write(&source_point_buffer)?;

            cursor = writer.into_inner()?;
        }

        // Assert that some bytes have been written. We could assert the exact number, but that might depend on implementation details
        // like padding that we don't really care about
        let vec = cursor.into_inner();
        assert!(vec.len() > 0);

        Ok(())
    }
}
