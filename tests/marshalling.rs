//! Contains all tests that cover marshalling types to and from C++

use std::mem::ManuallyDrop;

use opencv::core::{CommandLineParser, KeyPoint, Point2f, Ptr, Scalar, SparseMat, Tuple};
use opencv::prelude::*;
use opencv::{core, Result};

/// Passing simple struct as argument
#[test]
fn simple_struct_arg() -> Result<()> {
	#![cfg(ocvrs_has_module_imgproc)]

	use opencv::core::{Point, Size};
	use opencv::imgproc;

	let res = imgproc::get_structuring_element(imgproc::MORPH_CROSS, Size::new(100, 100), Point::new(50, 50))?;
	assert_eq!(res.typ(), 0);
	let size = res.size()?;
	assert_eq!(size.width, 100);
	assert_eq!(size.height, 100);
	Ok(())
}

#[test]
fn scalar_arg() -> Result<()> {
	let mut m = Mat::new_rows_cols_with_default(1, 3, u8::opencv_type(), 2.into())?;
	let sum = core::sum_elems(&m)?;
	assert_eq!(sum[0], 6.);
	let s = m.at_row_mut::<u8>(0)?;
	s[0] = 1;
	s[1] = 2;
	s[2] = 3;
	let sum = core::sum_elems(&m)?;
	assert_eq!(sum[0], 6.);
	Ok(())
}

/// Return of the primitive types and passing them as arguments
#[test]
fn primitives_return() -> Result<()> {
	assert!(core::get_tick_count()? > 10000);
	assert!(core::get_cpu_tick_count()? > 10000);
	assert!(core::get_tick_frequency()? > 1000.);
	assert!(core::get_number_of_cpus()? >= 1);
	core::set_use_optimized(true)?;
	assert!(core::use_optimized()?);
	core::set_use_optimized(false)?;
	assert!(!core::use_optimized()?);
	Ok(())
}

/// Passing callback and checking that it's called
#[test]
fn callback() -> Result<()> {
	#![cfg(ocvrs_has_module_highgui)]
	use std::sync::{Arc, Mutex};

	use opencv::{core, highgui, Error};

	// only run under X11 on linux
	if cfg!(target_os = "linux") && (option_env!("DISPLAY").is_some() || option_env!("WAYLAND_DISPLAY").is_some()) {
		{
			if let Err(Error {
				code: core::StsError, ..
			}) = highgui::named_window_def("test_1")
			{
				// means that OpenCV is not built with GUI support, just skip the test
				return Ok(());
			}
			let mut value = 50;
			let cb_value = Arc::new(Mutex::new(0));
			highgui::create_trackbar(
				"test_track_1",
				"test_1",
				Some(&mut value),
				100,
				Some(Box::new({
					let cb_value = cb_value.clone();
					move |s| {
						*cb_value.lock().unwrap() = s;
					}
				})),
			)?;
			assert_eq!(value, 50);
			highgui::set_trackbar_pos("test_track_1", "test_1", 10)?;
			assert_eq!(value, 10);
			assert_eq!(*cb_value.lock().unwrap(), 10);
		}

		{
			highgui::named_window_def("test_2")?;
			let cb_value = Arc::new(Mutex::new(0));
			highgui::create_trackbar(
				"test_track_2",
				"test_2",
				None,
				100,
				Some(Box::new({
					let cb_value = cb_value.clone();
					move |s| {
						*cb_value.lock().unwrap() = s;
					}
				})),
			)?;
			highgui::set_trackbar_pos("test_track_2", "test_2", 10)?;
			assert_eq!(*cb_value.lock().unwrap(), 10);
		}
	}
	Ok(())
}

/// Return of fixed array
#[test]
fn fixed_array_return() -> Result<()> {
	// mutable fixed array return and modification
	let m = Mat::new_rows_cols_with_default(5, 3, i32::opencv_type(), 1.into())?;
	let mut mat_step = m.mat_step();
	assert_eq!([12, 4], &mat_step.buf()[0..2]);
	mat_step.buf_mut()[0] = 16;
	mat_step.buf_mut()[1] = 2;
	assert_eq!([16, 2], &mat_step.buf()[0..2]);

	Ok(())
}

/// Sometimes functions return a boxed class by pointer, so they need an additional dereference
#[test]
fn boxed_return_by_pointer() -> Result<()> {
	let mut sm = SparseMat::from_mat(&Mat::new_rows_cols_with_default(5, 3, i32::opencv_type(), 1.into())?)?;
	let hdr = sm.hdr();
	assert_eq!(1, hdr.refcount());
	assert_eq!(2, hdr.dims());
	Ok(())
}

/// Return of an owned String
#[test]
fn string_return() -> Result<()> {
	let info = core::get_build_information()?;
	assert!(info.contains("\nGeneral configuration for OpenCV"));
	Ok(())
}

/// Return String via a mutable argument
#[test]
fn string_out_argument() -> Result<()> {
	#![cfg(not(ocvrs_opencv_branch_34))]
	use matches::assert_matches;
	use opencv::core::{FileNode, FileStorage, FileStorage_Mode};
	use opencv::Error;

	{
		let mut st = FileStorage::new_def(
			".yml",
			i32::from(FileStorage_Mode::WRITE) | i32::from(FileStorage_Mode::MEMORY),
		)?;
		st.write_str("test_str", "test string")?;
		let serialized = st.release_and_get_string()?;
		let st = FileStorage::new_def(&serialized, i32::from(FileStorage_Mode::MEMORY))?;
		let str_node = st.get("test_str")?;
		let mut str_out = String::new();
		core::read_str(&str_node, &mut str_out, "default string")?;
		assert_eq!("test string", str_out);
	}

	// correctly handle output string on error condition
	{
		let st = FileStorage::new_def("", i32::from(FileStorage_Mode::WRITE) | i32::from(FileStorage_Mode::MEMORY))?;
		let node = FileNode::new(&st, 0, 0)?;
		let mut out = String::new();
		assert_matches!(
			core::read_str(&node, &mut out, "123"),
			Err(Error {
				code: core::StsAssert,
				..
			})
		);
		assert_eq!("", out);
	}

	Ok(())
}

/// Test whether the return of the small simple structures works as intended
#[test]
fn simple_struct_return_infallible() -> Result<()> {
	#![cfg(ocvrs_has_module_imgproc)]

	use opencv::core::{Point2f, Rect, Size2f, Vector};
	use opencv::imgproc;
	/*
	There previously was an issue that return of the small simple structs like Point2f (2 floats, 64 bits in total) was handled
	differently by Rust (v1.57.0) and default compilers in Ubuntu 20.04 (Gcc 9.3.0 & Clang 10.0.0). That mismatch led to
	miscellaneous memory problems like segmentation faults. That shouldn't happen anymore because such returns are now handled by
	the output argument.
	 */
	let contour = Vector::<Point2f>::from_iter([
		Point2f::new(5., 5.),
		Point2f::new(5., 15.),
		Point2f::new(15., 15.),
		Point2f::new(15., 5.),
		Point2f::new(5., 5.),
	]);
	let bound_rect = imgproc::bounding_rect(&contour)?;
	assert_eq!(Rect::new(5, 5, 11, 11), bound_rect);
	let min_area_rect = imgproc::min_area_rect(&contour)?;
	assert_eq!(Point2f::new(10., 10.), min_area_rect.center);
	assert_eq!(Size2f::new(10., 10.), min_area_rect.size);
	// different versions of OpenCV return -90 and 90
	assert_eq!(90., min_area_rect.angle.abs());
	Ok(())
}

#[test]
fn tuple() -> Result<()> {
	#[cfg(all(ocvrs_has_module_imgproc, not(ocvrs_opencv_branch_34)))]
	{
		let src_tuple = (10, 20.);
		let tuple = Tuple::<(i32, f32)>::new(src_tuple);
		assert_eq!(10, tuple.get_0());
		assert_eq!(20., tuple.get_1());
		assert_eq!(src_tuple, tuple.into_tuple());
	}

	#[cfg(ocvrs_has_module_objdetect)]
	{
		use opencv::core::Rect;

		let src_tuple = (Rect::new(1, 2, 3, 4), 98);
		let tuple = Tuple::<(Rect, i32)>::new(src_tuple);
		assert_eq!(Rect::new(1, 2, 3, 4), tuple.get_0());
		assert_eq!(98, tuple.get_1());
		assert_eq!(src_tuple, tuple.into_tuple());
	}

	#[cfg(ocvrs_has_module_stitching)]
	{
		#[cfg(not(ocvrs_opencv_branch_34))]
		use opencv::core::AccessFlag::ACCESS_READ;
		#[cfg(ocvrs_opencv_branch_34)]
		use opencv::core::ACCESS_READ;
		use opencv::core::{UMat, UMatUsageFlags};

		let mat = Mat::new_rows_cols_with_default(10, 20, f64::opencv_type(), Scalar::all(76.))?;
		let src_tuple = (mat.get_umat(ACCESS_READ, UMatUsageFlags::USAGE_DEFAULT)?, 8);
		let tuple = Tuple::<(UMat, u8)>::new(src_tuple);
		assert_eq!(10, tuple.get_0().rows());
		assert_eq!(8, tuple.get_1());
		let (res_umat, res_val) = tuple.into_tuple();
		let res_mat = res_umat.get_mat(ACCESS_READ)?;
		assert_eq!(10, res_mat.rows());
		assert_eq!(20, res_mat.cols());
		assert_eq!(76., *res_mat.at_2d::<f64>(5, 5)?);
		assert_eq!(8, res_val);
	}
	Ok(())
}

// The TrackerSamplerPF_Params no longer exists since OpenCV 4.5.1 and there are no other methods that
// accept typed Mat's, so disable the test for now.
// #[test]
// fn typed_mat() -> Result<()> {
// 	#![cfg(ocvrs_has_module_tracking)]
//
// 	use opencv::{tracking::TrackerSamplerPF_Params};
// 	let mut params = TrackerSamplerPF_Params::default()?;
// 	assert_eq!(&[15., 15., 15., 15.], params.std().data_typed()?);
// 	let mat = Mat::new_rows_cols_with_default(1, 4, f64::opencv_type(), Scalar::all(8.))?.try_into_typed::<f64>()?;
// 	let mat_src = mat.try_clone()?.try_into_typed::<f64>()?;
// 	params.set_std(mat);
// 	let mat = params.std();
// 	assert_eq!(mat.size()?, mat_src.size()?);
// 	assert_eq!(mat.data_typed()?, mat_src.data_typed()?);
// 	Ok(())
// }

#[test]
fn string_array() -> Result<()> {
	let args = ["test", "-a=b"];
	let parser = CommandLineParser::new(&args, "{a | | }")?;
	assert!(parser.has("a")?);
	assert_eq!("b", parser.get_str("a", true)?);
	Ok(())
}

/// Setting and getting fields through Ptr
#[test]
fn field_access_on_ptr() -> Result<()> {
	let mut key_point = KeyPoint::default()?;
	assert_eq!(Point2f::new(0., 0.), key_point.pt());
	key_point.set_pt(Point2f::new(1., 2.));
	let mut key_point_ptr = Ptr::<KeyPoint>::new(key_point);
	assert_eq!(Point2f::new(1., 2.), key_point_ptr.pt());
	key_point_ptr.set_pt(Point2f::new(3., 4.));

	let wrapped_key_point = ManuallyDrop::new(unsafe { KeyPoint::from_raw(key_point_ptr.inner_as_raw_mut()) });
	assert_eq!(Point2f::new(3., 4.), wrapped_key_point.pt());

	Ok(())
}
