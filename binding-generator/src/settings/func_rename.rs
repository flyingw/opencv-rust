use std::collections::HashMap;

use crate::SupportedModule;

pub type FuncRename = HashMap<&'static str, &'static str>;

/// map of functions to rename, key is Func.identifier(), value is new name ("+" will be replaced by the old name)
pub fn func_rename_factory(module: SupportedModule) -> FuncRename {
	match module {
		SupportedModule::Aruco => aruco_factory(),
		SupportedModule::Bioinspired => bioinspired_factory(),
		SupportedModule::Calib3d | SupportedModule::Calib | SupportedModule::ThreeD => calib3d_factory(),
		SupportedModule::Core => core_factory(),
		SupportedModule::Dnn => dnn_factory(),
		SupportedModule::Features2d | SupportedModule::Features => features2d_factory(),
		SupportedModule::Hdf => hdf_factory(),
		SupportedModule::HighGui => highgui_factory(),
		SupportedModule::ImgCodecs => imgcodecs_factory(),
		SupportedModule::ImgProc => imgproc_factory(),
		SupportedModule::LineDescriptor => line_descriptor_factory(),
		SupportedModule::Ml => ml_factory(),
		SupportedModule::ObjDetect => objdetect_factory(),
		SupportedModule::Photo => photo_factory(),
		SupportedModule::Stitching => stitching_factory(),
		SupportedModule::SurfaceMatching => surface_matching_factory(),
		SupportedModule::Text => text_factory(),
		SupportedModule::VideoIo => videoio_factory(),
		SupportedModule::VideoStab => videostab_factory(),
		_ => FuncRename::default(),
	}
}

fn aruco_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([("cv_aruco_getPredefinedDictionary_int", "+_i32")]) // 3.4
}

fn bioinspired_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([
		(
			"cv_bioinspired_Retina_create_Size_const_bool_int_const_bool_const_float_const_float",
			"+_ext",
		),
		("cv_bioinspired_Retina_getMagnoRAW_const__OutputArrayR", "+_to"),
		("cv_bioinspired_Retina_getParvoRAW_const__OutputArrayR", "+_to"),
		("cv_bioinspired_Retina_setup_FileStorageR_const_bool", "+_from_storage"),
		("cv_bioinspired_Retina_setup_String_const_bool", "+_from_file"),
		("cv_bioinspired_Retina_write_const_FileStorageR", "+_to_storage"),
		(
			"cv_bioinspired_TransientAreasSegmentationModule_setup_FileStorageR_const_bool",
			"+_from_storage",
		),
		(
			"cv_bioinspired_TransientAreasSegmentationModule_setup_String_const_bool",
			"+_from_file",
		),
		(
			"cv_bioinspired_TransientAreasSegmentationModule_write_const_FileStorageR",
			"+_to_storage",
		),
	])
}

fn calib3d_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([
		("cv_LMSolver_create_const_PtrLCallbackGR_int_double", "+_ext"),
		("cv_findEssentialMat_const__InputArrayR_const__InputArrayR_const__InputArrayR_int_double_double_const__OutputArrayR", "+_matrix"),
		("cv_findFundamentalMat_const__InputArrayR_const__InputArrayR_const__OutputArrayR_int_double_double", "+_mask"),
		("cv_findHomography_const__InputArrayR_const__InputArrayR_int_double_const__OutputArrayR_const_int_const_double", "+_ext"),
		("cv_fisheye_initUndistortRectifyMap_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__InputArrayR_const_SizeR_int_const__OutputArrayR_const__OutputArrayR", "fisheye_+"),
		("cv_fisheye_projectPoints_const__InputArrayR_const__OutputArrayR_const_Affine3dR_const__InputArrayR_const__InputArrayR_double_const__OutputArrayR", "fisheye_+"),
		("cv_fisheye_projectPoints_const__InputArrayR_const__OutputArrayR_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__InputArrayR_double_const__OutputArrayR", "fisheye_+_vec"),
		("cv_fisheye_stereoCalibrate_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__InputOutputArrayR_const__InputOutputArrayR_const__InputOutputArrayR_const__InputOutputArrayR_Size_const__OutputArrayR_const__OutputArrayR_int_TermCriteria", "fisheye_+"),
		("cv_fisheye_stereoRectify_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__InputArrayR_const_SizeR_const__InputArrayR_const__InputArrayR_const__OutputArrayR_const__OutputArrayR_const__OutputArrayR_const__OutputArrayR_const__OutputArrayR_int_const_SizeR_double_double", "fisheye_+"),
		("cv_fisheye_undistortImage_const__InputArrayR_const__OutputArrayR_const__InputArrayR_const__InputArrayR_const__InputArrayR_const_SizeR", "fisheye_+"),
		("cv_fisheye_undistortPoints_const__InputArrayR_const__OutputArrayR_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__InputArrayR", "fisheye_+"),
		("cv_fisheye_undistortPoints_const__InputArrayR_const__OutputArrayR_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__InputArrayR_TermCriteria", "fisheye_+"),
		("cv_fisheye_distortPoints_const__InputArrayR_const__OutputArrayR_const__InputArrayR_const__InputArrayR_double", "fisheye_+"),
		("cv_recoverPose_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__OutputArrayR_const__OutputArrayR_const__InputOutputArrayR", "+_estimated"),
		("cv_recoverPose_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__OutputArrayR_const__OutputArrayR_double_const__InputOutputArrayR_const__OutputArrayR", "+_triangulated"),
		("cv_recoverPose_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__OutputArrayR_const__OutputArrayR_const__OutputArrayR_int_double_double_const__InputOutputArrayR", "+_2_cameras"),
	])
}

fn core_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([
		(
			"cv_Algorithm_write_const_const_PtrLFileStorageGR_const_StringR",
			"+_with_name",
		),
		("cv_AsyncArray_get_const_const__OutputArrayR_double", "+_with_timeout_f64"),
		("cv_AsyncArray_get_const_const__OutputArrayR_int64_t", "+_with_timeout"),
		("cv_AsyncArray_wait_for_const_double", "+_f64"),
		("cv_Cholesky_floatX_size_t_int_floatX_size_t_int", "+_f32"),
		("cv_DMatch_DMatch_int_int_int_float", "new_index"),
		("cv_FileStorage_write_const_StringR_const_MatR", "+_mat"),
		("cv_FileStorage_write_const_StringR_const_StringR", "+_str"),
		("cv_FileStorage_write_const_StringR_const_vectorLStringGR", "+_str_vec"),
		("cv_FileStorage_write_const_StringR_double", "+_f64"),
		("cv_FileStorage_write_const_StringR_int", "+_i32"),
		("cv_KeyPoint_KeyPoint_Point2f_float_float_float_int_int", "+_point"),
		("cv_KeyPoint_KeyPoint_float_float_float_float_float_int_int", "+_coords"),
		(
			"cv_KeyPoint_convert_const_vectorLPoint2fGR_vectorLKeyPointGR_float_float_int_int",
			"+_to",
		),
		("cv_LDA_LDA_const__InputArrayR_const__InputArrayR_int", "+_with_data"),
		("cv_LU_floatX_size_t_int_floatX_size_t_int", "lu_f32"),
		("cv_MatConstIterator_MatConstIterator_const_MatX", "over"),
		("cv_MatConstIterator_MatConstIterator_const_MatX_Point", "with_start"),
		("cv_MatConstIterator_MatConstIterator_const_MatX_const_intX", "+_slice"),
		("cv_MatConstIterator_MatConstIterator_const_MatX_int_int", "with_rows_cols"),
		("cv_MatConstIterator_pos_const_intX", "+_to"),
		("cv_MatConstIterator_seek_const_intX_bool", "+_idx"),
		("cv_MatExpr_MatExpr_const_MatR", "from_mat"),
		("cv_MatExpr_mul_const_const_MatExprR_double", "+_matexpr"),
		("cv_MatExpr_operator___const_const_RangeR_const_RangeR", "rowscols"),
		("cv_MatExpr_operator___const_const_RectR", "roi"),
		("cv_MatExpr_type_const", "typ"),
		("cv_MatOp_add_const_const_MatExprR_const_ScalarR_MatExprR", "+_scalar"),
		("cv_MatOp_divide_const_double_const_MatExprR_MatExprR", "+_f64"),
		("cv_MatOp_multiply_const_const_MatExprR_double_MatExprR", "+_f64"),
		("cv_MatOp_subtract_const_const_ScalarR_const_MatExprR_MatExprR", "+_scalar"),
		("cv_Mat_Mat_Size_int", "+_size"),
		("cv_Mat_Mat_Size_int_const_ScalarR", "+_size_with_default"),
		("cv_Mat_Mat_Size_int_voidX_size_t", "+_size_with_data_unsafe"),
		("cv_Mat_Mat_const_GpuMatR", "from_gpumat"),
		("cv_Mat_Mat_const_MatR_const_RangeR_const_RangeR", "rowscols"),
		("cv_Mat_Mat_const_MatR_const_RectR", "roi"),
		("cv_Mat_Mat_const_MatR_const_vectorLRangeGR", "ranges"),
		("cv_Mat_Mat_const_vectorLintGR_int", "+_nd_vec"),
		("cv_Mat_Mat_const_vectorLintGR_int_const_ScalarR", "+_nd_vec_with_default"),
		(
			"cv_Mat_Mat_const_vectorLintGR_int_voidX_const_size_tX",
			"+_nd_vec_with_data_unsafe",
		),
		("cv_Mat_Mat_int_const_intX_int", "+_nd"),
		("cv_Mat_Mat_int_const_intX_int_const_ScalarR", "+_nd_with_default"),
		("cv_Mat_Mat_int_const_intX_int_voidX_const_size_tX", "+_nd_with_data_unsafe"),
		("cv_Mat_Mat_int_int_int", "+_rows_cols"),
		("cv_Mat_Mat_int_int_int_const_ScalarR", "+_rows_cols_with_default"),
		("cv_Mat_Mat_int_int_int_voidX_size_t", "+_rows_cols_with_data_unsafe"),
		("cv_Mat_colRange_const_int_int", "col_bounds"),
		("cv_Mat_copyTo_const_const__OutputArrayR_const__InputArrayR", "+_masked"),
		("cv_Mat_create_Size_int", "+_size"),
		("cv_Mat_create_const_vectorLintGR_int", "+_nd_vec"),
		("cv_Mat_create_int_const_intX_int", "+_nd"),
		("cv_Mat_create_int_int_int", "+_rows_cols"),
		("cv_Mat_diag_const_MatR", "+_mat"),
		("cv_Mat_eye_Size_int", "+_size"),
		("cv_Mat_getUMat_const_AccessFlag_UMatUsageFlags", "get_umat"),
		("cv_Mat_ones_Size_int", "+_size"),
		("cv_Mat_ones_int_const_intX_int", "+_nd"),
		("cv_Mat_operator___const_Range_Range", "rowscols"),
		("cv_Mat_operator___const_const_RectR", "roi"),
		("cv_Mat_operator___const_const_vectorLRangeGR", "ranges"),
		("cv_Mat_ptr_const_const_intX", "+_nd"),
		("cv_Mat_ptr_const_intX", "+_nd_mut"),
		("cv_Mat_ptr_const_int_int", "+_2d"),
		("cv_Mat_ptr_const_int_int_int", "+_3d"),
		("cv_Mat_ptr_int", "+_mut"),
		("cv_Mat_ptr_int_int", "+_2d_mut"),
		("cv_Mat_ptr_int_int_int", "+_3d_mut"),
		("cv_Mat_reshape_const_int_const_vectorLintGR", "+_nd_vec"),
		("cv_Mat_reshape_const_int_int_const_intX", "+_nd"),
		("cv_Mat_resize_size_t_const_ScalarR", "+_with_default"),
		("cv_Mat_rowRange_const_int_int", "row_bounds"),
		("cv_Mat_total_const_int_int", "total_slice"),
		("cv_Mat_type_const", "typ"),
		("cv_Mat_zeros_Size_int", "+_size"),
		("cv_Mat_zeros_int_const_intX_int", "+_nd"),
		(
			"cv_PCACompute_const__InputArrayR_const__InputOutputArrayR_const__OutputArrayR_const__OutputArrayR_double",
			"+_variance",
		),
		(
			"cv_PCACompute_const__InputArrayR_const__InputOutputArrayR_const__OutputArrayR_double",
			"+_variance",
		),
		(
			"cv_PCA_PCA_const__InputArrayR_const__InputArrayR_int_double",
			"+_with_variance",
		),
		("cv_PCA_backProject_const_const__InputArrayR_const__OutputArrayR", "+_to"),
		("cv_PCA_project_const_const__InputArrayR_const__OutputArrayR", "+_to"),
		("cv_RNG_MT19937_operator___unsigned_int", "to_u32_with_max"),
		("cv_RNG_operator___unsigned_int", "+_range"),
		("cv_RNG_uniform_double_double", "+_f64"),
		("cv_RNG_uniform_float_float", "+_f32"),
		("cv_Range_Range_int_int", "new"),
		(
			"cv_RotatedRect_RotatedRect_const_Point2fR_const_Point2fR_const_Point2fR",
			"for_points",
		),
		("cv_RotatedRect_points_const_vectorLPoint2fGR", "+_vec"),
		(
			"cv_SVD_backSubst_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__OutputArrayR",
			"+_multi",
		),
		(
			"cv_SVD_compute_const__InputArrayR_const__OutputArrayR_const__OutputArrayR_const__OutputArrayR_int",
			"+_ext",
		),
		("cv_SparseMat_SparseMat_const_MatR", "from_mat"),
		("cv_SparseMat_begin", "+_mut"),
		("cv_SparseMat_copyTo_const_MatR", "+_mat"),
		("cv_SparseMat_end", "+_mut"),
		("cv_UMat_UMat_Size_int_UMatUsageFlags", "+_size"),
		("cv_UMat_UMat_Size_int_const_ScalarR_UMatUsageFlags", "+_size_with_default"),
		("cv_UMat_UMat_const_UMatR_const_RangeR_const_RangeR", "rowscols"),
		("cv_UMat_UMat_const_UMatR_const_RectR", "roi"),
		("cv_UMat_UMat_const_UMatR_const_vectorLRangeGR", "ranges"),
		("cv_UMat_UMat_int_const_intX_int_UMatUsageFlags", "+_nd"),
		(
			"cv_UMat_UMat_int_const_intX_int_const_ScalarR_UMatUsageFlags",
			"+_nd_with_default",
		),
		("cv_UMat_UMat_int_int_int_UMatUsageFlags", "+_rows_cols"),
		(
			"cv_UMat_UMat_int_int_int_const_ScalarR_UMatUsageFlags",
			"+_rows_cols_with_default",
		),
		("cv_UMat_colRange_const_int_int", "col_bounds"),
		("cv_UMat_copyTo_const_const__OutputArrayR_const__InputArrayR", "+_masked"),
		("cv_UMat_create_Size_int_UMatUsageFlags", "+_size"),
		("cv_UMat_create_const_vectorLintGR_int_UMatUsageFlags", "+_nd_vec"),
		("cv_UMat_create_int_const_intX_int_UMatUsageFlags", "+_nd"),
		("cv_UMat_create_int_int_int_UMatUsageFlags", "+_rows_cols"),
		("cv_UMat_diag_const_UMatR_UMatUsageFlags", "+_flags"),
		("cv_UMat_eye_Size_int", "+_size"),
		("cv_UMat_eye_Size_int_UMatUsageFlags", "+_size_flags"),
		("cv_UMat_eye_int_int_int_UMatUsageFlags", "+_flags"),
		("cv_UMat_ones_Size_int", "+_size"),
		("cv_UMat_ones_Size_int_UMatUsageFlags", "+_size_flags"),
		("cv_UMat_ones_int_const_intX_int", "+_nd"),
		("cv_UMat_ones_int_const_intX_int_UMatUsageFlags", "+_nd_flags"),
		("cv_UMat_ones_int_int_int_UMatUsageFlags", "+_flags"),
		("cv_UMat_operator___const_Range_Range", "rowscols"),
		("cv_UMat_operator___const_const_RectR", "roi"),
		("cv_UMat_operator___const_const_vectorLRangeGR", "ranges"),
		("cv_UMat_reshape_const_int_int_const_intX", "+_nd"),
		("cv_UMat_rowRange_const_int_int", "row_bounds"),
		("cv_UMat_type_const", "typ"),
		("cv_UMat_zeros_Size_int", "+_size"),
		("cv_UMat_zeros_Size_int_UMatUsageFlags", "+_size_flags"),
		("cv_UMat_zeros_int_const_intX_int", "+_nd"),
		("cv_UMat_zeros_int_const_intX_int_UMatUsageFlags", "+_nd_flags"),
		("cv_UMat_zeros_int_int_int_UMatUsageFlags", "+_flags"),
		("cv__InputArray__InputArray_const_BufferR", "from_opengl"),
		("cv__InputArray__InputArray_const_GpuMatR", "from_gpumat"),
		("cv__InputArray__InputArray_const_HostMemR", "from_hostmem"),
		("cv__InputArray__InputArray_const_MatExprR", "from_matexpr"),
		("cv__InputArray__InputArray_const_MatR", "from_mat"),
		("cv__InputArray__InputArray_const_UMatR", "from_umat"),
		("cv__InputArray__InputArray_const_doubleR", "from_f64"),
		("cv__InputArray__InputArray_const_vectorLGpuMatGR", "from_gpumat_vec"),
		("cv__InputArray__InputArray_const_vectorLMatGR", "from_mat_vec"),
		("cv__InputArray__InputArray_const_vectorLUMatGR", "from_umat_vec"),
		("cv__InputArray__InputArray_const_vectorLboolGR", "from_bool_vec"),
		(
			"cv__InputArray_copyTo_const_const__OutputArrayR_const__InputArrayR",
			"+_masked",
		),
		("cv__InputOutputArray__InputOutputArray_BufferR", "from_opengl_mut"),
		("cv__InputOutputArray__InputOutputArray_GpuMatR", "from_gpumat_mut"),
		("cv__InputOutputArray__InputOutputArray_HostMemR", "from_hostmem_mut"),
		("cv__InputOutputArray__InputOutputArray_MatR", "from_mat_mut"),
		("cv__InputOutputArray__InputOutputArray_UMatR", "from_umat_mut"),
		("cv__InputOutputArray__InputOutputArray_const_BufferR", "from_opengl"),
		("cv__InputOutputArray__InputOutputArray_const_GpuMatR", "from_gpumat"),
		("cv__InputOutputArray__InputOutputArray_const_HostMemR", "from_hostmem"),
		("cv__InputOutputArray__InputOutputArray_const_MatR", "from_mat"),
		("cv__InputOutputArray__InputOutputArray_const_UMatR", "from_umat"),
		(
			"cv__InputOutputArray__InputOutputArray_const_vectorLGpuMatGR",
			"from_gpumat_vec",
		),
		("cv__InputOutputArray__InputOutputArray_const_vectorLMatGR", "from_mat_vec"),
		("cv__InputOutputArray__InputOutputArray_const_vectorLUMatGR", "from_umat_vec"),
		("cv__InputOutputArray__InputOutputArray_vectorLMatGR", "from_mat_vec_mut"),
		("cv__InputOutputArray__InputOutputArray_vectorLUMatGR", "from_umat_vec_mut"),
		("cv__OutputArray__OutputArray_BufferR", "from_opengl_mut"),
		("cv__OutputArray__OutputArray_GpuMatR", "from_gpumat_mut"),
		("cv__OutputArray__OutputArray_HostMemR", "from_hostmem_mut"),
		("cv__OutputArray__OutputArray_MatR", "from_mat_mut"),
		("cv__OutputArray__OutputArray_UMatR", "from_umat_mut"),
		("cv__OutputArray__OutputArray_const_BufferR", "from_opengl"),
		("cv__OutputArray__OutputArray_const_GpuMatR", "from_gpumat"),
		("cv__OutputArray__OutputArray_const_HostMemR", "from_hostmem"),
		("cv__OutputArray__OutputArray_const_MatR", "from_mat"),
		("cv__OutputArray__OutputArray_const_UMatR", "from_umat"),
		("cv__OutputArray__OutputArray_const_vectorLMatGR", "from_mat_vec"),
		("cv__OutputArray__OutputArray_const_vectorLUMatGR", "from_umat_vec"),
		("cv__OutputArray__OutputArray_vectorLGpuMatGR", "from_gpumat_vec_mut"),
		("cv__OutputArray__OutputArray_vectorLMatGR", "from_mat_vec_mut"),
		("cv__OutputArray__OutputArray_vectorLUMatGR", "from_umat_vec_mut"),
		("cv__OutputArray_assign_const_const_MatR", "+_mat"),
		("cv__OutputArray_assign_const_const_UMatR", "+_umat"),
		("cv__OutputArray_assign_const_const_vectorLMatGR", "+_mat_vec"),
		("cv__OutputArray_assign_const_const_vectorLUMatGR", "+_umat_vec"),
		("cv__OutputArray_create_const_Size_int_int_bool_DepthMask", "+_size"),
		("cv__OutputArray_create_const_Size_int_int_bool_int", "+_size"), // 3.4
		("cv__OutputArray_create_const_int_const_intX_int_int_bool_DepthMask", "+_nd"),
		("cv__OutputArray_create_const_int_const_intX_int_int_bool_int", "+_nd"), // 3.4
		("cv__OutputArray_move_const_MatR", "+mat"),
		("cv__OutputArray_move_const_UMatR", "+umat"),
		("cv_abs_const_MatExprR", "+_matexpr"),
		("cv_cuda_GpuMatND_operator___const_IndexArray_Range_Range", "rowscols"),
		("cv_cuda_GpuMatND_operator___const_const_vectorLRangeGR", "ranges"),
		("cv_cuda_GpuMat_GpuMat_Size_int_AllocatorX", "+_size"),
		("cv_cuda_GpuMat_GpuMat_Size_int_Scalar_AllocatorX", "+_size_with_default"),
		("cv_cuda_GpuMat_GpuMat_Size_int_voidX_size_t", "+_size_with_data"),
		("cv_cuda_GpuMat_GpuMat_const_GpuMatR_Range_Range", "rowscols"),
		("cv_cuda_GpuMat_GpuMat_const_GpuMatR_Rect", "roi"),
		("cv_cuda_GpuMat_GpuMat_const__InputArrayR_AllocatorX", "from_hostmem"),
		("cv_cuda_GpuMat_GpuMat_int_int_int_AllocatorX", "+_rows_cols"),
		(
			"cv_cuda_GpuMat_GpuMat_int_int_int_Scalar_AllocatorX",
			"+_rows_cols_with_default",
		),
		("cv_cuda_GpuMat_GpuMat_int_int_int_voidX_size_t", "+_rows_cols_with_data"),
		("cv_cuda_GpuMat_colRange_const_int_int", "col_bounds"),
		("cv_cuda_GpuMat_copyTo_const_GpuMatR", "+_gpu_mat"),
		("cv_cuda_GpuMat_copyTo_const_GpuMatR_GpuMatR", "+_gpu_mat_mask"),
		("cv_cuda_GpuMat_copyTo_const_GpuMatR_GpuMatR_StreamR", "+_gpu_mat_mask_stream"),
		("cv_cuda_GpuMat_copyTo_const_GpuMatR_StreamR", "+_gpu_mat_stream"),
		("cv_cuda_GpuMat_copyTo_const_const__OutputArrayR_StreamR", "+_stream"),
		("cv_cuda_GpuMat_copyTo_const_const__OutputArrayR_const__InputArrayR", "+_mask"),
		(
			"cv_cuda_GpuMat_copyTo_const_const__OutputArrayR_const__InputArrayR_StreamR",
			"+_mask_stream",
		),
		("cv_cuda_GpuMat_download_const_const__OutputArrayR_StreamR", "+_async"),
		("cv_cuda_GpuMat_operator___const_Range_Range", "rowscols"),
		("cv_cuda_GpuMat_operator___const_Rect", "roi"),
		("cv_cuda_GpuMat_ptr_int", "+_mut"),
		("cv_cuda_GpuMat_rowRange_const_int_int", "row_bounds"),
		("cv_cuda_GpuMat_upload_const__InputArrayR_StreamR", "+_async"),
		("cv_directx_getTypeFromD3DFORMAT_const_int", "get_type_from_d3d_format"),
		(
			"cv_divide_const__InputArrayR_const__InputArrayR_const__OutputArrayR_double_int",
			"+2",
		),
		("cv_getNumberOfCPUs", "get_number_of_cpus"),
		("cv_hconcat_const__InputArrayR_const__InputArrayR_const__OutputArrayR", "+2"),
		("cv_max_const_MatR_const_MatR", "+_mat"),
		("cv_max_const_MatR_const_MatR_MatR", "+_mat_to"),
		("cv_max_const_MatR_double", "+_mat_f64"),
		("cv_max_const_UMatR_const_UMatR_UMatR", "+_umat_to"),
		("cv_max_double_const_MatR", "+_f64_mat"),
		("cv_minMaxLoc_const_SparseMatR_doubleX_doubleX_intX_intX", "+_sparse"),
		("cv_min_const_MatR_const_MatR", "+_mat"),
		("cv_min_const_MatR_const_MatR_MatR", "+_mat_to"),
		("cv_min_const_MatR_double", "+_mat_f64"),
		("cv_min_const_UMatR_const_UMatR_UMatR", "+_umat_to"),
		("cv_min_double_const_MatR", "+_f64_mat"),
		(
			"cv_mixChannels_const__InputArrayR_const__InputOutputArrayR_const_vectorLintGR",
			"+_vec",
		),
		("cv_norm_const_SparseMatR_int", "+_sparse"),
		("cv_norm_const__InputArrayR_const__InputArrayR_int_const__InputArrayR", "+2"),
		("cv_normalize_const_SparseMatR_SparseMatR_double_int", "+_sparse"),
		("cv_ocl_Context_Context_int", "+_with_type"),
		("cv_ocl_Context_create_int", "+_with_type"),
		(
			"cv_ocl_Kernel_create_const_charX_const_ProgramSourceR_const_StringR_StringX",
			"+_ext",
		),
		("cv_ocl_Kernel_set_int_const_KernelArgR", "+_kernel_arg"),
		("cv_ocl_Kernel_set_int_const_UMatR", "+_umat"),
		("cv_ocl_ProgramSource_ProgramSource_const_StringR", "from_str"),
		("cv_ocl_Program_getPrefix_const_StringR", "+_build_flags"),
		("cv_ogl_Buffer_create_Size_int_Target_bool", "+_size"),
		("cv_read_const_FileNodeR_DMatchR_const_DMatchR", "+_dmatch"),
		("cv_read_const_FileNodeR_KeyPointR_const_KeyPointR", "+_keypoint"),
		("cv_read_const_FileNodeR_MatR_const_MatR", "+_mat"),
		("cv_read_const_FileNodeR_SparseMatR_const_SparseMatR", "+_sparsemat"),
		("cv_read_const_FileNodeR_doubleR_double", "+_f64"),
		("cv_read_const_FileNodeR_floatR_float", "+_f32"),
		("cv_read_const_FileNodeR_intR_int", "+_i32"),
		("cv_read_const_FileNodeR_stringR_const_stringR", "+_str"),
		("cv_read_const_FileNodeR_vectorLDMatchGR", "+_dmatch_vec_legacy"),
		("cv_read_const_FileNodeR_vectorLKeyPointGR", "+_keypoint_vec_legacy"),
		("cv_repeat_const__InputArrayR_int_int_const__OutputArrayR", "+_to"),
		("cv_split_const_MatR_MatX", "+_slice"),
		("cv_swap_UMatR_UMatR", "+_umat"),
		("cv_vconcat_const__InputArrayR_const__InputArrayR_const__OutputArrayR", "+2"),
		("cv_writeScalar_FileStorageR_const_StringR", "+_str"),
		("cv_writeScalar_FileStorageR_double", "+_f64"),
		("cv_writeScalar_FileStorageR_float", "+_f32"),
		("cv_writeScalar_FileStorageR_int", "+_i32"),
		("cv_write_FileStorageR_const_StringR_const_MatR", "+_mat"),
		("cv_write_FileStorageR_const_StringR_const_SparseMatR", "+_sparsemat"),
		("cv_write_FileStorageR_const_StringR_const_StringR", "+_str"),
		("cv_write_FileStorageR_const_StringR_const_vectorLDMatchGR", "+_dmatch_vec"),
		(
			"cv_write_FileStorageR_const_StringR_const_vectorLKeyPointGR",
			"+_keypoint_vec",
		),
		("cv_write_FileStorageR_const_StringR_double", "+_f64"),
		("cv_write_FileStorageR_const_StringR_float", "+_f32"),
		("cv_write_FileStorageR_const_StringR_int", "+_i32"),
	])
}

fn dnn_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([
		("cv_dnn_DictValue_DictValue_bool", "from_bool"),
		("cv_dnn_DictValue_DictValue_const_charX", "from_str"),
		("cv_dnn_DictValue_DictValue_double", "from_f64"),
		("cv_dnn_DictValue_DictValue_int", "from_i32"),
		("cv_dnn_DictValue_DictValue_int64_t", "from_i64"),
		("cv_dnn_DictValue_DictValue_unsigned_int", "from_u32"),
		("cv_dnn_Dict_ptr_const_StringR", "+_mut"),
		("cv_dnn_Layer_finalize_const_vectorLMatGR", "+_mat"),
		("cv_dnn_Layer_finalize_const_vectorLMatGR_vectorLMatGR", "+_mat_to"),
		("cv_dnn_Layer_forward_vectorLMatXGR_vectorLMatGR_vectorLMatGR", "+_mat"),
		(
			"cv_dnn_NMSBoxes_const_vectorLRect2dGR_const_vectorLfloatGR_const_float_const_float_vectorLintGR_const_float_const_int",
			"+_f64",
		),
		(
			"cv_dnn_Net_addLayerToPrev_const_StringR_const_StringR_const_intR_LayerParamsR",
			"+_type",
		),
		(
			"cv_dnn_Net_addLayer_const_StringR_const_StringR_const_intR_LayerParamsR",
			"+_type",
		),
		("cv_dnn_Net_connect_String_String", "+_first_second"),
		("cv_dnn_Net_forward_const_StringR", "+_single"),
		("cv_dnn_Net_forward_const__OutputArrayR_const_StringR", "+_layer"),
		(
			"cv_dnn_Net_getMemoryConsumption_const_const_int_const_vectorLMatShapeGR_size_tR_size_tR",
			"+_for_layer",
		),
		(
			"cv_dnn_Net_getMemoryConsumption_const_const_vectorLMatShapeGR_vectorLintGR_vectorLsize_tGR_vectorLsize_tGR",
			"+_for_layers",
		),
		(
			"cv_dnn_TextDetectionModel_EAST_TextDetectionModel_EAST_const_stringR_const_stringR",
			"from_file",
		),
		(
			"cv_dnn_TextDetectionModel_detect_const_const__InputArrayR_vectorLvectorLPointGGR_vectorLfloatGR",
			"+_with_confidences",
		),
		(
			"cv_dnn_TextRecognitionModel_TextRecognitionModel_const_stringR_const_stringR",
			"from_file",
		),
		(
			"cv_dnn_blobFromImage_const__InputArrayR_const__OutputArrayR_double_const_SizeR_const_ScalarR_bool_bool_int",
			"+_to",
		),
		(
			"cv_dnn_blobFromImages_const__InputArrayR_const__OutputArrayR_double_Size_const_ScalarR_bool_bool_int",
			"+_to",
		),
		("cv_dnn_readNetFromCaffe_const_charX_size_t_const_charX_size_t", "+_str"),
		(
			"cv_dnn_readNetFromCaffe_const_vectorLunsigned_charGR_const_vectorLunsigned_charGR",
			"+_buffer",
		),
		("cv_dnn_readNetFromDarknet_const_charX_size_t_const_charX_size_t", "+_str"),
		(
			"cv_dnn_readNetFromDarknet_const_vectorLunsigned_charGR_const_vectorLunsigned_charGR",
			"+_buffer",
		),
		("cv_dnn_readNetFromONNX_const_charX_size_t", "+_str"),
		("cv_dnn_readNetFromONNX_const_vectorLunsigned_charGR", "+_buffer"),
		("cv_dnn_readNetFromTensorflow_const_charX_size_t_const_charX_size_t", "+_str"),
		(
			"cv_dnn_readNetFromTensorflow_const_vectorLunsigned_charGR_const_vectorLunsigned_charGR",
			"+_buffer",
		),
	])
}

fn features2d_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([
		("cv_AGAST_const__InputArrayR_vectorLKeyPointGR_int_bool_DetectorType", "+_with_type"),
		("cv_AGAST_const__InputArrayR_vectorLKeyPointGR_int_bool_int", "+_with_type"), // 3.x only
		("cv_BOWImgDescriptorExtractor_BOWImgDescriptorExtractor_const_PtrLFeature2DGR_const_PtrLDescriptorMatcherGR", "+_with_extractor"),
		("cv_BOWImgDescriptorExtractor_compute2_const_MatR_vectorLKeyPointGR_MatR", "compute2"),
		("cv_BOWImgDescriptorExtractor_compute_const__InputArrayR_vectorLKeyPointGR_const__OutputArrayR_vectorLvectorLintGGX_MatX", "+_desc"),
		("cv_BOWKMeansTrainer_cluster_const_const_MatR", "+_with_descriptor"),
		("cv_BOWTrainer_cluster_const_const_MatR", "+_with_descriptors"),
		("cv_BRISK_create_const_vectorLfloatGR_const_vectorLintGR_float_float_const_vectorLintGR", "+_with_pattern"),
		("cv_BRISK_create_int_int_const_vectorLfloatGR_const_vectorLintGR_float_float_const_vectorLintGR", "+_with_pattern_threshold_octaves"),
		("cv_DescriptorMatcher_create_const_MatcherTypeR", "+_with_matcher_type"),
		("cv_DescriptorMatcher_create_int", "+_with_matcher_type"), // 3.x only
		("cv_DescriptorMatcher_knnMatch_const_const__InputArrayR_const__InputArrayR_vectorLvectorLDMatchGGR_int_const__InputArrayR_bool", "knn_train_match"),
		("cv_DescriptorMatcher_match_const_const__InputArrayR_const__InputArrayR_vectorLDMatchGR_const__InputArrayR", "train_match"),
		("cv_DescriptorMatcher_radiusMatch_const_const__InputArrayR_const__InputArrayR_vectorLvectorLDMatchGGR_float_const__InputArrayR_bool", "radius_train_match"),
		("cv_DescriptorMatcher_read_const_FileNodeR", "+_from_node"),
		("cv_DescriptorMatcher_write_const_FileStorageR", "+_to_storage"),
		("cv_DescriptorMatcher_write_const_FileStorageR_const_StringR", "+_to_storage_with_name"),
		("cv_DescriptorMatcher_write_const_const_PtrLFileStorageGR_const_StringR", "+_to_storage_ptr_with_name"),
		("cv_FAST_const__InputArrayR_vectorLKeyPointGR_int_bool_DetectorType", "+_with_type"),
		("cv_FAST_const__InputArrayR_vectorLKeyPointGR_int_bool_int", "+_with_type"), // 3.x only
		("cv_Feature2D_compute_const__InputArrayR_vectorLvectorLKeyPointGGR_const__OutputArrayR", "+_multiple"),
		("cv_Feature2D_detect_const__InputArrayR_vectorLvectorLKeyPointGGR_const__InputArrayR", "+_multiple"),
		("cv_Feature2D_read_const_FileNodeR", "+_from_node"),
		("cv_Feature2D_write_const_FileStorageR", "+_to_storage"),
		("cv_Feature2D_write_const_FileStorageR_const_StringR", "+_to_storage_with_name"),
		("cv_Feature2D_write_const_const_PtrLFileStorageGR_const_StringR", "+_to_storage_ptr_with_name"),
		("cv_GFTTDetector_create_int_double_double_int_int_bool_double", "+_with_gradient"),
		("cv_SIFT_create_int_int_double_double_double_int", "+_with_descriptor_type"),
		("cv_SIFT_create_int_int_double_double_double_int_bool", "+_with_descriptor_type"),
		("cv_drawMatches_const__InputArrayR_const_vectorLKeyPointGR_const__InputArrayR_const_vectorLKeyPointGR_const_vectorLDMatchGR_const__InputOutputArrayR_const_int_const_ScalarR_const_ScalarR_const_vectorLcharGR_DrawMatchesFlags", "+_with_thickness"),
		("cv_drawMatches_const__InputArrayR_const_vectorLKeyPointGR_const__InputArrayR_const_vectorLKeyPointGR_const_vectorLDMatchGR_const__InputOutputArrayR_const_int_const_ScalarR_const_ScalarR_const_vectorLcharGR_int", "+_with_thickness"),

	])
}

fn hdf_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([
		("cv_hdf_HDF5_atread_StringX_const_StringR", "+_str"),
		("cv_hdf_HDF5_atread_doubleX_const_StringR", "+_f64"),
		("cv_hdf_HDF5_atread_intX_const_StringR", "+_i32"),
		("cv_hdf_HDF5_atwrite_const_StringR_const_StringR", "+_str"),
		("cv_hdf_HDF5_atwrite_const_double_const_StringR", "+_f64"),
		("cv_hdf_HDF5_atwrite_const_int_const_StringR", "+_i32"),
		(
			"cv_hdf_HDF5_dscreate_const_const_int_const_intX_const_int_const_StringR",
			"+_nd",
		),
		(
			"cv_hdf_HDF5_dscreate_const_const_int_const_intX_const_int_const_StringR_const_int",
			"+_nd_compress",
		),
		(
			"cv_hdf_HDF5_dscreate_const_const_int_const_intX_const_int_const_StringR_const_int_const_intX",
			"+_nd_compress_dims",
		),
		(
			"cv_hdf_HDF5_dscreate_const_const_int_const_int_const_int_const_StringR_const_int",
			"+_compress",
		),
		(
			"cv_hdf_HDF5_dscreate_const_const_int_const_int_const_int_const_StringR_const_int_const_vectorLintGR",
			"+_compress_dims",
		),
		(
			"cv_hdf_HDF5_dscreate_const_const_vectorLintGR_const_int_const_StringR_const_int_const_vectorLintGR",
			"+_nd_vec_compress_dims",
		),
		(
			"cv_hdf_HDF5_dsinsert_const_const__InputArrayR_const_StringR_const_vectorLintGR_const_vectorLintGR",
			"+_offset",
		),
		(
			"cv_hdf_HDF5_dsread_const_const__OutputArrayR_const_StringR_const_vectorLintGR_const_vectorLintGR",
			"+_offset",
		),
		(
			"cv_hdf_HDF5_dswrite_const_const__InputArrayR_const_StringR_const_vectorLintGR_const_vectorLintGR",
			"+_offset",
		),
	])
}

fn highgui_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([
		(
			"cv_addText_const_MatR_const_StringR_Point_const_StringR_int_Scalar_int_int_int",
			"+_with_font",
		),
		("cv_resizeWindow_const_StringR_const_SizeR", "+_size"),
		("cv_selectROI_const_StringR_const__InputArrayR_bool_bool", "+_for_window"),
		(
			"cv_selectROIs_const_StringR_const__InputArrayR_vectorLRectGR_bool_bool",
			"select_rois",
		),
	])
}

fn imgcodecs_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([
		("cv_imdecode_const__InputArrayR_int_MatX", "+_to"),
		("cv_imreadmulti_const_StringR_vectorLMatGR_int_int_int", "+_range"),
	])
}

fn imgproc_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([
		("cv_Canny_const__InputArrayR_const__InputArrayR_const__OutputArrayR_double_double_bool", "+_derivative"),
		("cv_GeneralizedHough_detect_const__InputArrayR_const__InputArrayR_const__InputArrayR_const__OutputArrayR_const__OutputArrayR", "+_with_edges"),
		("cv_Subdiv2D_insert_const_vectorLPoint2fGR", "+_multiple"),
		("cv_applyColorMap_const__InputArrayR_const__OutputArrayR_const__InputArrayR", "+_user"),
		("cv_clipLine_Size2l_Point2lR_Point2lR", "+_size_i64"),
		("cv_clipLine_Size_PointR_PointR", "clip_line_size"),
		("cv_ellipse2Poly_Point2d_Size2d_int_int_int_int_vectorLPoint2dGR", "ellipse_2_poly_f64"),
		("cv_ellipse2Poly_Point_Size_int_int_int_int_vectorLPointGR", "ellipse_2_poly"),
		("cv_ellipse_const__InputOutputArrayR_const_RotatedRectR_const_ScalarR_int_int", "ellipse_rotated_rect"),
		("cv_findContours_const__InputArrayR_const__OutputArrayR_const__OutputArrayR_int_int_Point", "+_with_hierarchy"), // 4.x
		("cv_findContours_const__InputOutputArrayR_const__OutputArrayR_const__OutputArrayR_int_int_Point", "+_with_hierarchy"), // 3.4
		("cv_floodFill_const__InputOutputArrayR_const__InputOutputArrayR_Point_Scalar_RectX_Scalar_Scalar_int", "+_mask"),
		("cv_getAffineTransform_const_Point2fXX_const_Point2fXX", "+_slice"),
		("cv_getPerspectiveTransform_const_Point2fXX_const_Point2fXX", "+_slice"), // 3.4
		("cv_getPerspectiveTransform_const_Point2fXX_const_Point2fXX_int", "+_slice"), // 4.x
		("cv_getRotationMatrix2D__Point2f_double_double", "get_rotation_matrix_2d_matx"),
		("cv_goodFeaturesToTrack_const__InputArrayR_const__OutputArrayR_int_double_double_const__InputArrayR_int_int_bool_double", "+_with_gradient"),
		("cv_rectangle_const__InputOutputArrayR_Point_Point_const_ScalarR_int_int_int", "+_points"),
	])
}

fn line_descriptor_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([(
		"cv_line_descriptor_LSDDetector_detect_const_const_vectorLMatGR_vectorLvectorLKeyLineGGR_int_int_const_vectorLMatGR",
		"+_multiple",
	)])
}

fn ml_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([
		("cv_ml_ParamGrid_ParamGrid_double_double_double", "for_range"),
		("cv_ml_SVM_trainAuto_const__InputArrayR_int_const__InputArrayR_int_PtrLParamGridG_PtrLParamGridG_PtrLParamGridG_PtrLParamGridG_PtrLParamGridG_PtrLParamGridG_bool", "+_with_data"),
		("cv_ml_StatModel_train_const_PtrLTrainDataGR_int", "+_with_data"),

	])
}

fn objdetect_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([
		("cv_BaseCascadeClassifier_detectMultiScale_const__InputArrayR_vectorLRectGR_vectorLintGR_double_int_int_Size_Size", "+_num"),
		("cv_BaseCascadeClassifier_detectMultiScale_const__InputArrayR_vectorLRectGR_vectorLintGR_vectorLdoubleGR_double_int_int_Size_Size_bool", "+_levels"),
		("cv_HOGDescriptor_HOGDescriptor_const_StringR", "+_from_file"),
		("cv_HOGDescriptor_detectMultiScale_const_const__InputArrayR_vectorLRectGR_vectorLdoubleGR_double_Size_Size_double_double_bool", "+_weights"),
		("cv_HOGDescriptor_detect_const_const_MatR_vectorLPointGR_vectorLdoubleGR_double_Size_Size_const_vectorLPointGR", "+_weights"), // 3.4
		("cv_HOGDescriptor_detect_const_const__InputArrayR_vectorLPointGR_vectorLdoubleGR_double_Size_Size_const_vectorLPointGR", "+_weights"), // 4.x
		("cv_HOGDescriptor_setSVMDetector_const__InputArrayR", "+_input_array"),
		("cv_aruco_getPredefinedDictionary_int", "+_i32"),
		("cv_groupRectangles_vectorLRectGR_int_double_vectorLintGX_vectorLdoubleGX", "+_levelweights"),
		("cv_groupRectangles_vectorLRectGR_vectorLintGR_int_double", "+_weights"),
		("cv_groupRectangles_vectorLRectGR_vectorLintGR_vectorLdoubleGR_int_double", "+_levels"),
	])
}

fn photo_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([
		(
			"cv_AlignMTB_process_const__InputArrayR_vectorLMatGR_const__InputArrayR_const__InputArrayR",
			"+_with_response",
		),
		(
			"cv_MergeDebevec_process_const__InputArrayR_const__OutputArrayR_const__InputArrayR_const__InputArrayR",
			"+_with_response",
		),
		(
			"cv_MergeMertens_process_const__InputArrayR_const__OutputArrayR_const__InputArrayR_const__InputArrayR",
			"+_with_response",
		),
		(
			"cv_MergeRobertson_process_const__InputArrayR_const__OutputArrayR_const__InputArrayR_const__InputArrayR",
			"+_with_response",
		),
		(
			"cv_cuda_fastNlMeansDenoisingColored_const__InputArrayR_const__OutputArrayR_float_float_int_int_StreamR",
			"+_cuda",
		),
		(
			"cv_cuda_fastNlMeansDenoising_const__InputArrayR_const__OutputArrayR_float_int_int_StreamR",
			"+_cuda",
		),
		(
			"cv_fastNlMeansDenoisingMulti_const__InputArrayR_const__OutputArrayR_int_int_const_vectorLfloatGR_int_int_int",
			"+_vec",
		),
		(
			"cv_fastNlMeansDenoising_const__InputArrayR_const__OutputArrayR_const_vectorLfloatGR_int_int_int",
			"+_vec",
		),
	])
}

fn stitching_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([
		(
			"cv_Stitcher_composePanorama_const__InputArrayR_const__OutputArrayR",
			"+_images",
		),
		(
			"cv_Stitcher_stitch_const__InputArrayR_const__InputArrayR_const__OutputArrayR",
			"+_mask",
		),
	])
}

fn surface_matching_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([(
		"cv_ppf_match_3d_ICP_registerModelToScene_const_MatR_const_MatR_vectorLPose3DPtrGR",
		"+_vec",
	)])
}

fn text_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([
		("cv_text_BaseOCR_run_MatR_MatR_stringR_vectorLRectGX_vectorLstringGX_vectorLfloatGX_int", "+_mask"),
		("cv_text_OCRBeamSearchDecoder_create_const_StringR_const_StringR_const__InputArrayR_const__InputArrayR_decoder_mode_int", "+_from_file"),
		("cv_text_OCRBeamSearchDecoder_run_MatR_MatR_stringR_vectorLRectGX_vectorLstringGX_vectorLfloatGX_int", "+_multiple_mask"),
		("cv_text_OCRBeamSearchDecoder_run_MatR_stringR_vectorLRectGX_vectorLstringGX_vectorLfloatGX_int", "+_multiple"),
		("cv_text_OCRBeamSearchDecoder_run_const__InputArrayR_const__InputArrayR_int_int", "+_mask"),
		("cv_text_OCRHMMDecoder_create_const_StringR_const_StringR_const__InputArrayR_const__InputArrayR_int_int", "+_from_file"),
		("cv_text_OCRHMMDecoder_run_MatR_MatR_stringR_vectorLRectGX_vectorLstringGX_vectorLfloatGX_int", "+_multiple_mask"),
		("cv_text_OCRHMMDecoder_run_MatR_stringR_vectorLRectGX_vectorLstringGX_vectorLfloatGX_int", "+_multiple"),
		("cv_text_OCRHMMDecoder_run_const__InputArrayR_const__InputArrayR_int_int", "+_mask"),
		("cv_text_OCRHolisticWordRecognizer_run_MatR_MatR_stringR_vectorLRectGX_vectorLstringGX_vectorLfloatGX_int", "+_mask"),
		("cv_text_OCRTesseract_run_MatR_MatR_stringR_vectorLRectGX_vectorLstringGX_vectorLfloatGX_int", "+_multiple_mask"),
		("cv_text_OCRTesseract_run_MatR_stringR_vectorLRectGX_vectorLstringGX_vectorLfloatGX_int", "+_multiple"),
		("cv_text_OCRTesseract_run_const__InputArrayR_const__InputArrayR_int_int", "+_mask"),
		("cv_text_TextDetectorCNN_create_const_StringR_const_StringR_vectorLSizeG", "+_with_sizes"),
		("cv_text_createERFilterNM1_const_StringR_int_float_float_float_bool_float", "+_from_file"),
		("cv_text_createERFilterNM2_const_StringR_float", "+_from_file"),
		("cv_text_detectRegions_const__InputArrayR_const_PtrLERFilterGR_const_PtrLERFilterGR_vectorLRectGR_int_const_StringR_float", "+_from_file"),

	])
}

fn videoio_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([
		("cv_VideoCapture_VideoCapture_const_StringR", "from_file_default"), // 3.4
		("cv_VideoCapture_VideoCapture_const_StringR_int", "from_file"),
		(
			"cv_VideoCapture_VideoCapture_const_StringR_int_const_vectorLintGR",
			"from_file_with_params",
		),
		("cv_VideoCapture_VideoCapture_int", "+_default"), // 3.4
		("cv_VideoCapture_VideoCapture_int_int_const_vectorLintGR", "+_with_params"),
		("cv_VideoCapture_open_const_StringR", "+_file_default"), // 3.4
		("cv_VideoCapture_open_const_StringR_int", "+_file"),
		(
			"cv_VideoCapture_open_const_StringR_int_const_vectorLintGR",
			"+_file_with_params",
		),
		("cv_VideoCapture_open_int", "+_default"), // 3.4
		("cv_VideoCapture_open_int_int_const_vectorLintGR", "+_with_params"),
		(
			"cv_VideoWriter_VideoWriter_const_StringR_int_int_double_Size_bool",
			"+_with_backend",
		),
		("cv_VideoWriter_open_const_StringR_int_int_double_Size_bool", "+_with_backend"),
	])
}

fn videostab_factory() -> HashMap<&'static str, &'static str> {
	HashMap::from([(
		"cv_videostab_KeypointBasedMotionEstimator_estimate_const_MatR_const_MatR_boolX",
		"+_mat",
	)])
}
