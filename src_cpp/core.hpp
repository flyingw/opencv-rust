#include "ocvrs_common.hpp"
#include <opencv2/core/affine.hpp>
#ifdef HAVE_OPENCL
	#include <opencv2/core/ocl.hpp>
	// opengl.hpp, va_intel.hpp and directx.hpp unconditionally include ocl.hpp thus it needs to be within ifdef HAVE_OPENCL
	#ifdef HAVE_OPENGL
		#include <opencv2/core/opengl.hpp>
	#endif
	#include <opencv2/core/va_intel.hpp>
	#include <opencv2/core/directx.hpp>
#endif
#ifdef HAVE_CUDA
	#include <opencv2/core/cuda.hpp>
#endif
#if (CV_VERSION_MAJOR == 3 && CV_VERSION_MINOR == 4 && CV_VERSION_REVISION >= 4) /* 3.4.4+ */ \
	|| (CV_VERSION_MAJOR == 4) /* 4.0+ */ \
	|| (CV_VERSION_MAJOR == 5) /* 5.0+ */
	#include <opencv2/core/bindings_utils.hpp>
#endif
#include <opencv2/core/utils/logger.hpp>
#if (CV_VERSION_MAJOR == 3 && CV_VERSION_MINOR == 4 && CV_VERSION_REVISION >= 7) /* 3.4.7+ */ \
	|| (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR == 1 && CV_VERSION_REVISION >= 1) /* 4.1.1+ */ \
	|| (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 2) /* 4.2+ */ \
	|| (CV_VERSION_MAJOR == 5) /* 5.0+ */
	#include <opencv2/core/async.hpp>
#endif

namespace cv {
	static const char *CV_VERSION_OCVRS_OVERRIDE = CV_VERSION;
}
