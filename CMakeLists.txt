# ===
# This is probably a _hot mess_
# But it works.
# ===

cmake_minimum_required( VERSION 3.10 )

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

### c++11 standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)

set(CMAKE_FIND_FRAMEWORK NEVER)
set(CMAKE_FIND_APPBUNDLE NEVER)

set( CMAKE_PREFIX_PATH "/usr/local")

project( athelas VERSION 0.2 LANGUAGES CXX )
enable_language(CXX)

set(HYDRO_DIR "${PROJECT_SOURCE_DIR}/Example/Hydro")
message(STATUS "HYDRO_DIR = ${HYDRO_DIR}")
include_directories(HYDRO_DIR)
include_directories(Library)

include_directories(athelas PUBLIC
  "${PROJECT_BINARY_DIR}"
  )

# --- Source Files ---
set( ATHELAS_SOURCES 
  "${HYDRO_DIR}/Initialization.cpp" 
  "Library/BoundaryConditionsLibrary.cpp"
  "Library/CharacteristicDecomposition.cpp"
  "Library/DataStructures.cpp"
  "Library/EquationOfStateLibrary_IDEAL.cpp"
  "Library/FluidUtilities.cpp"
  "Library/Fluid_Discretization.cpp"
  "Library/Grid.cpp"
  "Library/IOLibrary.cpp"
  "Library/LinearAlgebraModules.cpp"
  "Library/PolynomialBasis.cpp"
  "Library/QuadratureLibrary.cpp"
  "Library/SlopeLimiter.cpp"
  "Library/SlopeLimiter_Utilities.cpp"
  "Library/Timestepper.cpp")

add_library(libathelas SHARED ${ATHELAS_SOURCES})
set_target_properties(libathelas
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    PREFIX ""
    SUFFIX ".so"
)

# Check for FindOpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OPEN MP ${OpenMP_CXX_SPEC_DATE}")
    if((OpenMP_CXX_SPEC_DATE GREATER_EQUAL "201511"))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOMP45")
        message(STATUS "Using OpenMP 4.5 directives")
    else()
        message(STATUS "Not using OpenMP 4.5 directives: ${OpenMP_C_VERSION_MAJOR} ${OpenMP_C_VERSION_MINOR}")
    endif()
endif(OPENMP_FOUND)


# TODO: Need a better handling of HDF5/LAPACKE/cBLAS. 
# My system is finding versions of these that don't work (e.g., LAPACK from miniconda....)
# HDF5
find_package ( HDF5 REQUIRED COMPONENTS CXX HL )
# set( HDF5_INCLUDE_DIRS "/usr/local/Cellar/hdf5/1.12.1/include" )
# set( HDF5_C_LIBRARIES "/usr/local/Cellar/hdf5/1.12.1/lib/libhdf5_cpp.dylib" )
# set( HDF5_LIBRARIES "/usr/local/Cellar/hdf5/1.12.1/lib/libhdf5.dylib")

# Check BLAS/LAPACK

find_package( LAPACK REQUIRED  )
#set( LAPACK_LIBRARIES /usr/local/opt/lapack/lib/liblapacke.3.dylib )
#set( LAPACK_INCLUDE_DIRS /usr/local/opt/lapack/include )

find_package( BLAS REQUIRED  )
#set( BLAS_LIBRARIES /usr/local/opt/openblas/lib/libopenblas.dylib )
#set( BLAS_INCLUDE_DIRS /usr/local/opt/openblas/include )

target_link_libraries( libathelas PUBLIC ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES} )
target_include_directories( libathelas PUBLIC ${HDF5_INCLUDE_DIRS})

message(STATUS "HDF5_LIBRARIES = ${HDF5_CXX_LIBRARIES}")

target_link_libraries( libathelas PUBLIC ${LAPACK_LIBRARIES} )
target_include_directories( libathelas PUBLIC ${LAPACK_INCLUDE_DIRS})
message(STATUS "LAPACK_LIBRARIES = ${LAPACK_LIBRARIES}")


target_link_libraries( libathelas PUBLIC ${BLAS_LIBRARIES} )
target_include_directories( libathelas PUBLIC ${BLAS_INCLUDE_DIRS})
message( STATUS "BLAS_LIBRARIES: ${BLAS_LIBRARIES}" )

install(TARGETS libathelas DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/)

# --- main ---

add_executable( main ${HYDRO_DIR}/Driver.cpp )
set_target_properties(main
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

target_compile_options( main PRIVATE -fexceptions -O2 -Wall )
add_dependencies( main libathelas )
target_link_libraries( main libathelas ${HDF5_C_LIBRARIES} ${HDF5_LIBRARIES} ${BLAS_LIBRARIES} )
