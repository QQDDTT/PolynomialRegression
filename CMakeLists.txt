cmake_minimum_required(VERSION 3.10)
project(PolynomialRegression)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找必要的库
# 使用 pkg-config 获取 mlpack 的头文件和库路径
find_package(PkgConfig REQUIRED)
pkg_check_modules(MLPACK REQUIRED mlpack)

# 输出调试信息，确保 pkg-config 找到了正确的路径
message(STATUS "MLPACK_INCLUDE_DIRS: ${MLPACK_INCLUDE_DIRS}")
message(STATUS "MLPACK_LIBRARY_DIRS: ${MLPACK_LIBRARY_DIRS}")

# 包含头文件目录
include_directories(${MLPACK_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

# 确保库路径可用
link_directories(${MLPACK_LIBRARY_DIRS} ${ARMADILLO_LIBRARY_DIRS})

# 查找 src 文件夹下的所有 .cpp 文件
file(GLOB SOURCES
    "src/modules/*.cpp"
    "src/utils/*.cpp"
    "src/main.cpp"
)

# 输出找到的源文件
message(STATUS "Found source files: ${SOURCES}")

# 添加可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 定义项目根目录路径
add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# 链接库文件
target_link_libraries(${PROJECT_NAME} ${MLPACK_LIBRARIES} ${ARMADILLO_LIBRARIES})
