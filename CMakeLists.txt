cmake_minimum_required(VERSION 3.0.0)
project(HEPHero)

set(ANALYSIS "AP_Template_R3")

include_directories(include ${ANALYSIS}/include $ENV{MY_ONNX_PATH}/include)

link_directories($ENV{MY_ONNX_PATH}/lib)

#ROOT 6.10+
find_package(ROOT CONFIG REQUIRED)

find_program(FastJet_CONFIG_EXECUTABLE fastjet-config)
execute_process(COMMAND ${FastJet_CONFIG_EXECUTABLE} --cxxflags OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE FastJet_CXXFLAGS)
execute_process(COMMAND ${FastJet_CONFIG_EXECUTABLE} --libs OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE FastJet_LDFLAGS)

find_package(ZLIB)

add_subdirectory(include/HighFive)

#set(CMAKE_PREFIX_PATH $ENV{MY_TORCH_PATH})
#find_package(Torch REQUIRED)
##set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#add_executable(RunAnalysis ${SOURCES})

add_executable(RunAnalysis)
#target_sources(app PRIVATE "app.cpp")
add_subdirectory(src)
add_subdirectory(${ANALYSIS}/src)
add_subdirectory(${ANALYSIS}/ana)

target_link_libraries(RunAnalysis PUBLIC ROOT::Core ROOT::Imt ROOT::RIO ROOT::Net ROOT::Hist ROOT::Graf ROOT::Graf3d ROOT::Gpad ROOT::ROOTVecOps ROOT::Tree ROOT::TreePlayer ROOT::Rint ROOT::Postscript ROOT::Matrix ROOT::Physics ROOT::MathCore ROOT::Thread ROOT::MultiProc ROOT::ROOTDataFrame ROOT::MathMore ZLIB::ZLIB HighFive onnxruntime) #"${TORCH_LIBRARIES}")

target_compile_options(RunAnalysis PRIVATE ${FastJet_CXXFLAGS})
target_link_libraries(RunAnalysis PRIVATE ${FastJet_LDFLAGS})

set_property(TARGET RunAnalysis PROPERTY CXX_STANDARD 14)


