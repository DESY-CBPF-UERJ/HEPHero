cmake_minimum_required(VERSION 3.0.0)
project(HEPHero)

include_directories(include)

set(SOURCES src/RunAnalysis.cpp src/HEPHero.cpp src/correction.cc src/formula_ast.cc src/RoccoR.cc src/RecoilCorrector.cc src/HEPTools.cpp src/CMSTools.cpp src/ANATools.cpp src/HEPObjects.cpp ana/Test.cpp) 

#ROOT 6.10+
find_package(ROOT CONFIG REQUIRED)

find_package(ZLIB)

add_subdirectory(include/HighFive)

set(CMAKE_PREFIX_PATH $ENV{MY_TORCH_PATH})
find_package(Torch REQUIRED)
##set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(RunAnalysis ${SOURCES})

target_link_libraries(RunAnalysis PUBLIC ROOT::Core ROOT::Imt ROOT::RIO ROOT::Net ROOT::Hist ROOT::Graf ROOT::Graf3d ROOT::Gpad ROOT::ROOTVecOps ROOT::Tree ROOT::TreePlayer ROOT::Rint ROOT::Postscript ROOT::Matrix ROOT::Physics ROOT::MathCore ROOT::Thread ROOT::MultiProc ROOT::ROOTDataFrame ROOT::MathMore ZLIB::ZLIB HighFive "${TORCH_LIBRARIES}")

set_property(TARGET RunAnalysis PROPERTY CXX_STANDARD 14)
