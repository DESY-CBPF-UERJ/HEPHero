cmake_minimum_required(VERSION 3.0.0)
project(HEPHero)

include_directories(include)

set(SOURCES src/RunAnalysis.cpp src/HEPHero.cpp src/correction.cc src/formula_ast.cc src/RoccoR.cc src/RecoilCorrector.cc src/HEPTools.cpp src/GENTools.cpp src/CMSTools.cpp src/ANATools.cpp src/HEPObjects.cpp
src/HepMC3/GenCrossSection.cc
src/HepMC3/GenEvent.cc
src/HepMC3/GenHeavyIon.cc
src/HepMC3/GenParticle.cc
src/HepMC3/GenPdfInfo.cc
src/HepMC3/GenRunInfo.cc
src/HepMC3/GenVertex.cc
src/HepMC3/HEPEVT_Wrapper.cc
src/HepMC3/LHEFAttributes.cc
src/HepMC3/Print.cc
src/HepMC3/ReaderAscii.cc
src/HepMC3/ReaderAsciiHepMC2.cc
src/HepMC3/ReaderHEPEVT.cc
src/HepMC3/ReaderLHEF.cc
src/HepMC3/ReaderPlugin.cc
src/HepMC3/Setup.cc
src/HepMC3/WriterAscii.cc
src/HepMC3/WriterAsciiHepMC2.cc
src/HepMC3/WriterHEPEVT.cc
src/HepMC3/WriterPlugin.cc
src/HepMC3/WriterDOT.cc
src/HepMC3/Relatives.cc
src/HepMC3/Selector.cc
) 

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
