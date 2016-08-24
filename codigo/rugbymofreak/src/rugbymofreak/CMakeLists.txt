cmake_minimum_required(VERSION 2.6) #2.8

## start setting
SET(CMAKE_BUILD_TYPE Debug)
SET (this_target RugbyMoFREAK)
PROJECT(${this_target})


FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    message(STATUS "OpenMP FOUND")
    #SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()



#OPTION (USE_OpenMP "Use OpenMP" ON)
#IF(USE_OpenMP)
#  FIND_PACKAGE(OpenMP)
#  IF(OPENMP_FOUND)
#    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#  ENDIF()
#ENDIF()

SET("OpenCV_DIR" "/home/lucia/libs/opencv-2.4.2/release/installed")
SET("OpenCV_INCLUDE_DIR" "${OpenCV_DIR}/include")
SET("OpenCV_LIBRARIES_PATH" "${OpenCV_DIR}/lib")
SET("OpenCV_LIBRARIES" "${OpenCV_LIBRARIES_PATH}/libopencv_video.so" "${OpenCV_LIBRARIES_PATH}/libopencv_nonfree.so"
"${OpenCV_LIBRARIES_PATH}/libopencv_features2d.so"
"${OpenCV_LIBRARIES_PATH}/libopencv_imgproc.so"
"${OpenCV_LIBRARIES_PATH}/libopencv_highgui.so"
"${OpenCV_LIBRARIES_PATH}/libopencv_calib3d.so"
"${OpenCV_LIBRARIES_PATH}/libopencv_core.so"
"${OpenCV_LIBRARIES_PATH}/libopencv_videostab.so")

FIND_PACKAGE(Boost REQUIRED COMPONENTS system filesystem)
#FIND_PACKAGE(OpenCV REQUIRED COMPONENTS core calib3d highgui imgproc features2d nonfree video)

message(STATUS "OpenCV_DIR: " ${OpenCV_DIR})
message(STATUS "OpenCV_INCLUDE_DIR: " ${OpenCV_INCLUDE_DIR})
message(STATUS "OpenCV_LIBRARIES: " ${OpenCV_LIBRARIES})
message(STATUS "OpenMP_CXX_LIBRARIES: " ${OpenMP_CXX_LIBRARIES})


## section: include directory
INCLUDE_DIRECTORIES(
    ${Boost_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
    #../opencv/include
    #../boost/include
)

## section: source files
# Add your source files here (one file per line), please SORT in alphabetical order for future maintenance
SET (${this_target}_SOURCE_FILES
    agast5_8.cc
    agast5_8_nms.cc
    agast7_12d.cc
    agast7_12d_nms.cc
    agast7_12s.cc
    agast7_12s_nms.cc
    AstDetector.cc
    BagOfWordsRepresentation.cpp
    brisk.cpp
    Clustering.cpp
    main.cpp
    MoFREAKUtilities.cpp
    MoSIFTUtilities.cpp
    nonMaximumSuppression.cc
    oast9_16.cc
    oast9_16_nms.cc
    svm.cpp
    SVMInterface.cpp
    svm_interface.cpp
)

## section: header files
# Add your header files here(one file per line), please SORT in alphabetical order for future maintenance!
SET(${this_target}_HEADER_FILES
    agast5_8.h
    agast7_12d.h
    agast7_12s.h
    AstDetector.h
    BagOfWordsRepresentation.h
    brisk.h
    Clustering.h
    cvWrapper.h
    hammingsse.hpp
    MoFREAKUtilities.h
    MoSIFTUtilities.h
    oast9_16.h
    OpenCVToQtInterfacing.h
    svm.h
    SVMInterface.h
)

SOURCE_GROUP("Source Files" FILES 
    main.cpp
    svm.cpp
    svm_interface.cpp
    SVMInterface.cpp
    BagOfWordsRepresentation.cpp
    Clustering.cpp
    MoFREAKUtilities.cpp
    MoSIFTUtilities.cpp
)

SOURCE_GROUP("Header Files" FILES 
    BagOfWordsRepresentation.h
    Clustering.h
    MoFREAKUtilities.h
    MoSIFTUtilities.h
    OpenCVToQtInterfacing.h
    svm.h
    SVMInterface.h
)

SOURCE_GROUP("Resource Files" FILES 
)

SOURCE_GROUP("agaststuff" FILES 
    AstDetector.cc
    agast7_12s_nms.cc
    agast7_12s.cc
    agast7_12d_nms.cc
    agast7_12d.cc
    agast5_8_nms.cc
    agast5_8.cc
    nonMaximumSuppression.cc
    oast9_16.cc
    oast9_16_nms.cc
    agast7_12s.h
    agast7_12d.h
    agast5_8.h
    AstDetector.h
    cvWrapper.h
    oast9_16.h
)

SOURCE_GROUP("briskstuff" FILES 
    brisk.cpp
    brisk.h
    hammingsse.hpp
)

#ADD_LIBRARY(MyLibrary ${MySources})

SET_SOURCE_FILES_PROPERTIES(${this_target}_HEADER_FILES
                            PROPERTIES HEADER_FILE_ONLY TRUE)

#agregado
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x -msse2 -msse3 -mssse3 -pg ${OpenMP_CXX_FLAGS}") #
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

LIST(APPEND ${this_target}_SOURCE_FILES ${${this_target}_HEADER_FILES})

## section: add definitions
#   add prefix -D. example> -DSHP
#  - DO NOT add  the following definitions(already defined in ${OSP_DEFINITIONS}:
#   -DSHP, -DWIN32, -D_WINDOWS, -D_DEBUG, -D_USRDLL, -D_CRT_SECURE_NO_DEPRECATE
ADD_DEFINITIONS(
)

## section: add target

ADD_EXECUTABLE(${this_target} ${${this_target}_SOURCE_FILES})

## section: add dependency
# dependency determines overall build order.
#ADD_DEPENDENCIES(${this_target} 
    #boost_filesystem-vc100-mt-gd-1_51.so
    #boost_filesystem
    #/home/lucia/Documentos/git/tesina/codigo/rugbymofreak/src/boost/lib/libboost_filesystem.so
#)


## section: set link libraries
TARGET_LINK_LIBRARIES( ${this_target}
    #boost_filesystem-vc100-mt-gd-1_51.so
    #boost_filesystem
    #/home/lucia/Documentos/git/tesina/codigo/rugbymofreak/src/opencv/lib/libopencv_core.so
    #/home/lucia/Documentos/git/tesina/codigo/rugbymofreak/src/opencv/lib/libopencv_calib3d.so
    #/home/lucia/Documentos/git/tesina/codigo/rugbymofreak/src/opencv/lib/libopencv_highgui.so
    #/home/lucia/Documentos/git/tesina/codigo/rugbymofreak/src/opencv/lib/libopencv_imgproc.so
    #/home/lucia/Documentos/git/tesina/codigo/rugbymofreak/src/opencv/lib/libopencv_features2d.so
    #/home/lucia/Documentos/git/tesina/codigo/rugbymofreak/src/opencv/lib/libopencv_nonfree.so
    #/home/lucia/Documentos/git/tesina/codigo/rugbymofreak/src/opencv/lib/libopencv_video.so
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
)

