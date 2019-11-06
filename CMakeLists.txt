# Made by Pedro Ramoneda. 27/10/19


cmake_minimum_required(VERSION 3.13)
project( sdi )

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})

# separate versions of NetworkingUtils.cpp and UdpSocket.cpp are provided for Win32 and POSIX
# the IpSystemTypePath selects the correct ones based on the current platform

IF(WIN32)
    set(IpSystemTypePath ip/win32)
    set(LIBS ${LIBS} Ws2_32 winmm)
ELSE(WIN32)
    set(IpSystemTypePath ip/posix)
ENDIF(WIN32)

ADD_LIBRARY(oscpack

        ip/IpEndpointName.h
        ip/IpEndpointName.cpp

        ip/NetworkingUtils.h
        ${IpSystemTypePath}/NetworkingUtils.cpp

        ip/UdpSocket.h
        ${IpSystemTypePath}/UdpSocket.cpp

        ip/PacketListener.h
        ip/TimerListener.h

        osc/OscTypes.h
        osc/OscTypes.cpp
        osc/OscHostEndianness.h
        osc/OscException.h
        osc/OscPacketListener.h
        osc/MessageMappingOscPacketListener.h
        osc/OscReceivedElements.h
        osc/OscReceivedElements.cpp
        osc/OscPrintReceivedElements.h
        osc/OscPrintReceivedElements.cpp
        osc/OscOutboundPacketStream.h
        osc/OscOutboundPacketStream.cpp

        )

###############################


file(GLOB_RECURSE srcs ./src/*)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set(CMAKE_CXX_STANDARD 14)
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )


add_executable( sdi main.cpp)
target_link_libraries( sdi oscpack ${OpenCV_LIBS} ${LIBS})

##############################

if(MSVC)
    # Force to always compile with W4
    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    # Update if necessary
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-long-long -pedantic")
endif()