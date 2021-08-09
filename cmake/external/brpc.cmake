INCLUDE(ExternalProject)

find_package(OpenSSL REQUIRED)

message(STATUS "ssl:" ${OPENSSL_SSL_LIBRARY})
message(STATUS "crypto:" ${OPENSSL_CRYPTO_LIBRARY})

ADD_LIBRARY(ssl SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET ssl PROPERTY IMPORTED_LOCATION ${OPENSSL_SSL_LIBRARY})

ADD_LIBRARY(crypto SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET crypto PROPERTY IMPORTED_LOCATION ${OPENSSL_CRYPTO_LIBRARY})

SET(BRPC_SOURCES_DIR ${THIRD_PARTY_PATH}/brpc)
SET(BRPC_INSTALL_DIR ${THIRD_PARTY_PATH}/install/brpc)
SET(BRPC_INCLUDE_DIR "${BRPC_INSTALL_DIR}/include" CACHE PATH "brpc include directory." FORCE)
SET(BRPC_LIBRARIES "${BRPC_INSTALL_DIR}/lib/libbrpc.a" CACHE FILEPATH "brpc library." FORCE)

INCLUDE_DIRECTORIES(${BRPC_INCLUDE_DIR})

# Reference https://stackoverflow.com/questions/45414507/pass-a-list-of-prefix-paths-to-externalproject-add-in-cmake-args
set(prefix_path "${THIRD_PARTY_PATH}/install/gflags|${THIRD_PARTY_PATH}/install/leveldb|${THIRD_PARTY_PATH}/install/snappy|${THIRD_PARTY_PATH}/install/gtest|${THIRD_PARTY_PATH}/install/protobuf|${THIRD_PARTY_PATH}/install/zlib|${THIRD_PARTY_PATH}/install/glog")

# Disable narrowing warning
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -faligned-new")

IF (WITH_HIERARCHICAL_HCCL)
SET(BRPC_GIT_URL "https://github.com/lw921014/incubator-brpc")
SET(BRPC_GIT_TAG "paddle_eccl_ascend")
MESSAGE(STATUS "Build in ascend platform: Download form ${BRPC_GIT_URL}.git:${BRPC_GIT_TAG}")
ELSE()
SET(BRPC_GIT_URL "https://github.com/apache/incubator-brpc")
SET(BRPC_GIT_TAG "0.9.8-rc01")
ENDIF()
# If minimal .a is need, you can set  WITH_DEBUG_SYMBOLS=OFF
ExternalProject_Add(
        extern_brpc
        ${EXTERNAL_PROJECT_LOG_ARGS}
        # TODO(gongwb): change to de newst repo when they changed
        GIT_REPOSITORY  ${BRPC_GIT_URL}
        GIT_TAG         ${BRPC_GIT_TAG}
        PREFIX          ${BRPC_SOURCES_DIR}
        UPDATE_COMMAND  ""
        CMAKE_ARGS
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
        -DCMAKE_INSTALL_PREFIX=${BRPC_INSTALL_DIR}
        -DCMAKE_INSTALL_LIBDIR=${BRPC_INSTALL_DIR}/lib
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
        -DCMAKE_PREFIX_PATH=${prefix_path}
        -DWITH_GLOG=ON
        -DIOBUF_WITH_HUGE_BLOCK=ON
        -DBRPC_WITH_RDMA=${WITH_BRPC_RDMA}
        -DWITH_DEBUG_SYMBOLS=ON
        ${EXTERNAL_OPTIONAL_ARGS}
        LIST_SEPARATOR |
        CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${BRPC_INSTALL_DIR}
        -DCMAKE_INSTALL_LIBDIR:PATH=${BRPC_INSTALL_DIR}/lib
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
)
ADD_DEPENDENCIES(extern_brpc protobuf ssl crypto leveldb gflags glog snappy)
ADD_LIBRARY(brpc STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET brpc PROPERTY IMPORTED_LOCATION ${BRPC_LIBRARIES})
ADD_DEPENDENCIES(brpc extern_brpc)

add_definitions(-DBRPC_WITH_GLOG)

LIST(APPEND external_project_dependencies brpc)
