# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

function(generate_unify_header DIR_NAME)
    set(options "")
    set(oneValueArgs HEADER_NAME SKIP_SUFFIX)
    set(multiValueArgs "")
    cmake_parse_arguments(generate_unify_header "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    # get header name and suffix
    set(header_name "${DIR_NAME}")
    list(LENGTH generate_unify_header_HEADER_NAME generate_unify_header_HEADER_NAME_len)
    if(${generate_unify_header_HEADER_NAME_len} GREATER 0)
        set(header_name "${generate_unify_header_HEADER_NAME}")
    endif()
    set(skip_suffix "")
    list(LENGTH generate_unify_header_SKIP_SUFFIX generate_unify_header_SKIP_SUFFIX_len)
    if(${generate_unify_header_SKIP_SUFFIX_len} GREATER 0)
        set(skip_suffix "${generate_unify_header_SKIP_SUFFIX}")
    endif()

    # generate target header file
    set(header_file ${CMAKE_CURRENT_SOURCE_DIR}/include/${header_name}.h)
    file(WRITE ${header_file} "// Header file generated by paddle/phi/CMakeLists.txt for external users,\n// DO NOT edit or include it within paddle.\n\n#pragma once\n\n")

    # get all top-level headers and write into header file
    file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}\/${DIR_NAME}\/*.h")
    foreach(header ${HEADERS})
        if("${skip_suffix}" STREQUAL "")
            string(REPLACE "${PADDLE_SOURCE_DIR}\/" "" header "${header}")
            file(APPEND ${header_file} "#include \"${header}\"\n")
        else()
            string(FIND "${header}" "${skip_suffix}.h" skip_suffix_found)
            if(${skip_suffix_found} EQUAL -1)
                string(REPLACE "${PADDLE_SOURCE_DIR}\/" "" header "${header}")
                file(APPEND ${header_file} "#include \"${header}\"\n")
            endif()
        endif()
    endforeach()
    # append header into extension.h
    string(REPLACE "${PADDLE_SOURCE_DIR}\/" "" header_file "${header_file}")
    file(APPEND ${phi_extension_header_file} "#include \"${header_file}\"\n")
endfunction()

# call kernel_declare need to make sure whether the target of input exists
function(kernel_declare TARGET_LIST)
    foreach(kernel_path ${TARGET_LIST})
        file(READ ${kernel_path} kernel_impl)
        string(REGEX MATCH "(PD_REGISTER_KERNEL|PD_REGISTER_GENERAL_KERNEL)\\([ \t\r\n]*[a-z0-9_]*,[ \t\r\n\/]*[a-z0-9_]*" first_registry "${kernel_impl}")
        if (NOT first_registry STREQUAL "")
            # some gpu kernel only can run on cuda, not support rocm, so we add this branch
            if (WITH_ROCM)
                string(FIND "${first_registry}" "cuda_only" pos)
                if(pos GREATER 1)
                    continue()
                endif()
            endif()
            # parse the first kernel name
            string(REPLACE "PD_REGISTER_KERNEL(" "" kernel_name "${first_registry}")
            string(REPLACE "PD_REGISTER_GENERAL_KERNEL(" "" kernel_name "${kernel_name}")
            string(REPLACE "," "" kernel_name "${kernel_name}")
            string(REGEX REPLACE "[ \t\r\n]+" "" kernel_name "${kernel_name}")
            string(REGEX REPLACE "//cuda_only" "" kernel_name "${kernel_name}")
            # append kernel declare into declarations.h
            # TODO(chenweihang): default declare ALL_LAYOUT for each kernel
            if (${kernel_path} MATCHES "./cpu\/")
                file(APPEND ${kernel_declare_file} "PD_DECLARE_KERNEL(${kernel_name}, CPU, ALL_LAYOUT);\n")
            elseif (${kernel_path} MATCHES "./gpu\/")
                file(APPEND ${kernel_declare_file} "PD_DECLARE_KERNEL(${kernel_name}, GPU, ALL_LAYOUT);\n")
            elseif (${kernel_path} MATCHES "./xpu\/")
                file(APPEND ${kernel_declare_file} "PD_DECLARE_KERNEL(${kernel_name}, XPU, ALL_LAYOUT);\n")
            elseif (${kernel_path} MATCHES "./gpudnn\/")
                file(APPEND ${kernel_declare_file} "PD_DECLARE_KERNEL(${kernel_name}, GPUDNN, ALL_LAYOUT);\n")
            elseif (${kernel_path} MATCHES "./kps\/")
                file(APPEND ${kernel_declare_file} "PD_DECLARE_KERNEL(${kernel_name}, KPS, ALL_LAYOUT);\n")
            else ()
                # deal with device independent kernel, now we use CPU temporaary
                file(APPEND ${kernel_declare_file} "PD_DECLARE_KERNEL(${kernel_name}, CPU, ALL_LAYOUT);\n")
            endif()
        endif()
    endforeach()
endfunction()

function(kernel_library TARGET)
    set(common_srcs)
    set(cpu_srcs)
    set(gpu_srcs)
    set(xpu_srcs)
    set(gpudnn_srcs)
    set(kps_srcs)
    # parse and save the deps kerenl targets
    set(all_srcs)
    set(kernel_deps)

    set(oneValueArgs SUB_DIR)
    set(multiValueArgs SRCS DEPS)
    set(target_build_flag 1)

    cmake_parse_arguments(kernel_library "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})
    
    # used for cc_library selected_rows dir target
    set(target_suffix "")
    if ("${kernel_library_SUB_DIR}" STREQUAL "selected_rows")
        set(target_suffix "_sr")
    endif()
    if ("${kernel_library_SUB_DIR}" STREQUAL "sparse")
        set(target_suffix "_sp")
    endif()

    list(LENGTH kernel_library_SRCS kernel_library_SRCS_len)
    # one kernel only match one impl file in each backend
    if (${kernel_library_SRCS_len} EQUAL 0)
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cc)
            list(APPEND common_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cc)
        endif()
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cpu/${TARGET}.cc AND NOT WITH_XPU_KP)
            list(APPEND cpu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/${TARGET}.cc)
        endif()
        if (WITH_GPU OR WITH_ROCM)
            if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gpu/${TARGET}.cu)
                list(APPEND gpu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/gpu/${TARGET}.cu)
            endif()
            if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gpu/${TARGET}.cu.cc)
                list(APPEND gpu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/gpu/${TARGET}.cu.cc)
            endif()
            if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/kps/${TARGET}.cu)
                list(APPEND gpu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/kps/${TARGET}.cu)
            endif()
            if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gpudnn/${TARGET}.cu)
                list(APPEND gpudnn_srcs ${CMAKE_CURRENT_SOURCE_DIR}/gpudnn/${TARGET}.cu)
            endif()
        endif()
        if (WITH_XPU)
            if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/xpu/${TARGET}.cc)
                list(APPEND xpu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/xpu/${TARGET}.cc)
            endif()
        endif()
        if (WITH_XPU_KP)
            if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/kps/${TARGET}.cu)
                # Change XPU2 file suffix
                # NOTE(chenweihang): If we can be sure that the *.kps suffix is no longer used, it can be copied directly to *.xpu
                file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kps/${TARGET}.cu DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/kps)
                file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/kps/${TARGET}.cu ${CMAKE_CURRENT_BINARY_DIR}/kps/${TARGET}.kps)
                list(APPEND kps_srcs ${CMAKE_CURRENT_BINARY_DIR}/kps/${TARGET}.kps)
            endif()
            if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cpu/${TARGET}.cc )
                list(APPEND kps_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/${TARGET}.cc)
            endif()
        endif()
    else()
        # TODO(chenweihang): impl compile by source later
    endif()

    list(APPEND all_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.h)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/impl/${TARGET}_impl.h)
        list(APPEND all_srcs ${CMAKE_CURRENT_SOURCE_DIR}/impl/${TARGET}_impl.h)
    endif()
    list(APPEND all_srcs ${common_srcs})
    list(APPEND all_srcs ${cpu_srcs})
    list(APPEND all_srcs ${gpu_srcs})
    list(APPEND all_srcs ${xpu_srcs})
    list(APPEND all_srcs ${gpudnn_srcs})
    list(APPEND all_srcs ${kps_srcs})

    set(all_include_kernels)
    set(all_kernel_name)

    foreach(src ${all_srcs})
        file(READ ${src} target_content)
        # "kernels/xxx"(DenseTensor Kernel) can only include each other, but can't include "SUB_DIR/xxx" (such as selected_rows Kernel)
        string(REGEX MATCHALL "#include \"paddle\/phi\/kernels\/[a-z0-9_]+_kernel.h\"" include_kernels ${target_content})
        list(APPEND all_include_kernels ${include_kernels})

        # "SUB_DIR/xxx" can include "kernels/xx" and "SUB_DIR/xxx"
        if (NOT "${kernel_library_SUB_DIR}" STREQUAL "")
            string(REGEX MATCHALL "#include \"paddle\/phi\/kernels\/${kernel_library_SUB_DIR}\/[a-z0-9_]+_kernel.h\"" include_kernels ${target_content})
            list(APPEND all_include_kernels ${include_kernels})
        endif()
        
        foreach(include_kernel ${all_include_kernels})
            if ("${kernel_library_SUB_DIR}" STREQUAL "")
                string(REGEX REPLACE "#include \"paddle\/phi\/kernels\/" "" kernel_name ${include_kernel})
                string(REGEX REPLACE ".h\"" "" kernel_name ${kernel_name})
                list(APPEND all_kernel_name ${kernel_name})
            else()
                # NOTE(dev): we should firstly match kernel_library_SUB_DIR.
                if (${include_kernel} MATCHES "#include \"paddle\/phi\/kernels\/${kernel_library_SUB_DIR}\/")
                    string(REGEX REPLACE "#include \"paddle\/phi\/kernels\/${kernel_library_SUB_DIR}\/" "" kernel_name ${include_kernel})
                    # for selected_rows directory, add ${target_suffix}.
                    string(REGEX REPLACE ".h\"" "${target_suffix}" kernel_name ${kernel_name})
                    list(APPEND all_kernel_name ${kernel_name})
                else()
                    string(REGEX REPLACE "#include \"paddle\/phi\/kernels\/" "" kernel_name ${include_kernel})
                    string(REGEX REPLACE ".h\"" "" kernel_name ${kernel_name})
                    list(APPEND all_kernel_name ${kernel_name})
                endif()
            endif()
            list(APPEND kernel_deps ${all_kernel_name})
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES kernel_deps)
    list(REMOVE_ITEM kernel_deps ${TARGET}${target_suffix})

    list(LENGTH common_srcs common_srcs_len)
    list(LENGTH cpu_srcs cpu_srcs_len)
    list(LENGTH gpu_srcs gpu_srcs_len)
    list(LENGTH xpu_srcs xpu_srcs_len)
    list(LENGTH gpudnn_srcs gpudnn_srcs_len)
    list(LENGTH kps_srcs kps_srcs_len)

    # kernel source file level
    # level 1: base device kernel
    # - cpu_srcs / gpu_srcs / xpu_srcs / gpudnn_srcs / kps_srcs
    # level 2: device-independent kernel
    # - common_srcs
    set(base_device_kernels)
    set(device_independent_kernel)

    # 1. Base device kernel compile
    if (${cpu_srcs_len} GREATER 0)
        cc_library(${TARGET}_cpu${target_suffix} SRCS ${cpu_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
        list(APPEND base_device_kernels ${TARGET}_cpu${target_suffix})
    endif()
    if (${gpu_srcs_len} GREATER 0)
        if (WITH_GPU)
            nv_library(${TARGET}_gpu${target_suffix} SRCS ${gpu_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
        elseif (WITH_ROCM)
            hip_library(${TARGET}_gpu${target_suffix} SRCS ${gpu_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
        endif()
        list(APPEND base_device_kernels ${TARGET}_gpu${target_suffix})
    endif()
    if (${xpu_srcs_len} GREATER 0)
        cc_library(${TARGET}_xpu${target_suffix} SRCS ${xpu_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
        list(APPEND base_device_kernels ${TARGET}_xpu${target_suffix})
    endif()
    if (${gpudnn_srcs_len} GREATER 0)
        if (WITH_GPU)
            nv_library(${TARGET}_gpudnn${target_suffix} SRCS ${gpudnn_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
        elseif (WITH_ROCM)
            hip_library(${TARGET}_gpudnn${target_suffix} SRCS ${gpudnn_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
        endif()
        list(APPEND base_device_kernels ${TARGET}_gpudnn${target_suffix})
    endif()
    if (${kps_srcs_len} GREATER 0)
        # only when WITH_XPU_KP, the kps_srcs_len can be > 0
        xpu_library(${TARGET}_kps${target_suffix} SRCS ${kps_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
        list(APPEND base_device_kernels ${TARGET}_kps${target_suffix})
    endif()

    # 2. Device-independent kernel compile
    if (${common_srcs_len} GREATER 0)
        if (WITH_GPU)
            nv_library(${TARGET}_common${target_suffix} SRCS ${common_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps} ${base_device_kernels})
        elseif (WITH_ROCM)
            hip_library(${TARGET}_common${target_suffix} SRCS ${common_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps} ${base_device_kernels})
        elseif (WITH_XPU_KP)
            xpu_library(${TARGET}_common${target_suffix} SRCS ${common_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps} ${base_device_kernels})
        else()
            cc_library(${TARGET}_common${target_suffix} SRCS ${common_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps} ${base_device_kernels})
        endif()
        list(APPEND device_independent_kernel ${TARGET}_common${target_suffix})
    endif()


    # 3. Unify target compile
    list(LENGTH base_device_kernels base_device_kernels_len)
    list(LENGTH device_independent_kernel device_independent_kernel_len)
    if (${base_device_kernels_len} GREATER 0 OR ${device_independent_kernel_len} GREATER 0)
        if (WITH_GPU)
            nv_library(${TARGET}${target_suffix} DEPS ${kernel_library_DEPS} ${kernel_deps} ${base_device_kernels} ${device_independent_kernel})
        elseif (WITH_ROCM)
            hip_library(${TARGET}${target_suffix} DEPS ${kernel_library_DEPS} ${kernel_deps} ${base_device_kernels} ${device_independent_kernel})
        elseif (WITH_XPU_KP)
            xpu_library(${TARGET}${target_suffix} DEPS ${kernel_library_DEPS} ${kernel_deps} ${base_device_kernels} ${device_independent_kernel})
        else()
            cc_library(${TARGET}${target_suffix} DEPS ${kernel_library_DEPS} ${kernel_deps} ${base_device_kernels} ${device_independent_kernel})
        endif()
    else()
        set(target_build_flag 0)
    endif()

    if (${target_build_flag} EQUAL 1)
        if (${common_srcs_len} GREATER 0 OR ${cpu_srcs_len} GREATER 0 OR
            ${gpu_srcs_len} GREATER 0 OR ${xpu_srcs_len} GREATER 0 OR ${kps_srcs_len} GREATER 0 OR
            ${gpudnn_srcs_len} GREATER 0)
            # append target into PHI_KERNELS property
            get_property(phi_kernels GLOBAL PROPERTY PHI_KERNELS)
            set(phi_kernels ${phi_kernels} ${TARGET}${target_suffix})
            set_property(GLOBAL PROPERTY PHI_KERNELS ${phi_kernels})
        endif()

        # parse kernel name and auto generate kernel declaration
        # here, we don't need to check WITH_XXX, because if not WITH_XXX, the
        # xxx_srcs_len will be equal to 0
        if (${common_srcs_len} GREATER 0)
            kernel_declare(${common_srcs})
        endif()
        if (${cpu_srcs_len} GREATER 0)
            kernel_declare(${cpu_srcs})
        endif()
        if (${gpu_srcs_len} GREATER 0)
            kernel_declare(${gpu_srcs})
        endif()
        if (${xpu_srcs_len} GREATER 0)
            kernel_declare(${xpu_srcs})
        endif()
        if (${gpudnn_srcs_len} GREATER 0)
            kernel_declare(${gpudnn_srcs})
        endif()
        if (${kps_srcs_len} GREATER 0)
            kernel_declare(${kps_srcs})
        endif()
    endif()
endfunction()

function(register_kernels)
    set(options "")
    set(oneValueArgs SUB_DIR)
    set(multiValueArgs EXCLUDES DEPS)
    cmake_parse_arguments(register_kernels "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    file(GLOB KERNELS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_kernel.h")
    string(REPLACE ".h" "" KERNELS "${KERNELS}")
    list(LENGTH register_kernels_DEPS register_kernels_DEPS_len)

    foreach(target ${KERNELS})
        list(FIND register_kernels_EXCLUDES ${target} _index)
        if (${_index} EQUAL -1)
            if (${register_kernels_DEPS_len} GREATER 0)
                kernel_library(${target} DEPS ${register_kernels_DEPS} SUB_DIR ${register_kernels_SUB_DIR})
            else()
                kernel_library(${target} SUB_DIR ${register_kernels_SUB_DIR})
            endif()
        endif()
    endforeach()
endfunction()

function(append_op_util_declare TARGET)
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET} target_content)
    string(REGEX MATCH "(PD_REGISTER_BASE_KERNEL_NAME|PD_REGISTER_ARG_MAPPING_FN)\\([ \t\r\n]*[a-z0-9_]*" util_registrar "${target_content}")
    string(REPLACE "PD_REGISTER_ARG_MAPPING_FN" "PD_DECLARE_ARG_MAPPING_FN" util_declare "${util_registrar}")
    string(REPLACE "PD_REGISTER_BASE_KERNEL_NAME" "PD_DECLARE_BASE_KERNEL_NAME" util_declare "${util_declare}")
    string(APPEND util_declare ");\n")
    file(APPEND ${op_utils_header} "${util_declare}")
endfunction()

function(register_op_utils TARGET_NAME)
    set(utils_srcs)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs EXCLUDES DEPS)
    cmake_parse_arguments(register_op_utils "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    file(GLOB SIGNATURES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_sig.cc")
    foreach(target ${SIGNATURES})
        append_op_util_declare(${target})
        list(APPEND utils_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${target})
    endforeach()

    cc_library(${TARGET_NAME} SRCS ${utils_srcs} DEPS ${register_op_utils_DEPS})
endfunction()
