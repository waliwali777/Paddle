set(PART_CUDA_KERNEL_FILES)
function(op_library TARGET)
    # op_library is a function to create op library. The interface is same as
    # cc_library. But it handle split GPU/CPU code and link some common library
    # for ops.
    set(cc_srcs)
    set(cu_srcs)
    set(hip_cu_srcs)
    set(miopen_hip_cc_srcs)
    set(cu_cc_srcs)
    set(cudnn_cu_cc_srcs)
    set(CUDNN_FILE)
    set(mkldnn_cc_srcs)
    set(MKLDNN_FILE)
    set(op_common_deps operator op_registry math_function)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    set(pybind_flag 0)
    cmake_parse_arguments(op_library "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN})

    list(LENGTH op_library_SRCS op_library_SRCS_len)
    if (${op_library_SRCS_len} EQUAL 0)
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cc)
            list(APPEND cc_srcs ${TARGET}.cc)
        endif()
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cu.cc)
            list(APPEND cu_cc_srcs ${TARGET}.cu.cc)
        endif()
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cu)
            list(APPEND cu_srcs ${TARGET}.cu)
        endif()
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.part.cu)
            set(PART_CUDA_KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.part.cu
                    ${PART_CUDA_KERNEL_FILES} PARENT_SCOPE)
            list(APPEND cu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.part.cu)
        endif()

        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.hip.cu)
            list(APPEND hip_cu_srcs ${TARGET}.hip.cu)
        endif()
        string(REPLACE "_op" "_cudnn_op" CUDNN_FILE "${TARGET}")
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CUDNN_FILE}.cu.cc)
            list(APPEND cudnn_cu_cc_srcs ${CUDNN_FILE}.cu.cc)
        endif()
        if(WITH_AMD_GPU)
            string(REPLACE "_op" "_miopen_op" MIOPEN_FILE "${TARGET}")
            if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${MIOPEN_FILE}.hip.cc)
                list(APPEND miopen_hip_cc_srcs ${MIOPEN_FILE}.hip.cc)
            endif()
        endif()
        if(WITH_MKLDNN)
            string(REPLACE "_op" "_mkldnn_op" MKLDNN_FILE "${TARGET}")
            if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${MKLDNN_FILE}.cc)
                list(APPEND mkldnn_cc_srcs ${MKLDNN_FILE}.cc)
            endif()
        endif()
    else()
        foreach(src ${op_library_SRCS})
            if (${src} MATCHES ".*\\.hip.cu$")
                list(APPEND hip_cu_srcs ${src})
            elseif (${src} MATCHES ".*\\.cu$")
                list(APPEND cu_srcs ${src})
            elseif(${src} MATCHES ".*_cudnn_op.cu.cc$")
                list(APPEND cudnn_cu_cc_srcs ${src})
            elseif(WITH_AMD_GPU AND ${src} MATCHES ".*_miopen_op.hip.cc$")
                list(APPEND miopen_hip_cc_srcs ${src})
            elseif(WITH_MKLDNN AND ${src} MATCHES ".*_mkldnn_op.cc$")
                list(APPEND mkldnn_cc_srcs ${src})
            elseif(${src} MATCHES ".*\\.cu.cc$")
                list(APPEND cu_cc_srcs ${src})
            elseif(${src} MATCHES ".*\\.cc$")
                list(APPEND cc_srcs ${src})
            else()
                message(FATAL_ERROR "${TARGET} Source file ${src} should only be .cc or .cu")
            endif()
        endforeach()
    endif()

    list(LENGTH cc_srcs cc_srcs_len)
    if (${cc_srcs_len} EQUAL 0)
        message(FATAL_ERROR "The op library ${TARGET} should contains at least one .cc file")
    endif()
    if (WIN32)
    # remove windows unsupported op, because windows has no nccl, no warpctc such ops.
    foreach(windows_unsupport_op "nccl_op" "gen_nccl_id_op" "warpctc_op"
            "channel_send_op" "channel_create_op" "channel_close_op" "channel_recv_op")
        if ("${TARGET}" STREQUAL "${windows_unsupport_op}")
          return()
        endif()
    endforeach()
    endif(WIN32)
    set(OP_LIBRARY ${TARGET} ${OP_LIBRARY} CACHE INTERNAL "op libs")

    list(LENGTH op_library_DEPS op_library_DEPS_len)
    if (${op_library_DEPS_len} GREATER 0)
        set(DEPS_OPS ${TARGET} ${DEPS_OPS} PARENT_SCOPE)
    endif()
    if (WITH_GPU)
        nv_library(${TARGET} SRCS ${cc_srcs} ${cu_cc_srcs} ${cudnn_cu_cc_srcs} ${mkldnn_cc_srcs} ${cu_srcs} DEPS ${op_library_DEPS}
                ${op_common_deps})
    elseif (WITH_AMD_GPU)
        hip_library(${TARGET} SRCS ${cc_srcs} ${hip_cu_srcs} ${miopen_hip_cc_srcs} ${mkldnn_cc_srcs} DEPS ${op_library_DEPS}
                ${op_common_deps})
    else()
        cc_library(${TARGET} SRCS ${cc_srcs} ${mkldnn_cc_srcs} DEPS ${op_library_DEPS}
            ${op_common_deps})
    endif()

    # Define operators that don't need pybind here.
    foreach(manual_pybind_op "compare_op" "logical_op" "nccl_op"
"tensor_array_read_write_op" "tensorrt_engine_op")
        if ("${TARGET}" STREQUAL "${manual_pybind_op}")
            set(pybind_flag 1)
        endif()
    endforeach()

    # The registration of USE_OP, please refer to paddle/fluid/framework/op_registry.h.
    # Note that it's enough to just adding one operator to pybind in a *_op.cc file.
    # And for detail pybind information, please see generated paddle/pybind/pybind.h.
    file(READ ${TARGET}.cc TARGET_CONTENT)
    string(REGEX MATCH "REGISTER_OPERATOR\\(.*REGISTER_OPERATOR\\(" multi_register "${TARGET_CONTENT}")
    string(REGEX MATCH "REGISTER_OPERATOR\\([a-z0-9_]*," one_register "${multi_register}")
    if (one_register STREQUAL "")
        string(REPLACE "_op" "" TARGET "${TARGET}")
    else ()
        string(REPLACE "REGISTER_OPERATOR(" "" TARGET "${one_register}")
        string(REPLACE "," "" TARGET "${TARGET}")
    endif()

    # pybind USE_NO_KERNEL_OP
    # HACK: if REGISTER_OP_CPU_KERNEL presents the operator must have kernel
    string(REGEX MATCH "REGISTER_OP_CPU_KERNEL" regex_result "${TARGET_CONTENT}")
    string(REPLACE "_op" "" TARGET "${TARGET}")
    if (${pybind_flag} EQUAL 0 AND regex_result STREQUAL "")
        file(APPEND ${pybind_file} "USE_NO_KERNEL_OP(${TARGET});\n")
        set(pybind_flag 1)
    endif()

    # pybind USE_CPU_ONLY_OP
    list(LENGTH cu_srcs cu_srcs_len)
    list(LENGTH cu_cc_srcs cu_cc_srcs_len)
    list(LENGTH mkldnn_cc_srcs mkldnn_cc_srcs_len)
    list(LENGTH hip_cu_srcs hip_cu_srcs_len)
    list(LENGTH miopen_hip_cc_srcs miopen_hip_cc_srcs_len)
    if (${pybind_flag} EQUAL 0 AND ${mkldnn_cc_srcs_len} EQUAL 0 AND ${cu_srcs_len} EQUAL 0 AND ${cu_cc_srcs_len} EQUAL 0 AND
        ${hip_cu_srcs_len} EQUAL 0 AND ${miopen_hip_cc_srcs_len} EQUAL 0)
        file(APPEND ${pybind_file} "USE_CPU_ONLY_OP(${TARGET});\n")
        set(pybind_flag 1)
    endif()

    # pybind USE_OP_DEVICE_KERNEL for CUDNN
    list(LENGTH cudnn_cu_cc_srcs cudnn_cu_cc_srcs_len)
    if (WITH_GPU AND ${cudnn_cu_cc_srcs_len} GREATER 0)
        file(APPEND ${pybind_file} "USE_OP_DEVICE_KERNEL(${TARGET}, CUDNN);\n")
    endif()

    # pybind USE_OP_DEVICE_KERNEL for MIOPEN
    if (WITH_AMD_GPU AND ${miopen_hip_cc_srcs_len} GREATER 0)
        file(APPEND ${pybind_file} "USE_OP_DEVICE_KERNEL(${TARGET}, MIOPEN);\n")
    endif()

    # pybind USE_OP_DEVICE_KERNEL for MKLDNN
    if (WITH_MKLDNN AND ${mkldnn_cc_srcs_len} GREATER 0)
      # Append first implemented MKLDNN activation operator
      if (${MKLDNN_FILE} STREQUAL "activation_mkldnn_op")
        file(APPEND ${pybind_file} "USE_OP_DEVICE_KERNEL(relu, MKLDNN);\n")
      else()
        file(APPEND ${pybind_file} "USE_OP_DEVICE_KERNEL(${TARGET}, MKLDNN);\n")
      endif()
    endif()

    # pybind USE_OP
    if (${pybind_flag} EQUAL 0)
      # NOTE(*): activation use macro to regist the kernels, set use_op manually.
      if(${TARGET} STREQUAL "activation")
        file(APPEND ${pybind_file} "USE_OP(relu);\n")
      elseif(${TARGET} STREQUAL "fake_dequantize")
        file(APPEND ${pybind_file} "USE_OP(fake_dequantize_max_abs);\n")
      elseif(${TARGET} STREQUAL "fake_quantize")
        file(APPEND ${pybind_file} "USE_OP(fake_quantize_abs_max);\n")
      elseif(${TARGET} STREQUAL "tensorrt_engine_op")
          message(STATUS "Pybind skips [tensorrt_engine_op], for this OP is only used in inference")
      elseif(${TARGET} STREQUAL "fc")
        # HACK: fc only have mkldnn and cpu, which would mismatch the cpu only condition
        file(APPEND ${pybind_file} "USE_CPU_ONLY_OP(${TARGET});\n")
      else()
        file(APPEND ${pybind_file} "USE_OP(${TARGET});\n")
      endif()
    endif()
endfunction()


function(register_operators)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs EXCLUDES)
    cmake_parse_arguments(register_operators "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN})

    file(GLOB OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_op.cc")
    string(REPLACE "_mkldnn" "" OPS "${OPS}")
    string(REPLACE ".cc" "" OPS "${OPS}")
    list(REMOVE_DUPLICATES OPS)

    foreach(src ${OPS})
        list(FIND register_operators_EXCLUDES ${src} _index)
        if (${_index} EQUAL -1)
            op_library(${src})
        endif()
    endforeach()
endfunction()
