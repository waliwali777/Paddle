if (NOT WITH_ANAKIN)
  return()
endif()

set(ANAKIN_INSTALL_DIR "${THIRD_PARTY_PATH}/install/anakin" CACHE PATH
  "Anakin install path." FORCE)
set(ANAKIN_INCLUDE "${ANAKIN_INSTALL_DIR}" CACHE STRING "root of Anakin header files")
set(ANAKIN_LIBRARY "${ANAKIN_INSTALL_DIR}" CACHE STRING "path of Anakin library")

set(ANAKIN_COMPILE_EXTRA_FLAGS 
    -Wno-error=unused-but-set-variable -Wno-unused-but-set-variable
    -Wno-error=unused-variable -Wno-unused-variable 
    -Wno-error=format-extra-args -Wno-format-extra-args
    -Wno-error=comment -Wno-comment 
    -Wno-error=format -Wno-format 
    -Wno-error=maybe-uninitialized
    -Wno-error=switch -Wno-switch
    -Wno-error=return-type -Wno-return-type 
    -Wno-error=non-virtual-dtor -Wno-non-virtual-dtor
    -Wno-error=ignored-qualifiers
    -Wno-sign-compare
    -Wno-reorder 
    -Wno-error=cpp)

set(ANAKIN_LIBRARY_URL "https://github.com/pangge/Anakin/releases/download/Version0.1.0/anakin.tar.gz")

# A helper function used in Anakin, currently, to use it, one need to recursively include
# nearly all the header files.
function(fetch_include_recursively root_dir)
    if (IS_DIRECTORY ${root_dir})
        include_directories(${root_dir})
    endif()

    file(GLOB ALL_SUB RELATIVE ${root_dir} ${root_dir}/*)
    foreach(sub ${ALL_SUB})
        if (IS_DIRECTORY ${root_dir}/${sub})
            fetch_include_recursively(${root_dir}/${sub})
        endif()
    endforeach()
endfunction()

if (NOT EXISTS "${ANAKIN_INSTALL_DIR}")
    # download library
    message(STATUS "Download Anakin library from ${ANAKIN_LIBRARY_URL}")
    execute_process(COMMAND bash -c "mkdir -p ${ANAKIN_INSTALL_DIR}")
    execute_process(COMMAND bash -c "rm -rf ${ANAKIN_INSTALL_DIR}/*")
    execute_process(COMMAND bash -c "cd ${ANAKIN_INSTALL_DIR}; wget --no-check-certificate -q ${ANAKIN_LIBRARY_URL}")
    execute_process(COMMAND bash -c "mkdir -p ${ANAKIN_INSTALL_DIR}")
    execute_process(COMMAND bash -c "cd ${ANAKIN_INSTALL_DIR}; tar xzf anakin.tar.gz")
endif()

if (WITH_ANAKIN)
    message(STATUS "Anakin for inference is enabled")
    message(STATUS "Anakin is set INCLUDE:${ANAKIN_INCLUDE} LIBRARY:${ANAKIN_LIBRARY}")
    fetch_include_recursively(${ANAKIN_INCLUDE})
    link_directories(${ANAKIN_LIBRARY})
endif()
