# vk-video-caps: standalone Vulkan Video capability dumper.
# Only the two .cpp files below are needed to link VulkanDeviceContext; everything else
# used (VulkanVideoCapabilities, VkVideoCoreProfile, Helpers) is header-only.

set(sources
    main.cpp
    ${VK_VIDEO_COMMON_LIBS_SOURCE_ROOT}/VkCodecUtils/HelpersDispatchTable.cpp
    ${VK_VIDEO_COMMON_LIBS_SOURCE_ROOT}/VkCodecUtils/HelpersDispatchTable.h
    ${VK_VIDEO_COMMON_LIBS_SOURCE_ROOT}/VkCodecUtils/VulkanDeviceContext.cpp
    ${VK_VIDEO_COMMON_LIBS_SOURCE_ROOT}/VkCodecUtils/VulkanDeviceContext.h
)

set(definitions PRIVATE -DVK_NO_PROTOTYPES)

set(includes
    PRIVATE ${VK_VIDEO_COMMON_LIBS_SOURCE_ROOT}
    PRIVATE ${VULKAN_VIDEO_APIS_INCLUDE}
    PRIVATE ${VULKAN_VIDEO_APIS_INCLUDE}/vulkan
    PRIVATE ${VULKAN_VIDEO_APIS_INCLUDE}/nvidia_utils/vulkan
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)

set(libraries PRIVATE ${CMAKE_THREAD_LIBS_INIT})

if(TARGET vulkan)
    list(APPEND definitions PRIVATE -DUNINSTALLED_LOADER="$<TARGET_FILE:vulkan>")
endif()

if(WIN32)
    list(APPEND definitions PRIVATE -DVK_USE_PLATFORM_WIN32_KHR -DWIN32_LEAN_AND_MEAN)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    list(APPEND libraries PRIVATE -ldl -lrt -lpthread)
endif()

add_executable(vk-video-caps ${sources})
set_target_properties(vk-video-caps PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(vk-video-caps ${definitions})
target_include_directories(vk-video-caps ${includes})
target_link_libraries(vk-video-caps PUBLIC ${libraries})
add_dependencies(vk-video-caps GenerateDispatchTables)

install(TARGETS vk-video-caps RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
