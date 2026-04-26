set(TARGET whisper-stt)

add_library(${TARGET} SHARED
    WhisperSTT.cpp
    WhisperSTT.h
    )

include(DefaultTargetOptions)

target_compile_definitions(${TARGET} PRIVATE WHISPER_STT_EXPORTS)
target_link_libraries(${TARGET} PRIVATE common whisper ${FFMPEG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

set_target_properties(${TARGET} PROPERTIES FOLDER "examples")

if (WIN32)
    set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME "whisper-stt")
endif()

set(SMOKE_TARGET whisper-stt-smoke)

add_executable(${SMOKE_TARGET}
    smoke.cpp
    )

include(DefaultTargetOptions)

target_link_libraries(${SMOKE_TARGET} PRIVATE ${TARGET})
set_target_properties(${SMOKE_TARGET} PROPERTIES FOLDER "examples")

install(TARGETS ${TARGET} RUNTIME)
