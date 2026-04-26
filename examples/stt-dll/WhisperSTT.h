#pragma once

#include <stdint.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#ifdef WHISPER_STT_EXPORTS
#define WHISPER_STT_API __declspec(dllexport)
#else
#define WHISPER_STT_API __declspec(dllimport)
#endif

extern "C"
{
    WHISPER_STT_API int WhisperSTT_Ping();

    WHISPER_STT_API int WhisperSTT_Init(const wchar_t* modelPath, const wchar_t* cliOptions);
    WHISPER_STT_API int WhisperSTT_TranscribeFileUtf8(const wchar_t* wavPath, const wchar_t* cliOptionsOverride, char** outUtf8, int* outSize);

    WHISPER_STT_API void WhisperSTT_FreeBuffer(void* buffer);
    WHISPER_STT_API void WhisperSTT_Shutdown();
    WHISPER_STT_API const wchar_t* WhisperSTT_GetLastError();
}
