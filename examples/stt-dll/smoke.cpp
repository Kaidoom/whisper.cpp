#include "WhisperSTT.h"

#include <iostream>
#include <string>

int wmain(int argc, wchar_t ** argv) {
    if (argc < 3) {
        std::wcerr << L"usage: whisper-stt-smoke.exe <model> <wav> [init-options] [call-options]\n";
        return 2;
    }

    const wchar_t * init_options = argc > 3 ? argv[3] : L"-nt -np";
    const wchar_t * call_options = argc > 4 ? argv[4] : L"";

    if (!WhisperSTT_Init(argv[1], init_options)) {
        std::wcerr << L"WhisperSTT_Init failed: " << WhisperSTT_GetLastError() << L"\n";
        return 1;
    }

    char * text = nullptr;
    int size = 0;
    if (!WhisperSTT_TranscribeFileUtf8(argv[2], call_options, &text, &size)) {
        std::wcerr << L"WhisperSTT_TranscribeFileUtf8 failed: " << WhisperSTT_GetLastError() << L"\n";
        return 1;
    }

    std::cout.write(text, size);
    std::cout << "\n";

    WhisperSTT_FreeBuffer(text);
    WhisperSTT_Shutdown();
    return 0;
}
