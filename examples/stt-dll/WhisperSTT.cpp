#include "WhisperSTT.h"

#include "common-whisper.h"
#include "ggml-backend.h"
#include "whisper.h"

#include <algorithm>
#include <cfloat>
#include <cctype>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct stt_options {
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors  = 1;
    int32_t offset_t_ms   = 0;
    int32_t duration_ms   = 0;
    int32_t max_context   = -1;
    int32_t best_of       = whisper_full_default_params(WHISPER_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size     = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    int32_t audio_ctx     = 0;

    float word_thold      =  0.01f;
    float entropy_thold   =  2.40f;
    float logprob_thold   = -1.00f;
    float no_speech_thold =  0.6f;
    float temperature     = 0.0f;
    float temperature_inc = 0.2f;

    bool translate       = false;
    bool detect_language = false;
    bool no_fallback     = false;
    bool no_prints       = false;
    bool print_special   = false;
    bool print_progress  = false;
    bool no_timestamps   = false;
    bool use_gpu         = true;
    bool flash_attn      = true;
    bool suppress_nst    = false;
    bool keep_model      = false;

    std::string language = "en";
    std::string prompt;
    std::string model;
    std::string suppress_regex;
    std::string openvino_encode_device = "CPU";

    bool        vad           = false;
    std::string vad_model;
    float       vad_threshold = 0.5f;
    int         vad_min_speech_duration_ms = 250;
    int         vad_min_silence_duration_ms = 100;
    float       vad_max_speech_duration_s = FLT_MAX;
    int         vad_speech_pad_ms = 30;
    float       vad_samples_overlap = 0.1f;

    int gpu_device = 0;
};

std::mutex g_mutex;
std::wstring g_last_error;
std::string g_model_path;
stt_options g_default_options;
whisper_context * g_kept_context = nullptr;
bool g_backends_loaded = false;

void cb_log_disable(enum ggml_log_level, const char *, void *) {
}

void set_error(const std::wstring & message) {
    g_last_error = message;
}

std::wstring widen_ascii(const std::string & text) {
    return std::wstring(text.begin(), text.end());
}

std::string wide_to_utf8(const wchar_t * value) {
    if (value == nullptr || value[0] == L'\0') {
        return {};
    }

    const int needed = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (needed <= 0) {
        throw std::runtime_error("failed to convert UTF-16 text to UTF-8");
    }

    std::string result((size_t) needed - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, &result[0], needed, nullptr, nullptr);
    return result;
}

std::vector<std::string> split_options(const std::string & text) {
    std::vector<std::string> result;
    std::string current;
    char quote = 0;
    bool escaping = false;

    for (char ch : text) {
        if (escaping) {
            current.push_back(ch);
            escaping = false;
            continue;
        }

        if (ch == '\\' && quote != 0) {
            escaping = true;
            continue;
        }

        if ((ch == '\'' || ch == '"')) {
            if (quote == 0) {
                quote = ch;
                continue;
            }
            if (quote == ch) {
                quote = 0;
                continue;
            }
        }

        if (quote == 0 && (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n')) {
            if (!current.empty()) {
                result.push_back(current);
                current.clear();
            }
            continue;
        }

        current.push_back(ch);
    }

    if (escaping) {
        current.push_back('\\');
    }
    if (quote != 0) {
        throw std::runtime_error("unterminated quote in cli options");
    }
    if (!current.empty()) {
        result.push_back(current);
    }

    return result;
}

std::string require_value(const std::vector<std::string> & args, size_t & i, const std::string & arg) {
    if (i + 1 >= args.size()) {
        throw std::runtime_error("argument " + arg + " requires value");
    }
    return args[++i];
}

void parse_options(const std::string & text, stt_options & options) {
    const std::vector<std::string> args = split_options(text);

    for (size_t i = 0; i < args.size(); ++i) {
        const std::string & arg = args[i];

        if (arg == "-t"    || arg == "--threads")              { options.n_threads       = std::stoi(require_value(args, i, arg)); }
        else if (arg == "-p"    || arg == "--processors")      { options.n_processors    = std::stoi(require_value(args, i, arg)); }
        else if (arg == "-ot"   || arg == "--offset-t")        { options.offset_t_ms     = std::stoi(require_value(args, i, arg)); }
        else if (arg == "-d"    || arg == "--duration")        { options.duration_ms     = std::stoi(require_value(args, i, arg)); }
        else if (arg == "-mc"   || arg == "--max-context")     { options.max_context     = std::stoi(require_value(args, i, arg)); }
        else if (arg == "-bo"   || arg == "--best-of")         { options.best_of         = std::stoi(require_value(args, i, arg)); }
        else if (arg == "-bs"   || arg == "--beam-size")       { options.beam_size       = std::stoi(require_value(args, i, arg)); }
        else if (arg == "-ac"   || arg == "--audio-ctx")       { options.audio_ctx       = std::stoi(require_value(args, i, arg)); }
        else if (arg == "-wt"   || arg == "--word-thold")      { options.word_thold      = std::stof(require_value(args, i, arg)); }
        else if (arg == "-et"   || arg == "--entropy-thold")   { options.entropy_thold   = std::stof(require_value(args, i, arg)); }
        else if (arg == "-lpt"  || arg == "--logprob-thold")   { options.logprob_thold   = std::stof(require_value(args, i, arg)); }
        else if (arg == "-nth"  || arg == "--no-speech-thold") { options.no_speech_thold = std::stof(require_value(args, i, arg)); }
        else if (arg == "-tp"   || arg == "--temperature")     { options.temperature     = std::stof(require_value(args, i, arg)); }
        else if (arg == "-tpi"  || arg == "--temperature-inc") { options.temperature_inc = std::stof(require_value(args, i, arg)); }
        else if (arg == "-tr"   || arg == "--translate")       { options.translate       = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")     { options.no_fallback     = true; }
        else if (arg == "-np"   || arg == "--no-prints")       { options.no_prints       = true; }
        else if (arg == "-ps"   || arg == "--print-special")   { options.print_special   = true; }
        else if (arg == "-pp"   || arg == "--print-progress")  { options.print_progress  = true; }
        else if (arg == "-nt"   || arg == "--no-timestamps")   { options.no_timestamps   = true; }
        else if (arg == "-l"    || arg == "--language")        {
            options.language = require_value(args, i, arg);
            std::transform(options.language.begin(), options.language.end(), options.language.begin(), [](unsigned char c) { return (char) std::tolower(c); });
        }
        else if (arg == "-dl"   || arg == "--detect-language") { options.detect_language = true; }
        else if (arg == "--prompt")                            { options.prompt          = require_value(args, i, arg); }
        else if (arg == "-m"    || arg == "--model")           { options.model           = require_value(args, i, arg); }
        else if (arg == "-oved" || arg == "--ov-e-device")     { options.openvino_encode_device = require_value(args, i, arg); }
        else if (arg == "-ng"   || arg == "--no-gpu")          { options.use_gpu         = false; }
        else if (arg == "-dev"  || arg == "--device")          { options.gpu_device      = std::stoi(require_value(args, i, arg)); }
        else if (arg == "-fa"   || arg == "--flash-attn")      { options.flash_attn      = true; }
        else if (arg == "-nfa"  || arg == "--no-flash-attn")   { options.flash_attn      = false; }
        else if (arg == "-sns"  || arg == "--suppress-nst")    { options.suppress_nst    = true; }
        else if (arg == "--suppress-regex")                    { options.suppress_regex  = require_value(args, i, arg); }
        else if (arg == "--keep-model")                        { options.keep_model      = true; }
        else if (arg == "--vad")                               { options.vad             = true; }
        else if (arg == "-vm"   || arg == "--vad-model")       { options.vad_model       = require_value(args, i, arg); }
        else if (arg == "-vt"   || arg == "--vad-threshold")   { options.vad_threshold   = std::stof(require_value(args, i, arg)); }
        else if (arg == "-vspd" || arg == "--vad-min-speech-duration-ms")  { options.vad_min_speech_duration_ms  = std::stoi(require_value(args, i, arg)); }
        else if (arg == "-vsd"  || arg == "--vad-min-silence-duration-ms") { options.vad_min_silence_duration_ms = std::stoi(require_value(args, i, arg)); }
        else if (arg == "-vmsd" || arg == "--vad-max-speech-duration-s")   { options.vad_max_speech_duration_s   = std::stof(require_value(args, i, arg)); }
        else if (arg == "-vp"   || arg == "--vad-speech-pad-ms")           { options.vad_speech_pad_ms           = std::stoi(require_value(args, i, arg)); }
        else if (arg == "-vo"   || arg == "--vad-samples-overlap")         { options.vad_samples_overlap         = std::stof(require_value(args, i, arg)); }
        else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
}

bool validate_options(const stt_options & options, std::wstring & error) {
    if (options.n_threads <= 0) {
        error = L"Thread count must be greater than zero.";
        return false;
    }
    if (options.n_processors <= 0) {
        error = L"Processor count must be greater than zero.";
        return false;
    }
    if (options.language != "auto" && whisper_lang_id(options.language.c_str()) == -1) {
        error = L"Unknown language: " + widen_ascii(options.language);
        return false;
    }
    return true;
}

whisper_context_params make_context_params(const stt_options & options) {
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = options.use_gpu;
    cparams.flash_attn = options.flash_attn;
    cparams.gpu_device = options.gpu_device;
    return cparams;
}

whisper_context * load_context(const std::string & model_path, const stt_options & options, std::wstring & error) {
    whisper_context * ctx = whisper_init_from_file_with_params(model_path.c_str(), make_context_params(options));
    if (ctx == nullptr) {
        error = L"Failed to initialize whisper context.";
        return nullptr;
    }

    whisper_ctx_init_openvino_encoder(ctx, nullptr, options.openvino_encode_device.c_str(), nullptr);
    return ctx;
}

whisper_full_params make_full_params(const stt_options & options) {
    whisper_full_params wparams = whisper_full_default_params(
        options.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

    wparams.strategy         = options.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;
    wparams.print_realtime   = false;
    wparams.print_progress   = options.print_progress;
    wparams.print_timestamps = !options.no_timestamps;
    wparams.print_special    = options.print_special;
    wparams.translate        = options.translate;
    wparams.language         = options.language.c_str();
    wparams.detect_language  = options.detect_language;
    wparams.n_threads        = options.n_threads;
    wparams.n_max_text_ctx   = options.max_context >= 0 ? options.max_context : wparams.n_max_text_ctx;
    wparams.offset_ms        = options.offset_t_ms;
    wparams.duration_ms      = options.duration_ms;
    wparams.token_timestamps = false;
    wparams.thold_pt         = options.word_thold;
    wparams.audio_ctx        = options.audio_ctx;
    wparams.suppress_regex   = options.suppress_regex.empty() ? nullptr : options.suppress_regex.c_str();
    wparams.initial_prompt   = options.prompt.c_str();
    wparams.greedy.best_of        = options.best_of;
    wparams.beam_search.beam_size = options.beam_size;
    wparams.temperature_inc  = options.no_fallback ? 0.0f : options.temperature_inc;
    wparams.temperature      = options.temperature;
    wparams.entropy_thold    = options.entropy_thold;
    wparams.logprob_thold    = options.logprob_thold;
    wparams.no_speech_thold  = options.no_speech_thold;
    wparams.no_timestamps    = options.no_timestamps;
    wparams.suppress_nst     = options.suppress_nst;
    wparams.vad              = options.vad;
    wparams.vad_model_path   = options.vad_model.empty() ? nullptr : options.vad_model.c_str();
    wparams.vad_params.threshold               = options.vad_threshold;
    wparams.vad_params.min_speech_duration_ms  = options.vad_min_speech_duration_ms;
    wparams.vad_params.min_silence_duration_ms = options.vad_min_silence_duration_ms;
    wparams.vad_params.max_speech_duration_s   = options.vad_max_speech_duration_s;
    wparams.vad_params.speech_pad_ms           = options.vad_speech_pad_ms;
    wparams.vad_params.samples_overlap         = options.vad_samples_overlap;

    return wparams;
}

bool copy_utf8_to_heap_buffer(const std::string & text, char ** out_utf8, int * out_size, std::wstring & error) {
    if (out_utf8 == nullptr || out_size == nullptr) {
        error = L"Output pointers are null.";
        return false;
    }

    *out_utf8 = nullptr;
    *out_size = 0;

    if (text.size() > (size_t) INT32_MAX) {
        error = L"Output text is too large.";
        return false;
    }

    char * heap_text = nullptr;
    try {
        heap_text = new char[text.size() + 1];
    } catch (const std::bad_alloc &) {
        error = L"Failed to allocate UTF-8 output buffer.";
        return false;
    }

    if (!text.empty()) {
        memcpy(heap_text, text.data(), text.size());
    }
    heap_text[text.size()] = '\0';

    *out_utf8 = heap_text;
    *out_size = (int) text.size();
    return true;
}

bool ensure_backends_loaded(std::wstring & error) {
    if (g_backends_loaded) {
        return true;
    }

    try {
        ggml_backend_load_all();
    } catch (...) {
        error = L"Failed to load ggml backends.";
        return false;
    }

    g_backends_loaded = true;
    return true;
}

void free_kept_context() {
    if (g_kept_context != nullptr) {
        whisper_free(g_kept_context);
        g_kept_context = nullptr;
    }
}

}

extern "C" {

int WhisperSTT_Ping() {
    return 1;
}

int WhisperSTT_Init(const wchar_t* modelPath, const wchar_t* cliOptions) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_last_error.clear();

    try {
        std::string model_path = wide_to_utf8(modelPath);
        if (model_path.empty()) {
            set_error(L"Model path is required.");
            return 0;
        }

        stt_options options;
        parse_options(wide_to_utf8(cliOptions), options);
        if (!options.model.empty()) {
            model_path = options.model;
        }

        std::wstring error;
        if (!validate_options(options, error)) {
            set_error(error);
            return 0;
        }
        if (options.no_prints) {
            whisper_log_set(cb_log_disable, nullptr);
        }
        if (!ensure_backends_loaded(error)) {
            set_error(error);
            return 0;
        }

        free_kept_context();
        g_model_path = model_path;
        g_default_options = options;

        if (g_default_options.keep_model) {
            g_kept_context = load_context(g_model_path, g_default_options, error);
            if (g_kept_context == nullptr) {
                set_error(error);
                return 0;
            }
        }

        return 1;
    } catch (const std::exception & ex) {
        set_error(L"WhisperSTT_Init failed: " + widen_ascii(ex.what()));
        return 0;
    } catch (...) {
        set_error(L"WhisperSTT_Init failed.");
        return 0;
    }
}

int WhisperSTT_TranscribeFileUtf8(const wchar_t* wavPath, const wchar_t* cliOptionsOverride, char** outUtf8, int* outSize) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_last_error.clear();

    if (outUtf8 == nullptr || outSize == nullptr) {
        set_error(L"Output pointers are null.");
        return 0;
    }
    *outUtf8 = nullptr;
    *outSize = 0;

    try {
        if (g_model_path.empty()) {
            set_error(L"WhisperSTT_Init must be called before transcription.");
            return 0;
        }

        const std::string wav_path = wide_to_utf8(wavPath);
        if (wav_path.empty()) {
            set_error(L"WAV path is required.");
            return 0;
        }

        stt_options options = g_default_options;
        parse_options(wide_to_utf8(cliOptionsOverride), options);

        std::string model_path = g_model_path;
        if (!options.model.empty()) {
            model_path = options.model;
        }

        std::wstring error;
        if (!validate_options(options, error)) {
            set_error(error);
            return 0;
        }
        if (options.no_prints) {
            whisper_log_set(cb_log_disable, nullptr);
        }
        if (!ensure_backends_loaded(error)) {
            set_error(error);
            return 0;
        }

        std::vector<float> pcmf32;
        std::vector<std::vector<float>> pcmf32s;
        if (!read_audio_data(wav_path, pcmf32, pcmf32s, false)) {
            set_error(L"Failed to read audio file.");
            return 0;
        }

        whisper_context * ctx = nullptr;
        bool owns_context = false;
        if (g_kept_context != nullptr && model_path == g_model_path) {
            ctx = g_kept_context;
        } else {
            ctx = load_context(model_path, options, error);
            if (ctx == nullptr) {
                set_error(error);
                return 0;
            }
            owns_context = true;
        }

        if (!whisper_is_multilingual(ctx)) {
            options.language = "en";
            options.translate = false;
        }
        if (options.detect_language) {
            options.language = "auto";
        }

        whisper_full_params wparams = make_full_params(options);
        if (whisper_full_parallel(ctx, wparams, pcmf32.data(), (int) pcmf32.size(), options.n_processors) != 0) {
            if (owns_context) {
                whisper_free(ctx);
            }
            set_error(L"Failed to process audio.");
            return 0;
        }

        std::string text;
        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            const char * segment = whisper_full_get_segment_text(ctx, i);
            if (segment != nullptr) {
                text += segment;
            }
        }

        if (owns_context) {
            whisper_free(ctx);
        }

        if (!copy_utf8_to_heap_buffer(text, outUtf8, outSize, error)) {
            set_error(error);
            return 0;
        }

        return 1;
    } catch (const std::exception & ex) {
        set_error(L"WhisperSTT_TranscribeFileUtf8 failed: " + widen_ascii(ex.what()));
        return 0;
    } catch (...) {
        set_error(L"WhisperSTT_TranscribeFileUtf8 failed.");
        return 0;
    }
}

void WhisperSTT_FreeBuffer(void* buffer) {
    if (buffer != nullptr) {
        delete[] static_cast<char*>(buffer);
    }
}

void WhisperSTT_Shutdown() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_last_error.clear();
    free_kept_context();
    g_model_path.clear();
    g_default_options = stt_options();
}

const wchar_t* WhisperSTT_GetLastError() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_last_error.c_str();
}

}
