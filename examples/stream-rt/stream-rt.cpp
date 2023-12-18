
#include "common.h"  // Common includes for Whisper
#include "whisper.h" // Whisper API
#include <cstdio>
#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>  // This header defines the transform function

struct whisper_params
{
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t step_ms = 3000;
    int32_t length_ms = 10000;
    int32_t keep_ms = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx = 0;

    float vad_thold = 0.6f;
    float freq_thold = 100.0f;

    bool speed_up = false;
    bool from_wav_file = false; // New field
    bool translate = false;
    bool no_fallback = false;
    bool print_special = false;
    bool no_context = true;
    bool no_timestamps = false;
    bool tinydiarize = false;
    bool save_audio = false; // save audio to wav file
    bool use_gpu = true;

    std::string language = "en";
    std::string model = "models/ggml-base.en.bin";
    std::string fname_out;
};

void whisper_print_usage(int /*argc*/, char **argv, const whisper_params &params)
{
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n", params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n", params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n", params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n", params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n", params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n", params.audio_ctx);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n", params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n", params.freq_thold);
    fprintf(stderr, "  -su,      --speed-up      [%-7s] speed up audio by x2 (reduced accuracy)\n", params.speed_up ? "true" : "false");
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n", params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n", params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n", params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n", params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n", params.model.c_str());
    fprintf(stderr, "            --from-wav-file [%-7s] read audio from wav file\n", params.from_wav_file ? "true" : "false");
    fprintf(stderr, "  -f FNAME, --file FNAME    [%-7s] text output file name\n", params.fname_out.c_str());
    fprintf(stderr, "  -tdrz,    --tinydiarize   [%-7s] enable tinydiarize (requires a tdrz model)\n", params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -sa,      --save-audio    [%-7s] save the recorded audio to a file\n", params.save_audio ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n", params.use_gpu ? "false" : "true");
    fprintf(stderr, "\n");
}

// Function to read WAV data from standard input and convert it to a vector of floats
std::vector<float> readWavStreamAsFloats()
{
    // Skip WAV header (44 bytes for standard PCM WAV files)
    std::cin.ignore(44);

    // Buffer for 16-bit samples
    std::vector<int16_t> buffer16((1e-3 * 10000) * WHISPER_SAMPLE_RATE);

    // Read 16-bit samples from standard input
    std::cin.read(reinterpret_cast<char *>(buffer16.data()), buffer16.size() * sizeof(int16_t));
    if (std::cin.gcount() == 0)
    {
        throw std::runtime_error("Error reading WAV data from standard input.");
    }

    // Convert 16-bit to float
    std::vector<float> pcmf32(buffer16.size());
    std::transform(buffer16.begin(), buffer16.end(), pcmf32.begin(),
                   [](int16_t sample) -> float
                   { return sample / 32768.0f; });

    return pcmf32;
}

bool whisper_params_parse(int argc, char **argv, whisper_params &params)
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help")
        {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t" || arg == "--threads")
        {
            params.n_threads = std::stoi(argv[++i]);
        }
        else if (arg == "--step")
        {
            params.step_ms = std::stoi(argv[++i]);
        }
        else if (arg == "--length")
        {
            params.length_ms = std::stoi(argv[++i]);
        }
        else if (arg == "--keep")
        {
            params.keep_ms = std::stoi(argv[++i]);
        }
        else if (arg == "-c" || arg == "--capture")
        {
            params.capture_id = std::stoi(argv[++i]);
        }
        else if (arg == "-mt" || arg == "--max-tokens")
        {
            params.max_tokens = std::stoi(argv[++i]);
        }
        else if (arg == "-ac" || arg == "--audio-ctx")
        {
            params.audio_ctx = std::stoi(argv[++i]);
        }
        else if (arg == "-vth" || arg == "--vad-thold")
        {
            params.vad_thold = std::stof(argv[++i]);
        }
        else if (arg == "-fth" || arg == "--freq-thold")
        {
            params.freq_thold = std::stof(argv[++i]);
        }
        else if (arg == "-su" || arg == "--speed-up")
        {
            params.speed_up = true;
        }
        else if (arg == "-tr" || arg == "--translate")
        {
            params.translate = true;
        }
        else if (arg == "-nf" || arg == "--no-fallback")
        {
            params.no_fallback = true;
        }
        else if (arg == "-ps" || arg == "--print-special")
        {
            params.print_special = true;
        }
        else if (arg == "-kc" || arg == "--keep-context")
        {
            params.no_context = false;
        }
        else if (arg == "-l" || arg == "--language")
        {
            params.language = argv[++i];
        }
        else if (arg == "-m" || arg == "--model")
        {
            params.model = argv[++i];
        }
        else if (arg == "-f" || arg == "--file")
        {
            params.fname_out = argv[++i];
        }
        else if (arg == "-tdrz" || arg == "--tinydiarize")
        {
            params.tinydiarize = true;
        }
        else if (arg == "-sa" || arg == "--save-audio")
        {
            params.save_audio = true;
        }
        else if (arg == "-ng" || arg == "--no-gpu")
        {
            params.use_gpu = false;
        }
        else if (arg == "--from-wav-file")
        {
            params.from_wav_file = true;
        }
        else
        {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

// Convert milliseconds to timestamp string
std::string to_timestamp(int64_t t)
{
    int64_t sec = t / 100;
    int64_t msec = t - sec * 100;
    int64_t min = sec / 60;
    sec = sec - min * 60;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d.%03d", (int)min, (int)sec, (int)msec);

    return std::string(buf);
}

void whisper_print_usage(int argc, char **argv, const whisper_params &params);

int main(int argc, char **argv)
{
    std::cout << "Initializing Whisper RT Service..." << std::endl;

    whisper_params params;
    if (!whisper_params_parse(argc, argv, params))
    {
        std::cerr << "Error: Failed to parse parameters." << std::endl;
        return 1;
    }

    std::cout << "Parsed parameters successfully." << std::endl;

    struct whisper_context_params cparams;
    cparams.use_gpu = params.use_gpu;
    const int n_samples_step = (1e-3 * params.step_ms) * WHISPER_SAMPLE_RATE;

    std::cout << "Initializing Whisper context..." << std::endl;
    struct whisper_context *ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (!ctx)
    {
        std::cerr << "Failed to initialize whisper context." << std::endl;
        return 2;
    }

    std::cout << "Whisper context initialized successfully." << std::endl;

    std::vector<float> pcmf32(n_samples_step, 0.0f);
    std::vector<float> pcmf32_old;

    std::ofstream fout;
    if (!params.fname_out.empty())
    {
        std::cout << "Opening output file: " << params.fname_out << std::endl;
        fout.open(params.fname_out);
        if (!fout.is_open())
        {
            std::cerr << "Failed to open output file: " << params.fname_out << std::endl;
            return 3;
        }
        std::cout << "Output file opened successfully." << std::endl;
    }

    bool is_running = true;
    int n_iter = 0;

    std::cout << "Starting main audio processing loop..." << std::endl;
    while (is_running)
    {
        if (params.from_wav_file)
        {
            std::cout << "Reading audio data from standard input..." << std::endl;
            pcmf32 = readWavStreamAsFloats();
        }
        else
        {
            std::cout << "Reading audio data from standard input..." << std::endl;
            std::cin.read(reinterpret_cast<char *>(pcmf32.data()), pcmf32.size() * sizeof(float));
            if (std::cin.eof())
            {
                std::cout << "End of audio stream detected." << std::endl;
                break;
            }
        }
        if (std::cin.eof())
        {
            std::cout << "End of audio stream detected." << std::endl;
            break;
        }

        std::cout << "Processing audio data, iteration " << n_iter + 1 << "..." << std::endl;
        // std::cin.read(reinterpret_cast<char *>(pcmf32.data()), pcmf32.size() * sizeof(float));
        if (std::cin.eof())
        {
            std::cout << "End of audio stream detected." << std::endl;
            // break;
        }

        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.print_progress = false;
        wparams.print_special = params.print_special;
        wparams.print_realtime = false;
        wparams.print_timestamps = !params.no_timestamps;
        wparams.translate = params.translate;
        wparams.single_segment = true;
        wparams.max_tokens = params.max_tokens;
        wparams.language = params.language.c_str();
        wparams.n_threads = params.n_threads;
        wparams.audio_ctx = params.audio_ctx;
        wparams.speed_up = params.speed_up;
        wparams.tdrz_enable = params.tinydiarize;
        wparams.temperature_inc = params.no_fallback ? 0.0f : wparams.temperature_inc;

        if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0)
        {
            fprintf(stderr, "%s: failed to process audio\n", argv[0]);
            is_running = false;
            continue;
        }

        std::cout << "Audio processed successfully." << std::endl;
        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i)
        {
            const char *text = whisper_full_get_segment_text(ctx, i);
            if (params.no_timestamps)
            {
                printf("%s", text);
                if (fout.is_open())
                {
                    fout << text;
                }
            }
            else
            {
                const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
                printf("[%s --> %s] %s\n", to_timestamp(t0).c_str(), to_timestamp(t1).c_str(), text);
                if (fout.is_open())
                {
                    fout << "[" << to_timestamp(t0) << " --> " << to_timestamp(t1) << "] " << text << std::endl;
                }
            }
        }

        ++n_iter;
    }

    if (fout.is_open())
    {
        std::cout << "Closing output file." << std::endl;
        fout.close();
    }

    std::cout << "Freeing Whisper context." << std::endl;
    whisper_free(ctx);

    std::cout << "Whisper RT Service terminated successfully." << std::endl;

    return 0;
}
