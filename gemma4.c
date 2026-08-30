#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include "win.h"
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include <cpuid.h>
#include <omp.h>
#include <immintrin.h>

#define NUM_LAYERS 35
#define HIDDEN_SIZE 1536
#define VOCAB_SIZE 262144
#define MAX_CONTEXT 131072
#define SLIDING_WINDOW 512
#define BATCH_SIZE 512

// ----------------------------------------------------------------------------
// Tokenizer

typedef struct {
    char key[8]; // Holds one UTF-8 piece or a pair of 32-bit token IDs.
    int result, rank;
} LookupEntry;

typedef struct {
    char token[94]; // The longest vocabulary piece is 93 bytes plus the null terminator.
    int id;
} VocabEntry;

typedef struct {
    int merge_count;
    int encode_vocab_count;
    int special_count;
    char decoded_tokens[VOCAB_SIZE][94];
    VocabEntry specials[256]; // Reserves space for the checkpoint's 24 special tokens.
    LookupEntry encode_vocab[32768]; // Reserves space for 19,249 directly encoded pieces.
    LookupEntry merges[514906]; // This checkpoint contains 514,906 merge rules.
} Tokenizer;

int compare_lookup_keys(const void *key, const void *entry) {
    return memcmp(key, ((const LookupEntry *)entry)->key, 8);
}

// Repeatedly applies the highest-priority learned merge until no adjacent token pair matches.
int apply_bpe_merges(const Tokenizer *tokenizer, int *tokens, int count) {
    for (;;) {
        const LookupEntry *best_merge = NULL;
        int position = -1;
        for (int i = 0; i + 1 < count; i++) {
            const LookupEntry *merge = bsearch(tokens + i, tokenizer->merges, tokenizer->merge_count, sizeof(LookupEntry), compare_lookup_keys);
            if (merge && (!best_merge || merge->rank < best_merge->rank)) { best_merge = merge; position = i; }
        }
        if (!best_merge) return count;

        tokens[position] = best_merge->result;
        memmove(tokens + position + 1, tokens + position + 2, (count - position - 2) * sizeof(*tokens));
        count--;
    }
}

// Converts the three prompt segments from UTF-8 into vocabulary pieces, falls back to byte tokens when needed, applies BPE, and prepends <bos>.
int tokenize(const Tokenizer *tokenizer, const char *segments[3], int *tokens, int capacity) {
    int count = 1;

    for (int segment = 0; segment < 3; segment++)
        for (const char *cursor = segments[segment]; *cursor;) {
            if (count >= capacity) return -1;
            int special = -1;
            if (*cursor == '<')
                for (int i = 0; i < tokenizer->special_count && special < 0; i++) {
                    int length = (int)strlen(tokenizer->specials[i].token);
                    if (!strncmp(cursor, tokenizer->specials[i].token, length)) { special = tokenizer->specials[i].id; cursor += length; }
                }
            if (special >= 0) { tokens[count++] = special; continue; }
            char piece[8] = {0};
            if (*cursor == ' ') { memcpy(piece, "\xE2\x96\x81", 3); cursor++; } // SentencePiece represents spaces with U+2581.
            else {
                piece[0] = *cursor++;
                if ((piece[0] & 0xC0) == 0xC0)
                    for (int i = 1; i < 4 && (*cursor & 0xC0) == 0x80; i++) piece[i] = *cursor++;
            }
            const LookupEntry *entry = bsearch(piece, tokenizer->encode_vocab, tokenizer->encode_vocab_count,
                                      sizeof(LookupEntry), compare_lookup_keys);
            if (entry) { tokens[count++] = entry->result; continue; }

            for (const unsigned char *byte = (const unsigned char *)piece; *byte; byte++) {
                if (count >= capacity) return -1;
                tokens[count++] = 238 + *byte; // Byte tokens occupy IDs 238 through 493.
            }
        }

    count = 1 + apply_bpe_merges(tokenizer, tokens + 1, count - 1);
    tokens[0] = 2; // Token 2 is <bos>.
    return count;
}

const char *token_text(const Tokenizer *tokenizer, int token) {
    return token >= 0 && token < VOCAB_SIZE ? tokenizer->decoded_tokens[token] : "";
}

// ----------------------------------------------------------------------------
// Model

// data and scales begin as file offsets and become pointers after the model is memory-mapped.
typedef struct {
    void *data;
    uint16_t *scales;
    int shape[4];
} Tensor;

typedef struct {
    Tensor input_layernorm;
    Tensor layer_scalar;
    Tensor pre_ffn_layernorm;
    Tensor post_attn_layernorm;
    Tensor post_ffn_layernorm;
    Tensor post_per_layer_input_norm;
    Tensor per_layer_input_gate;
    Tensor per_layer_projection;
    Tensor q_norm;
    Tensor k_norm;
    Tensor q_proj;
    Tensor k_proj;
    Tensor v_proj;
    Tensor o_proj;
    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
    Tensor rope_cos;
    Tensor rope_sin;
} LayerWeights;

typedef struct {
    Tensor embed;
    Tensor embed_per_layer;
    LayerWeights layers[NUM_LAYERS];
    Tensor norm;
    Tensor per_layer_model_projection;
    Tensor per_layer_projection_norm;
    Tensor gelu_table;
} ModelWeights;

typedef struct {
    float residual[BATCH_SIZE * HIDDEN_SIZE];                           // Carries each token's hidden state through all 35 layers.
    float hidden[BATCH_SIZE * 8 * HIDDEN_SIZE];                         // Reused for intermediate results and sized for the largest 12,288-value MLP output.
    float auxiliary[BATCH_SIZE * 8 * HIDDEN_SIZE];                      // Holds a second intermediate when attention or the MLP needs two results at once.
    int8_t quantized[BATCH_SIZE * 8 * HIDDEN_SIZE];                     // Holds the current linear input after dynamic int8 quantization.
    float activation_scales[BATCH_SIZE * 8 * HIDDEN_SIZE / 64];         // Stores one float scale for every 64 quantized values.
    float per_layer_inputs[BATCH_SIZE * NUM_LAYERS * 256];              // Stores one 256-value conditioning vector for every token and layer.
    float sliding_cache[3][4][2 * (SLIDING_WINDOW + BATCH_SIZE) * 256]; // Keeps the previous window and current batch for twelve sliding KV caches.
    float full_cache[3][2 * MAX_CONTEXT * 512];                         // Holds the complete context for three 512-wide full-attention KV caches.
    int token_ids[MAX_CONTEXT];                                         // Holds the tokenized prompt before prefill.
} InferenceState;

typedef struct {
    char magic[4];
    Tokenizer tokenizer;
    ModelWeights weights;
} Model;

// Verifies that the compiler laid out the memory-mapped model exactly as the exporter expects.
_Static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(void *) == 8 && sizeof(VocabEntry) == 100 && offsetof(VocabEntry, id) == 96 && sizeof(LookupEntry) == 16 && sizeof(Tokenizer) == 33429932 && sizeof(Tensor) == 32 && sizeof(ModelWeights) == 21472 && sizeof(Model) == 33451408 && offsetof(Model, weights) == 33429936 && BATCH_SIZE == SLIDING_WINDOW && !((SLIDING_WINDOW + BATCH_SIZE) & (SLIDING_WINDOW + BATCH_SIZE - 1)) && !(MAX_CONTEXT & (MAX_CONTEXT - 1)), "MOG ABI mismatch");

// ----------------------------------------------------------------------------
// Kernels

// Uses one OpenMP thread per physical core because each core already uses SIMD, unless OMP_NUM_THREADS overrides it.
static inline int thread_count(void) {
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(0xB, 0, eax, ebx, ecx, edx);
    int logical_cpus = omp_get_num_procs();
    int threads_per_core = ebx & 0xffff;
    return getenv("OMP_NUM_THREADS") ? omp_get_max_threads() : logical_cpus / (threads_per_core ? threads_per_core : 1);
}

// Multiplies dynamically quantized activations by packed int8 weights in blocks of 16
// output rows. VNNI handles eight input rows at once while AVX2 handles four. Both
// accumulate integer dot products before restoring float values with their scales.
#if defined(__AVX512VNNI__)
static inline __attribute__((always_inline)) void matmul_block(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t rows, size_t output_block) {
    const size_t block_rows = 16;
    size_t groups_per_row = (size_t)weight->shape[1] / 64;
    const int8_t *packed_weights = (const int8_t *)weight->data + output_block * block_rows * weight->shape[1];
    for (size_t row_start = 0; row_start < rows; row_start += 8) {
        size_t active_rows = rows - row_start < 8 ? rows - row_start : 8;
        __m512 result[8] = {0};
        for (size_t group = 0; group < groups_per_row; group++) {
            __m512i dot[8] = {0};
            __m512i correction = _mm512_setzero_si512();
            __m512i offset_bytes = _mm512_set1_epi8(-128); // Flip signed activations into the unsigned range required by VNNI.
            for (int chunk = 0; chunk < 16; chunk++) {
                __m512i weight_values = _mm512_load_si512((const __m512i *)(packed_weights + group * 1024 + chunk * 64));
                correction = _mm512_dpbusd_epi32(correction, offset_bytes, weight_values);
                for (size_t row = 0; row < active_rows; row++) {
                    __m512i input_values = _mm512_broadcastd_epi32(
                        _mm_loadu_si32(input_q + (row_start + row) * weight->shape[1] + group * 64 + chunk * 4));
                    input_values = _mm512_xor_si512(input_values, offset_bytes);
                    dot[row] = _mm512_dpbusd_epi32(dot[row], input_values, weight_values);
                }
            }
            __m512 weight_scales = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(weight->scales + (output_block * groups_per_row + group) * block_rows)));
            for (size_t row = 0; row < active_rows; row++)
                result[row] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(dot[row], correction)), _mm512_mul_ps(weight_scales, _mm512_set1_ps(input_scales[(row_start + row) * groups_per_row + group])), result[row]);
        }
        for (size_t row = 0; row < active_rows; row++)
            _mm512_storeu_ps(output + (row_start + row) * weight->shape[0] + output_block * block_rows, result[row]);
    }
}
#else
static inline __attribute__((always_inline)) void matmul_block(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t rows, size_t output_block) {
    const size_t block_rows = 16;
    size_t groups_per_row = (size_t)weight->shape[1] / 64;
    const int8_t *packed_weights = (const int8_t *)weight->data + output_block * block_rows * weight->shape[1];
    for (size_t row_start = 0; row_start < rows; row_start += 4) {
        size_t active_rows = rows - row_start < 4 ? rows - row_start : 4;
        for (int half = 0; half < 2; half++) {
            __m256 result[4] = {0};
            for (size_t group = 0; group < groups_per_row; group++) {
                __m256i dot[4] = {0};
                for (int chunk = 0; chunk < 16; chunk++) {
                    __m256i weight_values = _mm256_loadu_si256((const __m256i *)(packed_weights + group * 1024 + chunk * 64 + half * 32));
                    __m256i weight_magnitudes = _mm256_abs_epi8(weight_values);
                    for (size_t row = 0; row < active_rows; row++) {
                        __m256i input_values = _mm256_broadcastd_epi32(
                            _mm_loadu_si32(input_q + (row_start + row) * weight->shape[1] + group * 64 + chunk * 4));
                        dot[row] = _mm256_add_epi32(dot[row], _mm256_madd_epi16(_mm256_maddubs_epi16(weight_magnitudes, _mm256_sign_epi8(input_values, weight_values)), _mm256_set1_epi16(1)));
                    }
                }
                __m256 weight_scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(weight->scales + (output_block * groups_per_row + group) * block_rows + half * 8)));
                for (size_t row = 0; row < active_rows; row++)
                    result[row] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot[row]), _mm256_mul_ps(weight_scales, _mm256_set1_ps(input_scales[(row_start + row) * groups_per_row + group])), result[row]);
            }
            for (size_t row = 0; row < active_rows; row++)
                _mm256_storeu_ps(output + (row_start + row) * weight->shape[0] + output_block * block_rows + half * 8, result[row]);
        }
    }
}
#endif

void matmul_int8(float *output, const int8_t *input_q, const float *input_scales,
                 const Tensor *weight, size_t rows) {
    const size_t block_rows = 16;
    #pragma omp for schedule(static)
    for (size_t output_block = 0; output_block < (size_t)weight->shape[0] / block_rows; output_block++)
        matmul_block(output, input_q, input_scales, weight, rows, output_block);
}

// Converts each input row to int8 in groups of 64 values with a float scale recording each group's magnitude.
void quantize(int8_t *quantized, float *scales, const float *input, size_t rows, size_t width) {
    #pragma omp for schedule(static)
    for (size_t group_index = 0; group_index < rows * (width / 64); group_index++) {
        const float *group = input + group_index * 64;
        float max_abs = 0.0f;
        for (int j = 0; j < 64; j++) {
            float value = fabsf(group[j]);
            if (value > max_abs) max_abs = value;
        }
        float scale = max_abs / 127.0f;
        float inverse_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int j = 0; j < 64; j++) quantized[group_index * 64 + j] = (int8_t)rintf(group[j] * inverse_scale);
        scales[group_index] = scale;
    }
}

void attention_scores(float *scores, const float *query, const float *key_cache,
        int first_key, int num_keys, int cache_mask, int head_dim) {
    for (int key_index = 0; key_index < num_keys; key_index++) {
        const float *key = key_cache + ((first_key + key_index) & cache_mask) * head_dim;
        __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
        for (int j = 0; j < head_dim; j += 16) {
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(query + j), _mm256_loadu_ps(key + j), sum0);
            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(query + j + 8), _mm256_loadu_ps(key + j + 8), sum1);
        }
        __m256 sum8 = _mm256_add_ps(sum0, sum1);
        __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(sum8), _mm256_extractf128_ps(sum8, 1));
        sum4 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        scores[key_index] = _mm_cvtss_f32(_mm_add_ss(sum4, _mm_movehdup_ps(sum4)));
    }
}

void weighted_value_sum(float *output, const float *probabilities, const float *value_cache,
        int first_key, int num_keys, int cache_mask, int head_dim) {
    for (int j = 0; j < head_dim; j += 64) {
        __m256 sum[8] = {0};
        for (int key_index = 0; key_index < num_keys; key_index++) {
            const float *value = value_cache + ((first_key + key_index) & cache_mask) * head_dim + j;
            __m256 probability = _mm256_set1_ps(probabilities[key_index]);
            for (int u = 0; u < 8; u++)
                sum[u] = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + u * 8), sum[u]);
        }
        for (int u = 0; u < 8; u++) _mm256_storeu_ps(output + j + u * 8, sum[u]);
    }
}

// Approximates GELU from the exported lookup table and multiplies it by the up projection to produce the MLP's gated activation.
void geglu(float *gate, const float *up, int rows, int width, int up_stride, const Tensor *gelu_table) {
    const float *table = (const float *)gelu_table->data;
    const int table_size = gelu_table->shape[0];
    const float lower = (float)gelu_table->shape[1];
    const float upper = (float)gelu_table->shape[2];
    const float scale = (float)(table_size - 1) / (upper - lower);
    #pragma omp for collapse(2) schedule(static)
    for (int row = 0; row < rows; row++) {
        for (int i = 0; i < width; i++) {
            float x = gate[row * width + i];
            if (x <= lower) {
                x = table[0];
            } else if (!(x >= upper)) {
                float position = (x - lower) * scale;
                int index = (int)position;
                float fraction = position - (float)index;
                x = table[index] + fraction * (table[index + 1] - table[index]);
            }
            gate[row * width + i] = x * up[row * up_stride + i];
        }
    }
}

// ----------------------------------------------------------------------------
// Transformer

// Looks up packed int8 embedding rows and dequantizes them directly without materializing the full embedding table.
void embedding(float *output, const Tensor *table, const int *tokens, size_t token_count, float multiplier) {
    const int block_rows = 16;
    int width = table->shape[1];
    int groups = width / 64;
    #pragma omp for schedule(static)
    for (size_t token = 0; token < token_count; token++) {
        size_t block = (size_t)(tokens[token] / block_rows);
        int row = tokens[token] % block_rows;
        float *vector = output + token * width;
        const int8_t *block_data = (const int8_t *)table->data + block * block_rows * width;
        const uint16_t *block_scales = table->scales + block * groups * block_rows;
        for (size_t group_index = 0; group_index < (size_t)groups; group_index++) {
            const int8_t *group = block_data + group_index * block_rows * 64;
            float scale = _cvtsh_ss(block_scales[group_index * block_rows + row]) * multiplier;
            for (int j = 0; j < 64; j++) {
                int chunk = j / 4;
                int offset = j % 4;
                vector[group_index * 64 + j] = (float)group[chunk * block_rows * 4 + row * 4 + offset] * scale;
            }
        }
    }
}

void rmsnorm(float *output, const float *input, const Tensor *weight, int width, float epsilon, size_t row_count) {
    const float *weights = weight ? (const float *)weight->data : NULL;
    #pragma omp for schedule(static)
    for (size_t row = 0; row < row_count; row++) {
        const float *input_row = input + row * width;
        float *output_row = output + row * width;
        float sum_squares = 0.0f;
        for (int i = 0; i < width; i++)
            sum_squares += input_row[i] * input_row[i];
        float inverse_rms = 1.0f / sqrtf(sum_squares / (float)width + epsilon);
        for (int i = 0; i < width; i++)
            output_row[i] = (weights ? weights[i] : 1.0f) * (inverse_rms * input_row[i]);
    }
}

void add_and_scale(float *output, const float *addend, size_t count, float scale) {
    #pragma omp for schedule(static)
    for (size_t i = 0; i < count; i++) output[i] = (output[i] + addend[i]) * scale;
}

// Rotates pairs of query or key channels using each position's sine and cosine values so attention can distinguish token order.
void apply_rope(const Tensor *cosines, const Tensor *sines, float *vectors,
                int num_heads, int head_dim, int start_pos, size_t token_count) {
    int pairs = cosines->shape[1];
    #pragma omp for schedule(static)
    for (size_t token = 0; token < token_count; token++) {
        const float *cosine = (float *)cosines->data + (start_pos + token) * pairs;
        const float *sine = (float *)sines->data + (start_pos + token) * pairs;
        for (size_t head = 0; head < (size_t)num_heads; head++) {
            float *vector = vectors + (token * num_heads + head) * head_dim;
            for (int j = 0; j < pairs; j++) {
                float first = vector[j];
                float second = vector[j + head_dim / 2];
                vector[j] = first * cosine[j] - second * sine[j];
                vector[j + head_dim / 2] = second * cosine[j] + first * sine[j];
            }
        }
    }
}

void softmax(float *values, int count) {
    float max = values[0], sum = 1.0f;
    for (int i = 1; i < count; i++) {
        if (values[i] > max) { sum = sum * expf(max - values[i]) + 1.0f; max = values[i]; } // Rescale the sum when a new maximum appears so expf() stays in range.
        else sum += expf(values[i] - max);
    }
    for (int i = 0; i < count; i++) values[i] = expf(values[i] - max) / sum;
}

// Builds queries, updates the KV cache, and computes causal attention over 512 tokens or the full context while shared layers reuse the latest compatible cache.
void attention(InferenceState *state, const LayerWeights *layers, int layer,
               int start_pos, size_t token_count, float *scores) {
    const LayerWeights *weights = &layers[layer];
    int full_attention = layer % 5 == 4; // Every fifth layer uses full attention.
    int cache_len = full_attention ? MAX_CONTEXT : SLIDING_WINDOW + BATCH_SIZE;
    int cache_mask = cache_len - 1; // Both cache lengths are powers of two, so masking wraps positions without division.
    int head_dim = weights->q_norm.shape[0];
    int query_width = weights->q_proj.shape[0];
    int cache_owner = layer;
    while (!layers[cache_owner].k_proj.data || (cache_owner % 5 == 4) != full_attention) cache_owner--; // Shared layers reuse the latest cache of the same attention type.
    float *key_cache = full_attention ? state->full_cache[cache_owner / 5] : state->sliding_cache[cache_owner / 5][cache_owner % 5];
    float *value_cache = key_cache + (size_t)cache_len * head_dim;

    quantize(state->quantized, state->activation_scales, state->hidden, token_count, weights->q_proj.shape[1]);
    matmul_int8(state->auxiliary, state->quantized, state->activation_scales, &weights->q_proj, token_count);
    rmsnorm(state->auxiliary, state->auxiliary, &weights->q_norm, head_dim, 1e-6f, token_count * (query_width / head_dim));
    apply_rope(&weights->rope_cos, &weights->rope_sin, state->auxiliary, query_width / head_dim, head_dim, start_pos, token_count);

    if (weights->k_proj.data) {
        float *new_keys = key_cache + ((size_t)start_pos & cache_mask) * head_dim;
        float *new_values = value_cache + ((size_t)start_pos & cache_mask) * head_dim;
        matmul_int8(new_keys, state->quantized, state->activation_scales, &weights->k_proj, token_count);
        matmul_int8(new_values, state->quantized, state->activation_scales, &weights->v_proj, token_count);
        rmsnorm(new_keys, new_keys, &weights->k_norm, head_dim, 1e-6f, token_count);
        rmsnorm(new_values, new_values, NULL, head_dim, 1e-6f, token_count); // Value vectors are normalized without a learned weight.
        apply_rope(&weights->rope_cos, &weights->rope_sin, new_keys, 1, head_dim, start_pos, token_count);
    }

    #pragma omp for collapse(2) schedule(dynamic, 1)
    for (size_t head = 0; head < (size_t)(query_width / head_dim); head++) {
        for (size_t token = 0; token < token_count; token++) {
            int first_key = !full_attention && start_pos + (int)token + 1 > SLIDING_WINDOW ? start_pos + (int)token + 1 - SLIDING_WINDOW : 0;
            int num_keys = start_pos + (int)token + 1 - first_key;
            float *head_output = state->hidden + token * query_width + head * head_dim;
            const float *query = state->auxiliary + token * query_width + head * head_dim;
            attention_scores(scores, query, key_cache, first_key, num_keys, cache_mask, head_dim);
            softmax(scores, num_keys);
            weighted_value_sum(head_output, scores, value_cache, first_key, num_keys, cache_mask, head_dim);
        }
    }

    quantize(state->quantized, state->activation_scales, state->hidden, token_count, weights->o_proj.shape[1]);
    matmul_int8(state->hidden, state->quantized, state->activation_scales, &weights->o_proj, token_count);
}

void forward(Model *model, InferenceState *state, const int *tokens, size_t token_count, int start_pos) {
    int per_layer_width = model->weights.per_layer_projection_norm.shape[0];
    // One OpenMP team stays alive for the full forward pass while each kernel divides its own loop.
    #pragma omp parallel num_threads(thread_count())
    {
    float scores[(size_t)start_pos + token_count]; // Each thread needs private scratch large enough for every visible key.
    embedding(state->residual, &model->weights.embed, tokens, token_count, sqrtf((float)HIDDEN_SIZE));

    // Build the token-conditioned input that each transformer layer will receive.
    quantize(state->quantized, state->activation_scales, state->residual, token_count, HIDDEN_SIZE);
    matmul_int8(state->per_layer_inputs, state->quantized, state->activation_scales, &model->weights.per_layer_model_projection, token_count);
    rmsnorm(state->per_layer_inputs, state->per_layer_inputs, &model->weights.per_layer_projection_norm, per_layer_width, 1e-6f * HIDDEN_SIZE, token_count * NUM_LAYERS);

    embedding(state->hidden, &model->weights.embed_per_layer, tokens, token_count, sqrtf((float)per_layer_width));
    add_and_scale(state->per_layer_inputs, state->hidden, token_count * NUM_LAYERS * per_layer_width,
               1.0f / sqrtf(2.0f)); // Keeps the variance of the combined inputs unchanged.

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        LayerWeights *weights = &model->weights.layers[layer];

        rmsnorm(state->hidden, state->residual, &weights->input_layernorm, HIDDEN_SIZE, 1e-6f, token_count);
        attention(state, model->weights.layers, layer, start_pos, token_count, scores);
        rmsnorm(state->hidden, state->hidden, &weights->post_attn_layernorm, HIDDEN_SIZE, 1e-6f, token_count);
        add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE, 1.0f);

        rmsnorm(state->hidden, state->residual, &weights->pre_ffn_layernorm, HIDDEN_SIZE, 1e-6f, token_count);
        quantize(state->quantized, state->activation_scales, state->hidden, token_count, weights->gate_proj.shape[1]);
        matmul_int8(state->hidden, state->quantized, state->activation_scales, &weights->gate_proj, token_count);
        matmul_int8(state->auxiliary, state->quantized, state->activation_scales, &weights->up_proj, token_count);
        geglu(state->hidden, state->auxiliary, token_count, weights->gate_proj.shape[0], weights->gate_proj.shape[0], &model->weights.gelu_table);
        quantize(state->quantized, state->activation_scales, state->hidden, token_count, weights->down_proj.shape[1]);
        matmul_int8(state->hidden, state->quantized, state->activation_scales, &weights->down_proj, token_count);
        rmsnorm(state->hidden, state->hidden, &weights->post_ffn_layernorm, HIDDEN_SIZE, 1e-6f, token_count);
        add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE, 1.0f);
        // Gate and project this layer's conditioning input before adding it to the residual stream with a learned scale.
        quantize(state->quantized, state->activation_scales, state->residual, token_count, HIDDEN_SIZE);
        matmul_int8(state->hidden, state->quantized, state->activation_scales, &weights->per_layer_input_gate, token_count);
        geglu(state->hidden, state->per_layer_inputs + layer * per_layer_width, token_count, per_layer_width, NUM_LAYERS * per_layer_width, &model->weights.gelu_table);
        quantize(state->quantized, state->activation_scales, state->hidden, token_count, weights->per_layer_projection.shape[1]);
        matmul_int8(state->hidden, state->quantized, state->activation_scales, &weights->per_layer_projection, token_count);
        rmsnorm(state->hidden, state->hidden, &weights->post_per_layer_input_norm, HIDDEN_SIZE, 1e-6f, token_count);
        add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE,
                   ((float *)weights->layer_scalar.data)[0]);
    }
    }
}

// Reuses the embedding matrix to turn the final token representation into vocabulary logits, then applies Gemma's tanh soft cap.
float *logits(Model *model, InferenceState *state, size_t token) {
    #pragma omp parallel num_threads(thread_count())
    {
        rmsnorm(state->hidden, state->residual + token * HIDDEN_SIZE, &model->weights.norm, HIDDEN_SIZE, 1e-6f, 1);
        quantize(state->quantized, state->activation_scales, state->hidden, 1, HIDDEN_SIZE);
        matmul_int8(state->hidden, state->quantized, state->activation_scales, &model->weights.embed, 1);
        #pragma omp for schedule(static)
        for (int i = 0; i < VOCAB_SIZE; i++) state->hidden[i] = 30.0f * tanhf(state->hidden[i] / 30.0f);
    }
    return state->hidden;
}

// ----------------------------------------------------------------------------
// Generation

unsigned long long rng_state = 42;

float random_uniform(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (float)((rng_state * 0x2545F4914F6CDD1DULL) >> 40) / 16777216.0f;
}

int sample(float *logits, int vocab_size, float temperature) {
    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    struct { float score; int token; } top[64]; // Sampling considers only the 64 highest logits.
    for (int i = 0; i < 64; i++) top[i].score = -INFINITY;
    for (int token = 0; token < vocab_size; token++) {
        if (logits[token] <= top[63].score) continue;
        int i = 63;
        while (i > 0 && logits[token] > top[i - 1].score) { top[i] = top[i - 1]; i--; }
        top[i].score = logits[token]; top[i].token = token;
    }
    float sum = 0.0f, max = top[0].score / temperature;
    for (int i = 0; i < 64; i++) sum += top[i].score = expf(top[i].score / temperature - max);
    float mass = 0.0f;
    int count = 0;
    while (mass < 0.95f * sum) mass += top[count++].score; // Keep the smallest prefix containing 95% of the top-64 probability mass.
    float threshold = random_uniform() * mass;
    for (int i = 0; i < count; i++)
        if ((threshold -= top[i].score) <= 0.0f) return top[i].token;
    return top[count - 1].token;
}

void prefill(Model *model, InferenceState *state, const int *tokens, int token_count, int dump_logits) {
    for (int position = 0; position < token_count; position += BATCH_SIZE) {
        int chunk = token_count - position < BATCH_SIZE ? token_count - position : BATCH_SIZE;
        forward(model, state, tokens + position, chunk, position);
        if (dump_logits) {
            for (int i = 0; i < chunk; i++) {
                fwrite(logits(model, state, i), sizeof(float), VOCAB_SIZE, stdout);
            }
        }
    }
}

void generate(Model *model, InferenceState *state, const char *prompt,
              int max_new_tokens, float temperature, int dump_logits) {
    Tokenizer *tokenizer = &model->tokenizer;
    int styled = !dump_logits && isatty(STDOUT_FILENO);

    if (max_new_tokens < 0) {
        fprintf(stderr, "-n must be non-negative\n");
        exit(1);
    }
    const char *segments[3] = {dump_logits ? "" : "<|turn>user\n", prompt,
                               dump_logits ? "" : "<turn|>\n<|turn>model\n"};
    int prompt_tokens = tokenize(tokenizer, segments, state->token_ids, MAX_CONTEXT);
    if (prompt_tokens < 0) {
        fprintf(stderr, "prompt exceeds the %d-token context limit\n", MAX_CONTEXT);
        exit(1);
    }

    if (styled) {
        fputs("\n\033[2;36m────────────────────────────────\033[0m\n", stdout);
        fflush(stdout);
    }

    prefill(model, state, state->token_ids, prompt_tokens, dump_logits);
    if (dump_logits) return;

    int end = prompt_tokens + max_new_tokens;
    if (end > MAX_CONTEXT || end < prompt_tokens) end = MAX_CONTEXT;
    for (int position = prompt_tokens; position < end; position++) {
        int next_token = sample(logits(model, state, position == prompt_tokens ? (prompt_tokens - 1) % BATCH_SIZE : 0), VOCAB_SIZE, temperature);
        if (next_token == 1 || next_token == 106) break; // Stop at <eos> or <turn|>.

        fputs(token_text(tokenizer, next_token), stdout);
        fflush(stdout);
        forward(model, state, &next_token, 1, position);
    }
    putchar('\n');
}

double time_seconds(void) {
#ifdef _WIN32
    static double frequency;
    LARGE_INTEGER ticks;
    if (!frequency) {
        QueryPerformanceFrequency(&ticks);
        frequency = (double)ticks.QuadPart;
    }
    QueryPerformanceCounter(&ticks);
    return (double)ticks.QuadPart / frequency;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
#endif
}

void benchmark(Model *model, InferenceState *state, int prefill_tokens, int generated_tokens) {
    if (prefill_tokens > 0) {
        for (int i = 0; i < prefill_tokens; i++)
            state->token_ids[i] = 2 + i % 1000;
        double start = time_seconds();
        prefill(model, state, state->token_ids, prefill_tokens, 0);
        printf("pp%d %.2f tok/s\n", prefill_tokens, (double)prefill_tokens / (time_seconds() - start));
    }
    if (generated_tokens > 0) {
        int token = 2;
        double start = time_seconds();
        for (int position = prefill_tokens; position < prefill_tokens + generated_tokens; position++) {
            forward(model, state, &token, 1, position);
            token = sample(logits(model, state, 0), VOCAB_SIZE, 0.0f);
        }
        printf("tg%d@d%d %.2f tok/s\n", generated_tokens, prefill_tokens, (double)generated_tokens / (time_seconds() - start));
    }
}

int main(int argc, char **argv) {
#ifdef _WIN32
    argv_utf8(&argc, &argv);
#endif
    const char *model_path = "gemma4-E2B-int8.bin";
    const char *prompt = "Why is the sky blue?";
    float temperature = 1.0f;
    int max_new_tokens = 1024;
    int benchmark_mode = 0, dump_logits = 0, prefill_tokens = 0, generated_tokens = 256;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) temperature = atof(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_new_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bench")) {
            benchmark_mode = 1;
            if (i + 1 < argc) prefill_tokens = atoi(argv[++i]);
            if (i + 1 < argc) generated_tokens = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--dump-logits")) dump_logits = 1;
        else prompt = argv[i];
    }
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    if (dump_logits) _setmode(_fileno(stdout), _O_BINARY);
#endif
    int fd = open(model_path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) { perror(model_path); return 1; }
    Model *model = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (model == MAP_FAILED) { perror("mmap"); return 1; }

    if (memcmp(model->magic, "MOG", 4) != 0) { fprintf(stderr, "bad model file\n"); return 1; }
    Tensor *tensors = (Tensor *)&model->weights;
    for (size_t i = 0; i < sizeof(model->weights) / sizeof(*tensors); i++) {
        tensors[i].data = tensors[i].data ? (void *)((uint8_t *)model + (uintptr_t)tensors[i].data) : NULL;
        tensors[i].scales = tensors[i].scales ? (uint16_t *)((uint8_t *)model + (uintptr_t)tensors[i].scales) : NULL;
    }
    InferenceState *state = calloc(1, sizeof(*state));

    rng_state = (unsigned long long)(time_seconds() * 1e9);
    if (benchmark_mode) benchmark(model, state, prefill_tokens, generated_tokens);
    else generate(model, state, prompt, max_new_tokens, temperature, dump_logits);
    free(state);
    munmap(model, (size_t)st.st_size);
    return 0;
}

//      |\__/,|   (`\_
//    *.|o o  |*   ) )
//---(((---(((------------------
//|                            |
//|          gemma4.c          |
//|____________________________|
