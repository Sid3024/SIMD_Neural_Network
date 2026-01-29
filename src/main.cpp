#include <immintrin.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <chrono>
#include <array>
#include <random>
#include "simd_functions.h"

int NUM_RUNS = 100000; //num_runs to test execution time

const int input_size = 1024;
const int hidden_dim = 256;

template <typename T>
struct AlignedDeleter {
    void operator()(T* ptr) const {
        std::free(ptr);
    }
};
struct move_indices {
    int8_t start_square_index;
    int8_t end_square_index;
    int8_t promotion_index;
};

//params
const size_t degree = 7;
const int n_eval_passes = 1;
//PARAMS
//input block
alignas(32) int8_t mat_params_ib[input_size*hidden_dim];
alignas(32) int32_t bias_ib[hidden_dim];

//eval block layer 1
alignas(32) float coeff_eb_L1[(degree+1)*n_eval_passes];
alignas(32) int8_t mat_params_eb_L1[hidden_dim*hidden_dim];
alignas(32) int32_t bias_eb_L1[hidden_dim];

//eval block layer 2
alignas(32) float coeff_eb_L2[(degree+1)*n_eval_passes];
alignas(32) int8_t mat_params_eb_L2[hidden_dim*hidden_dim];
alignas(32) int32_t bias_eb_L2[hidden_dim];

//value head
alignas(32) float coeff_vh[(degree+1)*hidden_dim];
alignas(32) int8_t mat_params_vh[hidden_dim*hidden_dim];
alignas(32) int32_t bias_vh[hidden_dim];


class Layer {
private:
    size_t N;
    size_t K;
    size_t M;
    //input-> (N,K), output-> (K,M)
    size_t blockSizeA;
    size_t blockSizeB;
    size_t degree = 7;
    bool params_stored;
    float scaling_factor_float_int8 = 127.0/2.0;
    float scaling_factor_int32_float = 127.0/2.0;
    //naming arrays-> dtype_N_M_K_a (a stands for array)
    inline static int8_t int8_holding_array[3072]; //max size is for linear layer to process each move embd to a single value, 32_hidden_dim_1, 32xhidden_dim = 3072
     // max size is for expansion linear layer transforming the move embd, 32_any_hidden_dim, 32xhidden_dim=3072
    inline static float float_storage_array[hidden_dim];
    
    inline static int policy_logits_counter = 0;

    int32_t* float_holding_array_int32_ptr = reinterpret_cast<int32_t*>(float_holding_array);


public:
    static void reset_policy_counter() {
        policy_logits_counter = 0;
    }

    inline static float float_holding_array[3072]; //made public so can access directly, no need extra overhead of get() function
    inline static float logits_array[332];
    Layer(size_t n, size_t k, size_t m, size_t bs_A, size_t bs_B) : N(n), K(k), M(m), blockSizeA(bs_A), blockSizeB(bs_B) {}

    void operator()(float* input_values, float* coeff, bool distinct_coeff_across_dim, int8_t* matmul_params, int32_t* bias_params, move_indices* legal_moves_indices_array, int n_legal_moves, bool store, int n_additions_of_array, int size_stored_array, bool store_logits) {
        if (n_legal_moves !=-1) {
            N = n_legal_moves;
        }
        if (coeff != nullptr) {
            LPA_SP_in_place(coeff, input_values, distinct_coeff_across_dim, degree, N, K);
        }

        if (legal_moves_indices_array == nullptr && n_legal_moves == -1) {
            scale_float32_int8(input_values, int8_holding_array, N, K, scaling_factor_float_int8);
        }
        // else if (legal_moves_indices_array == nullptr) {
            
        // }
        // else {
        //     fill_move_embd(legal_moves_indices_array, n_legal_moves);
        // }
        
        matmul_simd_blocked_large(int8_holding_array, matmul_params, float_holding_array_int32_ptr, N, K, M, blockSizeA, blockSizeB);
        add_bias(float_holding_array_int32_ptr, bias_params, N, M);
        scale_int32_float32_in_place(float_holding_array_int32_ptr, N, M, scaling_factor_int32_float);
        if (n_additions_of_array != -1) {
            add_stored_array(n_additions_of_array, size_stored_array);
        }
        if (store) {
            store_float_array(float_holding_array, size_stored_array, float_storage_array);
        }
        if (store_logits) {
            store_float_array(float_holding_array, n_legal_moves, &logits_array[policy_logits_counter]);
            policy_logits_counter += n_legal_moves;
        }

    }

    void add_stored_array(int n_additions_of_array, int size_stored_array) {
        for (int j=0; j<8; j+=8) {
            __m256 sub_adding_arr = _mm256_load_ps(&float_holding_array[j]);
            for (int i=0; i<n_additions_of_array; i++) {
                __m256 sub_vec = _mm256_load_ps(&float_holding_array[i*size_stored_array+j]);
                sub_vec = _mm256_add_ps(sub_vec, sub_adding_arr);
                _mm256_load_ps(&float_holding_array[i*size_stored_array+j]);
            }
        }
    }

    void store_float_array(float* arr_to_store, int array_size, float* store_location) {
        int i=0;
        for (; i+7<array_size; i+=8) {
            __m256 store_ps_register = _mm256_load_ps(&arr_to_store[i]);
            _mm256_store_ps(&store_location[i], store_ps_register);
        }
        //manually allocate remaining if size not divisible by 8
        //delete code if not req
        for (; i<array_size; i++) {
            store_location[i] = arr_to_store[i];
        }
    }

};

int main() {
    //input array -> (1,input_size)
    alignas(32) float input_array[input_size];
    alignas(32) move_indices legal_moves_indices_array[332];
    int n_legal_moves = 0;
    size_t n_value_bins = hidden_dim;

    
    //load random values
    std::random_device rd;  // Seed generator
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_int_distribution<int> dist(0, 99); // Range [0, 99]

    for (int i = 0; i < input_size; i++) {input_array[i] = dist(gen);}

    for (int i = 0; i < input_size*hidden_dim; i++) {mat_params_ib[i] = dist(gen);}
    for (int i = 0; i < hidden_dim; i++) {bias_ib[i] = dist(gen);}

    for (int i = 0; i < 8*n_eval_passes; i++) {coeff_eb_L1[i] = dist(gen);}
    for (int i = 0; i < hidden_dim*hidden_dim; i++) {mat_params_eb_L1[i] = dist(gen);}
    for (int i = 0; i < hidden_dim; i++) {bias_eb_L1[i] = dist(gen);}

    for (int i = 0; i < 8*n_eval_passes; i++) {coeff_eb_L2[i] = dist(gen);}
    for (int i = 0; i < hidden_dim*hidden_dim; i++) {mat_params_eb_L2[i] = dist(gen);}
    for (int i = 0; i < hidden_dim; i++) {bias_eb_L2[i] = dist(gen);}

    for (int i = 0; i < 8*hidden_dim; i++) {coeff_vh[i] = dist(gen);}
    for (int i = 0; i < hidden_dim*hidden_dim; i++) {mat_params_vh[i] = dist(gen);}
    for (int i = 0; i < hidden_dim; i++) {bias_vh[i] = dist(gen);}

    Layer input_layer(1,input_size,hidden_dim,hidden_dim,hidden_dim);
    Layer eval_L1(1,hidden_dim,hidden_dim,hidden_dim,hidden_dim);
    Layer eval_L2(1,hidden_dim,hidden_dim,hidden_dim,hidden_dim);
    Layer generate_move_embeddings(1,32,hidden_dim,32,hidden_dim);
    Layer value_layer(1,hidden_dim,hidden_dim,hidden_dim,hidden_dim);
    float* residual_stream = input_layer.float_holding_array;
    float* logits_ptr = input_layer.logits_array;
    int32_t total_time = 0;
    for (int i=0; i<NUM_RUNS; i++) {
        Layer::reset_policy_counter();
        
        
        auto start = std::chrono::high_resolution_clock::now();

        input_layer(input_array, nullptr, true, mat_params_ib, bias_ib, nullptr, -1, true, -1, -1, false);
        for (int i=0; i<n_eval_passes-1; i++) {
            eval_L1(residual_stream, &coeff_eb_L1[i*(degree+1)], false, mat_params_eb_L1, bias_eb_L1, nullptr, -1, false, -1, -1, false);
            eval_L2(residual_stream, &coeff_eb_L2[i*(degree+1)], false, mat_params_eb_L2, bias_eb_L2, nullptr, -1, false, 1, hidden_dim, false);
        }
        eval_L1(residual_stream, &coeff_eb_L1[(n_eval_passes-1)*(degree+1)], true, mat_params_eb_L1, bias_eb_L1, nullptr, -1, false, -1, -1, false);
        eval_L2(residual_stream, &coeff_eb_L2[(n_eval_passes-1)*(degree+1)], true, mat_params_eb_L2, bias_eb_L2, nullptr, -1, true, -1, -1, false); //on final eval pass dont add back input vector, and store result
        value_layer(residual_stream, coeff_vh, true, mat_params_vh, bias_vh, nullptr, -1, false, -1, -1, -1);

        auto end = std::chrono::high_resolution_clock::now();
        
        // Get duration in microseconds and accumulate
        total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    }
    double average_time = static_cast<double>(total_time) / NUM_RUNS;

    std::cout << "Average Execution time: " << average_time << " microseconds\n";

    return 0;
}