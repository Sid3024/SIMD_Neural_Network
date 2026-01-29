#ifndef SIMD_FUNCTIONS_H
#define SIMD_FUNCTIONS_H
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

inline float hsum_avx(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);                 // Extract lower 128 bits
    __m128 vhigh = _mm256_extractf128_ps(v, 1);               // Extract upper 128 bits
    __m128 sum = _mm_add_ps(vlow, vhigh);                      // Add lower and upper parts
    sum = _mm_hadd_ps(sum, sum);                                // Horizontal add
    sum = _mm_hadd_ps(sum, sum);                                // Horizontal add again
    return _mm_cvtss_f32(sum);                                  // Extract the sum
}

// Helper function to compute the horizontal maximum of a __m256 vector
inline float hmax_avx(__m256 v) {
    // Step 1: Compare upper and lower 128 bits
    __m256 temp = _mm256_max_ps(v, _mm256_permute2f128_ps(v, v, 1));
    
    // Step 2: Shuffle and compare within 128 bits
    temp = _mm256_max_ps(temp, _mm256_shuffle_ps(temp, temp, 0x4E));
    temp = _mm256_max_ps(temp, _mm256_shuffle_ps(temp, temp, 0xB1));
    
    // Step 3: Extract the lower 128 bits and perform horizontal max
    __m128 tmp128 = _mm256_castps256_ps128(temp);
    // tmp128 = _mm_max_ps(tmp128, _mm_movehl_ps(tmp128, tmp128));
    // tmp128 = _mm_max_ps(tmp128, _mm_shuffle_ps(tmp128, tmp128, 0x1));
    
    // Step 4: Extract the maximum value from the lower 128 bits
    return _mm_cvtss_f32(tmp128);
}

//LINEAR LAYER

// Function for blocked matrix multiplication (primarily for larger matrices)
void matmul_simd_blocked_large(
    const int8_t* A,
    const int8_t* B_col,
    int32_t* C,
    size_t N,
    size_t K,
    size_t M,
    size_t blockSizeA,
    size_t blockSizeB
) {
    // Result Matrix C: N x M (row-major)
    //std::vector<int32_t> C(N * M, 0);
    //std::fill(C, C + N * M, 0);
    for (size_t i0 = 0; i0 < N; i0 += blockSizeA) {
        size_t i_max = std::min(i0 + blockSizeA, N);
        for (size_t j0 = 0; j0 < M; j0 += blockSizeB) {
            size_t j_max = std::min(j0 + blockSizeB, M);
            for (size_t k0 = 0; k0 < K; k0 += blockSizeA) { // Using blockSizeA for k
                size_t k_max = std::min(k0 + blockSizeA, K);
                for (size_t i = i0; i < i_max; ++i) { // Iterate over rows within block
                    for (size_t j = j0; j < j_max; ++j) { // Iterate over columns within block
                        size_t k = k0;
                        __m256i sum_vec = _mm256_setzero_si256(); // Initialize SIMD accumulator
                        for (; k + 31 < k_max; k += 32) { // Process 32 elements at a time
                            // Load 32 int8_t elements from A (row-major)
                            __m256i va = _mm256_load_si256(reinterpret_cast<const __m256i*>(&A[i * K + k]));
                            
                            // Load 32 int8_t elements from B_col (column-major)
                            __m256i vb = _mm256_load_si256(reinterpret_cast<const __m256i*>(&B_col[j * K + k]));
                            
                            // Multiply and add adjacent bytes: (A[k] * B[j][k]) + (A[k+1] * B[j][k+1]), etc.
                            __m256i prod = _mm256_maddubs_epi16(va, vb);
                            
                            // Convert lower and upper 128 bits from int16_t to int32_t
                            __m256i prod_low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod));
                            __m256i prod_high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod, 1));
                            
                            // Accumulate the results
                            sum_vec = _mm256_add_epi32(sum_vec, prod_low);
                            sum_vec = _mm256_add_epi32(sum_vec, prod_high);
                        }
                        
                        // Horizontal summation to combine SIMD accumulator into a single scalar
                        __m128i sum_low = _mm256_castsi256_si128(sum_vec); // Lower 128 bits
                        __m128i sum_high = _mm256_extracti128_si256(sum_vec, 1); // Upper 128 bits
                        __m128i sum_pair = _mm_add_epi32(sum_low, sum_high); // Add low and high parts
                        
                        sum_pair = _mm_hadd_epi32(sum_pair, sum_pair); // First horizontal add
                        sum_pair = _mm_hadd_epi32(sum_pair, sum_pair); // Second horizontal add
                        
                        int32_t total = _mm_cvtsi128_si32(sum_pair); // Extract the result
                        
                        // Handle remaining elements that don't fit into SIMD processing
                        for (; k < k_max; ++k) {
                            total += static_cast<int32_t>(A[i * K + k]) * static_cast<int32_t>(B_col[j * K + k]);
                        }
                        
                        C[i * M + j] += total; // Update the result matrix
                    }
                }
            }
        }
    }
    
    // Example: Return sum of all elements in C (for verification)
    // int32_t matmul_sum = 0;
    // for (size_t i = 0; i < N * M; ++i) {
    //     matmul_sum += C[i];
    // }
    // return matmul_sum;
}

//adding bias after matmul
void add_bias(int32_t* C, const int32_t* bias, size_t N, size_t M) {
    for (size_t i=0; i<N; i++) {
        size_t j=0;
        for (; j+7<M; j+=8) {
            __m256i sub_vc = _mm256_load_si256(reinterpret_cast<const __m256i*>(&C[i*M+j]));
            __m256i vb = _mm256_load_si256(reinterpret_cast<const __m256i*>(&bias[j]));
            sub_vc = _mm256_add_epi32(sub_vc, vb);
            _mm256_store_si256(reinterpret_cast<__m256i*>(&C[i*M+j]), sub_vc);
        }
        // for (; j<M; ++j) {
        //     C[i*M+j] += bias[j];
        // }
    }
}

//LPA
//coeff stored as follows: first shapeOne elements are the coeff for degree 0 for each dim of shapeOne,
//next shapeOne elements are the coeff for degree 1 for each dim of shapeOne, and so on
void LPA_in_place(const float* coeff, float* inp, size_t degree, size_t shapeZero, size_t shapeOne) { 
    for (size_t i=0; i<shapeZero; i++) {
        size_t j=0;
        for (; j+7<shapeOne; j+=8) { 
            __m256 accumulation_vector = _mm256_load_ps(&coeff[degree * shapeOne + j]);
            __m256 sub_inp = _mm256_load_ps(&inp[i * shapeOne + j]);
            for (int k=degree-1; k>=0; k--) { // need to start horner's method from the degree, not 0
                __m256 sub_coeff = _mm256_load_ps(&coeff[k * shapeOne + j]);
                accumulation_vector = _mm256_mul_ps(accumulation_vector, sub_inp);
                accumulation_vector = _mm256_add_ps(accumulation_vector, sub_coeff);
            }
            _mm256_store_ps(&inp[i * shapeOne + j], accumulation_vector);
        }
    }
}

//LPA
void LPA_SP_in_place(const float* coeff, float* inp, size_t degree, bool distinct_coeff_across_dim, size_t shapeZero, size_t shapeOne) { 
    __m256 sub_coeff;
    float coeff_value;
    for (size_t i=0; i<shapeZero; i++) {
        size_t j=0;
        for (; j+7<shapeOne; j+=8) { 
            __m256 accumulation_vector = _mm256_setzero_ps();
            __m256 sub_inp = _mm256_load_ps(&inp[i * shapeOne + j]);
            for (int k=degree; k>=0; k--) { // need to start horner's method from the degree, not 0
                if (distinct_coeff_across_dim) {
                    sub_coeff = _mm256_load_ps(&coeff[k * shapeOne + j]);
                }
                else {
                    coeff_value = coeff[k];
                    sub_coeff = _mm256_set1_ps(coeff_value);
                }
                accumulation_vector = _mm256_mul_ps(accumulation_vector, sub_inp);
                accumulation_vector = _mm256_add_ps(accumulation_vector, sub_coeff);
            }
            _mm256_store_ps(&inp[i * shapeOne + j], accumulation_vector);
        }
    }
}



void softmax_simd(float* input, const float* coeff, size_t shapeZero, size_t shapeOne, size_t degree) {
    for (size_t i = 0; i < shapeZero; i++) {
        // Step 1: Find the maximum value in the current row
        __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        size_t j = 0;
        for (; j + 7 < shapeOne; j += 8) {
            __m256 val = _mm256_load_ps(&input[i * shapeOne + j]);
            max_vec = _mm256_max_ps(max_vec, val);
        }
        // Horizontal max
        float overall_max = hmax_avx(max_vec);
        // Handle remaining elements
        for (; j < shapeOne; j++) {
            if (input[i * shapeOne + j] > overall_max) {
                overall_max = input[i * shapeOne + j];
            }
        }

        // Step 2: Subtract the max value from each element in the row
        __m256 max_vec_broadcast = _mm256_set1_ps(overall_max);
        j = 0;
        for (; j + 7 < shapeOne; j += 8) {
            __m256 val = _mm256_load_ps(&input[i * shapeOne + j]);
            __m256 adjusted = _mm256_sub_ps(val, max_vec_broadcast);
            _mm256_store_ps(&input[i * shapeOne + j], adjusted);
        }
        // Handle remaining elements
        for (; j < shapeOne; j++) {
            input[i * shapeOne + j] -= overall_max;
        }

        // Step 3: Apply polynomial approximation to compute exp(x) for each element
        LPA_in_place(coeff, &input[i * shapeOne], degree, 1, shapeOne);

        // Step 4: Compute the sum of exponentials in the row
        __m256 sum_vec = _mm256_setzero_ps();
        j = 0;
        for (; j + 7 < shapeOne; j += 8) {
            __m256 val = _mm256_load_ps(&input[i * shapeOne + j]);
            sum_vec = _mm256_add_ps(sum_vec, val);
        }
        // Horizontal sum of sum_vec
        float exponential_sum = hsum_avx(sum_vec);
        // Handle remaining elements
        for (; j < shapeOne; j++) {
            exponential_sum += input[i * shapeOne + j];
        }

        // Step 5: Normalize the exponentials by the sum to get softmax probabilities
        __m256 sum_vec_broadcast = _mm256_set1_ps(exponential_sum);
        j = 0;
        for (; j + 7 < shapeOne; j += 8) {
            __m256 val = _mm256_load_ps(&input[i * shapeOne + j]);
            __m256 normalized = _mm256_div_ps(val, sum_vec_broadcast);
            _mm256_store_ps(&input[i * shapeOne + j], normalized);
        }
        // Handle remaining elements
        for (; j < shapeOne; j++) {
            input[i * shapeOne + j] /= exponential_sum;
        }
    }
}


//scaling
void scale_int32_float32_in_place(int32_t* data, size_t shapeZero, size_t shapeOne, float scaling_factor) {
    __m256 scaling_vec = _mm256_set1_ps(scaling_factor);
    for (size_t i=0; i<shapeZero; i++) {
        for (size_t j=0; j+7<shapeOne; j+=8) {
            __m256i sub_data = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i * shapeOne + j]));
            __m256 sub_data_f = _mm256_cvtepi32_ps(sub_data);
            sub_data_f = _mm256_mul_ps(sub_data_f, scaling_vec);
            _mm256_store_ps(reinterpret_cast<float*>(&data[i * shapeOne + j]), sub_data_f);

        }
        
    }

}

void scale_float32_int8(float* in_values, int8_t* out_values, size_t shapeZero, size_t shapeOne, float scaling_factor) {
    __m256 scaling_vec = _mm256_set1_ps(scaling_factor);
    __m256 min_value_vec = _mm256_set1_ps(-128.0f);
    __m256 max_value_vec = _mm256_set1_ps(127.0f);
    alignas(32) __m256i m256i_array[4];
    for (size_t i=0; i<shapeZero; i++) {
        for (size_t j=0; j+31<shapeOne; j+=32) {
            for (int8_t k=0; k<4; k++) {
                __m256 sub_in_values = _mm256_load_ps(&in_values[i * shapeOne + j + 8 * k]);
                sub_in_values = _mm256_mul_ps(sub_in_values, scaling_vec);
                __m256 rounded = _mm256_round_ps(sub_in_values,_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                rounded = _mm256_max_ps(rounded, min_value_vec);
                rounded = _mm256_min_ps(rounded, max_value_vec);
                __m256i int32_vec = _mm256_cvtps_epi32(rounded);
                m256i_array[k] = int32_vec;
                //storing as pointer isnt possible bcos int32_vec will be destoyed once out of for loop scope
                //storing on heap will take too long for memory alloc & freeing
                //most optimized solution is to manually roll out for loop (type out the code 4 times manually), not implemented here
            }
            __m256i int16_vec_0 = _mm256_packs_epi32(m256i_array[0], m256i_array[1]);
            __m256i int16_vec_1 = _mm256_packs_epi32(m256i_array[2], m256i_array[3]);
            __m256i int8_vec = _mm256_packs_epi16(int16_vec_0, int16_vec_1);
            _mm256_store_si256(reinterpret_cast<__m256i*>(&out_values[i * shapeOne + j]), int8_vec);
        }
    }
}





// Function to find the index of the maximum value in the first n elements of a float array
size_t find_max_index(const float* array, size_t n) {
    // Ensure that n is divisible by 8
    if (n % 8 != 0) {
        // Handle error as per your requirements
        // For this implementation, we'll return 0
        return 0;
    }

    __m256 current_max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity()); // Initialize to the smallest possible float
    float current_max = -std::numeric_limits<float>::infinity();
    size_t max_index = 0;

    for (size_t i = 0; i < n; i += 8) {
        // Load 8 floats from the array
        __m256 block = _mm256_loadu_ps(&array[i]);

        // Compute the maximum value in the current block
        float block_max = hmax_avx(block);

        // If the block maximum is greater than the current maximum, update it
        if (block_max > current_max) {
            // Broadcast the block maximum to compare
            __m256 block_max_vec = _mm256_set1_ps(block_max);

            // Compare each element in the block with the block_max to create a mask
            __m256 cmp = _mm256_cmp_ps(block, block_max_vec, _CMP_EQ_OQ);

            // Create a bitmask from the comparison results
            int mask = _mm256_movemask_ps(cmp);

            if (mask != 0) { // Ensure at least one element matches
                // Find the index of the first set bit in the mask
                // _tzcnt_u32 counts the number of trailing zeros
                // Ensure that the CPU supports BMI1 (as per CPU flags)
                int offset = _tzcnt_u32(mask);

                // Update the current maximum and its index
                current_max = block_max;
                max_index = i + offset;
            }
        }
    }

    return max_index;
}



#endif