#include <iostream>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <climits>
#include <omp.h>
#include <immintrin.h>

#include "aligned_vector.hpp"

using namespace std;

typedef std::chrono::steady_clock::time_point tp;

class Time {
public:
    static void show(tp t1, tp t2) { //time passed since t1
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << '\t';
        printf("nanoseconds \n");
    }
    tp add() {
        tp p = std::chrono::steady_clock::now();
        return p;
    }
};

void _AVX_add_int32_vectors(vector <int>& v1, vector <int>& v2) { //long, but fast
    /*basically does
    for (auto i = 0; i < v1.size(); i++) {
        v1[i] += v2[i];
    }*/
    for (auto i = 0; i < v1.size() - 8; i += 8) {
        __m256i _vect1 = _mm256_loadu_si256((__m256i*) &v1[i]);
        __m256i _vect2 = _mm256_loadu_si256((__m256i*) &v2[i]);
        _vect1 = _mm256_add_epi32(_vect1, _vect2);
        _mm256_store_si256((__m256i*) & v1[i], _vect1);

    }
    for (auto i = v1.size() - v1.size() % 8; i < v1.size(); i += 8) {
        auto tmp = v1.size() % 8;
        ++tmp;
        char chArr[8] = {};
        for (auto j = 0; j < 8; ++j) {
            chArr[j] -= --tmp;
        } //masks slower, but needed for computing last 8 elements
        __m256i _mask = _mm256_setr_epi32(chArr[0],
            chArr[1], chArr[2], chArr[3], chArr[4], chArr[5], chArr[6], chArr[7]);
        __m256i _vect1 = _mm256_maskload_epi32(&v1[i], _mask);
        __m256i _vect2 = _mm256_maskload_epi32(&v2[i], _mask);
        _vect1 = _mm256_add_epi32(_vect1, _vect2);
        _mm256_maskstore_epi32(&v1[i], _mask, _vect1);
    }
}
void _AVX_add_int32_vectors_slow(vector <int>& v1, vector <int>& v2) {//simple, but slow 
    /*basically does
    for (auto i = 0; i < v1.size(); i++) {
        v1[i] += v2[i];
    }*/
    for (auto i = 0; i < v1.size(); i += 8) {
        auto tmp = v1.size() % 8;
        ++tmp;
        char chArr[8] = {};
        for (auto j = 0; j < 8; ++j) {
            chArr[j] -= --tmp;
        } //masks slower, but needed for computing last 8 elements
        __m256i _mask = _mm256_setr_epi32(chArr[0],
            chArr[1], chArr[2], chArr[3], chArr[4], chArr[5], chArr[6], chArr[7]);
        __m256i _vect1 = _mm256_maskload_epi32(&v1[i], _mask);
        __m256i _vect2 = _mm256_maskload_epi32(&v2[i], _mask);
        _vect1 = _mm256_add_epi32(_vect1, _vect2);
        _mm256_maskstore_epi32(&v1[i], _mask, _vect1);
    }
}
void _AVX_substract_int32_vectors(vector <int>& v1, vector <int> & v2) {
    /*basically does
    for (auto i = 0; i < v1.size(); i++) {
        v1[i] -= v2[i];
    }*/
    for (auto i = 0; i < v1.size() - 8; i += 8) {
        __m256i _vect1 = _mm256_loadu_si256((__m256i*) &v1[i]);
        __m256i _vect2 = _mm256_loadu_si256((__m256i*) &v2[i]);
        _vect1 = _mm256_sub_epi32(_vect1, _vect2);
        _mm256_store_si256((__m256i*) & v1[i], _vect1);

    }
    for (auto i = v1.size() - v1.size() % 8; i < v1.size(); i += 8) {
        auto tmp = v1.size() % 8;
        ++tmp;
        char chArr[8] = {};
        for (auto j = 0; j < 8; ++j) {
            chArr[j] -= --tmp;
        } //masks slower, but needed for computing last 8 elements
        __m256i _mask = _mm256_setr_epi32(chArr[0],
            chArr[1], chArr[2], chArr[3], chArr[4], chArr[5], chArr[6], chArr[7]);
        __m256i _vect1 = _mm256_maskload_epi32(&v1[i], _mask);
        __m256i _vect2 = _mm256_maskload_epi32(&v2[i], _mask);
        _vect1 = _mm256_sub_epi32(_vect1, _vect2);
        _mm256_maskstore_epi32(&v1[i], _mask, _vect1);
    }
    
}
void _AVX_add_int32_aligned_vectors(is::aligned_vector<int32_t, 32>& v1,
                    is::aligned_vector<int32_t, 32>& v2) {
    for (auto i = 0; i < v1.size() - 8; i += 8) {
        __m256i _vect1 = _mm256_load_si256((__m256i*) & v1[i]);
        __m256i _vect2 = _mm256_load_si256((__m256i*) & v2[i]);
        _vect1 = _mm256_add_epi32(_vect1, _vect2);
        _mm256_store_si256((__m256i*) & v1[i], _vect1);

    }
    for (auto i = v1.size() - v1.size() % 8; i < v1.size(); i += 8) {
        auto tmp = v1.size() % 8;
        ++tmp;
        char chArr[8] = {};
        for (auto j = 0; j < 8; ++j) {
            chArr[j] -= --tmp;
        } //masks slower, but needed for computing last 8 elements
        __m256i _mask = _mm256_setr_epi32(chArr[0],
            chArr[1], chArr[2], chArr[3], chArr[4], chArr[5], chArr[6], chArr[7]);
        __m256i _vect1 = _mm256_maskload_epi32(&v1[i], _mask);
        __m256i _vect2 = _mm256_maskload_epi32(&v2[i], _mask);
        _vect1 = _mm256_add_epi32(_vect1, _vect2);
        _mm256_maskstore_epi32(&v1[i], _mask, _vect1);
    }

}
int main()
{
    Time time;
    is::aligned_vector<int32_t, 32> alvec1;
    is::aligned_vector<int32_t, 32> alvec2;
    vector <int> vec1 = {};
    vector <int> vec2 = {};
    for (auto i = 0; i < SHRT_MAX*100; i++) {
        vec1.push_back(i);
        vec2.push_back(i << 1);
        alvec1.push_back(i);
        alvec2.push_back(i << 1);
    }
    auto t11 = time.add();
    _AVX_add_int32_aligned_vectors(alvec1, alvec2);
    auto t12 = time.add();
    auto t1 = time.add();
    //_AVX_add_int32_vectors(vec1, vec2);
    auto t2 = time.add();
    for (auto i = 0; i < vec1.size(); i++) {
        vec1[i] += vec2[i];
    }
    auto t3 = time.add();
    for (auto i = 0; i < vec1.size(); i++) {
        vec1[i] += vec2[i];
    }
    auto t4 = time.add();
    //_AVX_substract_int32_vectors(vec1, vec2);
    auto t5 = time.add();
    time.show(t11, t12);
    time.show(t1, t2);
    time.show(t2, t3);
    time.show(t3, t4);
    time.show(t4,t5);
	return 0;
}