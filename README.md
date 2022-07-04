# LearningAVX
AVX2 calculator. Measured perfomance difference between standard and AVX2 code.

Basic arithmetic operations (addition, substraction, multiplication and division) of 2 std::vectors <int> provided. For add/sub/mullo/div of 2 std::vectors call one single function:
    
    _AVX_compute_int32_vectors(vec1, vec2, &AVX::add, &Arithmetic::add);

As of now previously created separate functions, each of which providing only 1 arithmetic operations kept for comparison only purposes. They use both aligned and unaligned vectors with AVX aligned/unaligned load & store respectively.

Measurements taken using std::chrono::steady_clock shows that created AVX2 functions speeds up arithmetic operations about 6-7X times. Let's take on example with 2 vectors consisting of 1 millions elements each.

    _AVX_compute_int32_vectors(vec1, vec2, &AVX::add, &Arithmetic::add);
    //5087 microseconds
    _AVX_add_int32_vectors(vec1, vec2);
    //4481 microseconds
    _AVX_halved_add_int32_vectors(vec1, vec2);
    //4245 microseconds
    for (auto i = 0; i < vec1.size(); ++i) {
        vec1[i] += vec2[i];
    }
    //33258 microseconds

Overall about 6X faster. Same speedup in other basic arithmetic operations. Not the most stable benchmark possible, results may be different in 4-8X range.
