// Check for required defines
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE MUST be defined"
#endif
#ifndef A_WIDTH
#error "A_WIDTH MUST be defined"
#endif
#ifndef B_WIDTH
#error "B_WIDTH MUST be defined"
#endif
#ifndef AB_COMMON
#error "AB_COMMON MUST be defined"
#endif
#ifndef STORE_OUTPUT
#error "STORE_OUTPUT MUST be defined"
#endif

// Support for multiple includes
#ifdef ALIGNED
#undef ALIGNED
#endif

// Set default precision level
#ifndef PRECISION_LEVEL
#define PRECISION_LEVEL 0
#endif

// Choose to vectorize block sum or not
#ifndef VECTOR_OPT
#define VECTOR_OPT 0
#endif
#if (VECTOR_OPT <= 0) || (BLOCK_SIZE % 4 != 0)
#define BLOCK_SUM_VECTORIZED 0
#else
#ifndef MULTIPLY
#define BLOCK_SUM_VECTORIZED 1
#else
#define BLOCK_SUM_VECTORIZED 0
#endif
#endif

// Compute number of blocks at the common matrix side
#if (AB_COMMON % BLOCK_SIZE) == 0
#define N_BLOCKS (AB_COMMON / BLOCK_SIZE)
#if ((A_WIDTH % BLOCK_SIZE) == 0) && ((B_WIDTH % BLOCK_SIZE) == 0)
#define ALIGNED
#endif
#else
#define N_BLOCKS (AB_COMMON / BLOCK_SIZE + 1)
#endif

// Define default offsets
#ifndef A_REAL_OFFS
#define A_REAL_OFFS a_offs
#endif
#ifndef B_REAL_OFFS
#define B_REAL_OFFS b_offs
#endif
#ifndef A_REAL_OFFS_VALID
#define A_REAL_OFFS_VALID 1
#endif
#ifndef B_REAL_OFFS_VALID
#define B_REAL_OFFS_VALID 1
#endif
