/*
@brief Generates equidistributed random numbers.
@param states Array of the random generators states.
@param output Equidistributed random numbers, result is written as follows:
              random1(state1)..randomN(state1)
              random1(stateK)..randomN(stateK).
@param rounds Number of rounds for generation,
              each round generates 128 bytes for each random state.
@details global_size is the number of states, and algorithm is described in Wikipedia:

http://en.wikipedia.org/wiki/Xorshift

Written in 2014 by Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>.

The following generator has 1024 bits of state, a maximal period of 2^1024 − 1, passes BigCrush,
emit a sequence of 64-bit values that is equidistributed in the maximum possible dimension.

This is xorshift1024* from http://xorshift.di.unimi.it.

uint64_t s[ 16 ];
int p;

uint64_t next(void) {
  uint64_t s0 = s[ p ];
  uint64_t s1 = s[ p = ( p + 1 ) & 15 ];
  s1 ^= s1 << 31; // a
  s1 ^= s1 >> 11; // b
  s0 ^= s0 >> 30; // c
  return ( s[ p ] = s0 ^ s1 ) * 1181783497276652981LL;
}
*/
__kernel
void random(__global ulong /* IN, OUT */    *states,
            const int           /* IN */    rounds,
            __global ulong     /* OUT */    *output) {
  int id = get_global_id(0) << 4;
  ulong s[16];
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    s[i] = states[id + i];
  }

  for (int round = 0, offs = id * rounds; round < rounds; round++, offs += 16) {
    #pragma unroll
    for (int i = 0, p = 0; i < 16; i++) {
      ulong s0 = s[ p ];
      ulong s1 = s[ p = ( p + 1 ) & 15 ];
      s1 ^= s1 << 31; // a
      s1 ^= s1 >> 11; // b
      s0 ^= s0 >> 30; // c
      output[offs + i] = ( s[ p ] = s0 ^ s1 ) * 1181783497276652981;
    }
  }

  #pragma unroll
  for (int i = 0; i < 16; i++) {
    states[id + i] = s[i];
  }
}

/*
@brief Generates equidistributed random numbers.
@param state Random generators state (updated inside).
@param random The resulting random number.

Written in 2014 by Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>.

   This is the fastest generator passing BigCrush without systematic
   errors, but due to the relatively short period it is acceptable only
   for applications with a very mild amount of parallelism; otherwise, use
   a xorshift1024* generator.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to pass it twice through MurmurHash3's
   avalanching function.

uint64_t s[ 2 ];

uint64_t next(void) {
  uint64_t s1 = s[ 0 ];
  const uint64_t s0 = s[ 1 ];
  s[ 0 ] = s0;
  s1 ^= s1 << 23; // a
  return ( s[ 1 ] = ( s1 ^ s0 ^ ( s1 >> 17 ) ^ ( s0 >> 26 ) ) ) + s0; // b, c
}
*/
#define xorshift128plus(state, random) \
  do { \
    ulong2 seed = state; \
    ulong s1 = seed.x; \
    const ulong s0 = seed.y; \
    seed.x = s0; \
    s1 ^= s1 << 23; \
    random = (seed.y = (s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26))) + s0; \
    state = seed; \
  } while (0)

