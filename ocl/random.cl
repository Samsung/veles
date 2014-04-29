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

The following generator has 1024 bits of state, a maximal period of 2^1024 − 1, passes BigCrush,
emit a sequence of 64-bit values that is equidistributed in the maximum possible dimension.

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
            __global ulong     /* OUT */    *output,
            const int           /* IN */    rounds) {
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
