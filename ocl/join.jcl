#include "defines.cl"
#include "highlight.cl"


inline void copy(const __global etype *input, const int size,
                 __global etype *output) {
  for (int i = 0; i < size; i++) {
    output[i] = input[i];
  }
}

/// @brief Copies several buffers into one.
__kernel
void join(__global etype *output
{% for i in range(inputs|length) %}
    , __global const etype *input{{ i }}
{% endfor %}
  ) {

  int index = get_global_id(0);
  const int sizes[] = {
   {% for input in inputs %}
     {{ input.size // input.shape[0] }},
   {% endfor %}
  };
  const int inputs_number = {{ inputs|length }};
  int output_size = 0;
  for (int i = 0; i < inputs_number; i++) {
    output_size += sizes[i];
  }
  int output_offset = index * output_size;
  int input_size = 0;

  {% for i in range(inputs|length) %}
    input_size = sizes[{{ i }}];
    copy(input{{ i }} + index * input_size, input_size, output + output_offset);
    output_offset += input_size;
  {% endfor %}
}
