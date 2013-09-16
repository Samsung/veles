/* EINA - EFL data type library
 * Copyright (C) 2007-2008 Carsten Haitzler, Vincent Torri, Jorge Luis Zapata Muga
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library;
 * if not, see <http://www.gnu.org/licenses/>.
 */

#ifndef EINA_TYPES_H_
#define EINA_TYPES_H_

/**
 * @addtogroup Eina_Core_Group Core
 *
 * @{
 */

/**
 * @defgroup Eina_Types_Group Types
 *
 * @{
 */

#ifdef EAPI
# undef EAPI
#endif

#ifdef _WIN32
# ifdef EFL_EINA_BUILD
#  ifdef DLL_EXPORT
#   define EAPI __declspec(dllexport)
#  else
#   define EAPI
#  endif /* ! DLL_EXPORT */
# else
#  define EAPI __declspec(dllimport)
# endif /* ! EFL_EINA_BUILD */
#else
# ifdef __GNUC__
#  if __GNUC__ >= 4
#   define EAPI __attribute__ ((visibility("default")))
#  else
#   define EAPI
#  endif
# else
/**
 * @def EAPI
 * @brief Used to export functions(by changing visibility).
 */
#  define EAPI
# endif
#endif

#include "eina_config.h"

#ifdef EINA_UNUSED
# undef EINA_UNUSED
#endif
#ifdef EINA_WARN_UNUSED_RESULT
# undef EINA_WARN_UNUSED_RESULT
#endif
#ifdef EINA_ARG_NONNULL
# undef EINA_ARG_NONNULL
#endif
#ifdef EINA_DEPRECATED
# undef EINA_DEPRECATED
#endif
#ifdef EINA_MALLOC
# undef EINA_MALLOC
#endif
#ifdef EINA_PURE
# undef EINA_PURE
#endif
#ifdef EINA_PRINTF
# undef EINA_PRINTF
#endif
#ifdef EINA_SCANF
# undef EINA_SCANF
#endif
#ifdef EINA_FORMAT
# undef EINA_FORMAT
#endif
#ifdef EINA_CONST
# undef EINA_CONST
#endif
#ifdef EINA_NOINSTRUMENT
# undef EINA_NOINSTRUMENT
#endif
#ifdef EINA_UNLIKELY
# undef EINA_UNLIKELY
#endif
#ifdef EINA_LIKELY
# undef EINA_LIKELY
#endif
#ifdef EINA_SENTINEL
# undef EINA_SENTINEL
#endif

#ifdef __GNUC__

# if __GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1)
#  define EINA_UNUSED __attribute__ ((__unused__))
# else
#  define EINA_UNUSED
# endif

# if __GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
#  define EINA_WARN_UNUSED_RESULT __attribute__ ((__warn_unused_result__))
# else
#  define EINA_WARN_UNUSED_RESULT
# endif

# if (!defined(EINA_SAFETY_CHECKS)) && (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#  define EINA_ARG_NONNULL(...) __attribute__ ((__nonnull__(__VA_ARGS__)))
# else
#  define EINA_ARG_NONNULL(...)
# endif

# if __GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1)
#  define EINA_DEPRECATED __attribute__ ((__deprecated__))
# else
#  define EINA_DEPRECATED
# endif

# if __GNUC__ > 2 || (__GNUC__ == 2 && __GNUC_MINOR__ >= 96)
#  define EINA_MALLOC __attribute__ ((__malloc__))
#  define EINA_PURE   __attribute__ ((__pure__))
# else
#  define EINA_MALLOC
#  define EINA_PURE
# endif

# if __GNUC__ > 2 || (__GNUC__ == 2 && __GNUC_MINOR__ > 4)
#  if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 3)
#   define EINA_PRINTF(fmt, arg) __attribute__((__format__ (__gnu_printf__, fmt, arg)))
#  else
#   define EINA_PRINTF(fmt, arg) __attribute__((__format__ (__printf__, fmt, arg)))
#  endif
#  define EINA_SCANF(fmt, arg)  __attribute__((__format__ (__scanf__, fmt, arg)))
#  define EINA_FORMAT(fmt)      __attribute__((__format_arg__(fmt)))
#  define EINA_CONST        __attribute__((__const__))
#  define EINA_NOINSTRUMENT __attribute__((__no_instrument_function__))
#  define EINA_UNLIKELY(exp)    __builtin_expect((exp), 0)
#  define EINA_LIKELY(exp)      __builtin_expect((exp), 1)
#  define EINA_SENTINEL __attribute__((__sentinel__))
# else
#  define EINA_PRINTF(fmt, arg)
#  define EINA_SCANF(fmt, arg)
#  define EINA_FORMAT(fmt)
#  define EINA_CONST
#  define EINA_NOINSTRUMENT
#  define EINA_UNLIKELY(exp) exp
#  define EINA_LIKELY(exp)   exp
#  define EINA_SENTINEL
# endif

#elif defined(_MSC_VER)
# define EINA_UNUSED
# define EINA_WARN_UNUSED_RESULT
# define EINA_ARG_NONNULL(...)
# if _MSC_VER >= 1300
#  define EINA_DEPRECATED __declspec(deprecated)
# else
#  define EINA_DEPRECATED
# endif
# define EINA_MALLOC
# define EINA_PURE
# define EINA_PRINTF(fmt, arg)
# define EINA_SCANF(fmt, arg)
# define EINA_FORMAT(fmt)
# define EINA_CONST
# define EINA_NOINSTRUMENT
# define EINA_UNLIKELY(exp) exp
# define EINA_LIKELY(exp)   exp
# define EINA_SENTINEL

#elif defined(__SUNPRO_C)
# define EINA_UNUSED
# define EINA_WARN_UNUSED_RESULT
# define EINA_ARG_NONNULL(...)
# define EINA_DEPRECATED
# if __SUNPRO_C >= 0x590
#  define EINA_MALLOC __attribute__ ((malloc))
#  define EINA_PURE   __attribute__ ((pure))
# else
#  define EINA_MALLOC
#  define EINA_PURE
# endif
# define EINA_PRINTF(fmt, arg)
# define EINA_SCANF(fmt, arg)
# define EINA_FORMAT(fmt)
# if __SUNPRO_C >= 0x590
#  define EINA_CONST __attribute__ ((const))
# else
#  define EINA_CONST
# endif
# define EINA_NOINSTRUMENT
# define EINA_UNLIKELY(exp) exp
# define EINA_LIKELY(exp)   exp
# define EINA_SENTINEL

#else /* ! __GNUC__ && ! _MSC_VER && ! __SUNPRO_C */

/**
 * @def EINA_UNUSED
 * Used to warn when an argument of the function is not used.
 */
# define EINA_UNUSED

/**
 * @def EINA_WARN_UNUSED_RESULT
 * Used to warn when the returned value of the function is not used.
 */
# define EINA_WARN_UNUSED_RESULT

/**
 * @def EINA_ARG_NONNULL
 * Used to warn when the specified arguments of the function are @c NULL.
 */
# define EINA_ARG_NONNULL(...)

/**
 * @def EINA_DEPRECATED
 * Used to warn when the function is considered as deprecated.
 */
# define EINA_DEPRECATED
/**
 * @def EINA_MALLOC
 * @brief EINA_MALLOC is used to tell the compiler that a function may be treated
 * as if any non-NULL pointer it returns cannot alias any other pointer valid when
 * the function returns and that the memory has undefined content.
 */
# define EINA_MALLOC
/**
 * @def EINA_PURE
 * @brief EINA_PURE is used to tell the compiler this functions has no effects
 * except the return value and their return value depends only on the parameters
 * and/or global variables.
 */
# define EINA_PURE
/**
 * @def EINA_PRINTF
 * @param fmt The format to be used.
 * @param arg The argument to be used.
 */
# define EINA_PRINTF(fmt, arg)
/**
 * @def EINA_SCANF
 * @param fmt The format to be used.
 * @param arg The argument to be used.
 */
# define EINA_SCANF(fmt, arg)
/**
 * @def EINA_FORMAT
 * @param fmt The format to be used.
 */
# define EINA_FORMAT(fmt)
/**
 * @def EINA_CONST
 * @brief Attribute from gcc to prevent the function to read/modify any global memory.
 */
# define EINA_CONST
/**
 * @def EINA_NOINSTRUMENT
 * @brief Attribute from gcc to disable instrumentation for a specific function.
 */
# define EINA_NOINSTRUMENT
/**
 * @def EINA_UNLIKELY
 * @param exp The expression to be used.
 */
# define EINA_UNLIKELY(exp) exp
/**
 * @def EINA_LIKELY
 * @param exp The expression to be used.
 */
# define EINA_LIKELY(exp)   exp
/**
 * @def EINA_SENTINEL
 * @brief Attribute from gcc to prevent calls without the necessary NULL
 * sentinel in certain variadic functions
 * @since 1.7.0
 */
# define EINA_SENTINEL
#endif /* ! __GNUC__ && ! _WIN32 && ! __SUNPRO_C */

/**
 * @typedef Eina_Bool
 * Type to mimic a boolean.
 *
 * @note it differs from stdbool.h as this is defined as an unsigned
 *       char to make it usable by bitfields (Eina_Bool name:1) and
 *       also take as few bytes as possible.
 */
typedef unsigned char Eina_Bool;

/**
 * @def EINA_FALSE
 * boolean value FALSE (numerical value 0)
 */
#define EINA_FALSE ((Eina_Bool)0)

/**
 * @def EINA_TRUE
 * boolean value TRUE (numerical value 1)
 */
#define EINA_TRUE  ((Eina_Bool)1)

EAPI extern const unsigned int eina_prime_table[];

/**
 * @typedef Eina_Compare_Cb
 * Function used in functions using sorting. It compares @p data1 and
 * @p data2. If @p data1 is 'less' than @p data2, -1 must be returned,
 * if it is 'greater', 1 must be returned, and if they are equal, 0
 * must be returned.
 */
typedef int (*Eina_Compare_Cb)(const void *data1, const void *data2);

/**
 * @def EINA_COMPARE_CB
 * Macro to cast to Eina_Compare_Cb.
 */
#define EINA_COMPARE_CB(function) ((Eina_Compare_Cb)function)

/**
 * @typedef Eina_Random_Cb
 * Function used in shuffling functions. An integer betwen min and max
 * inclusive must be returned.
 *
 * @since 1.8
 */
typedef int (*Eina_Random_Cb)(const int min, const int max);

/**
 * @def EINA_RANDOM_CB
 * Macro to cast to Eina_Random_Cb.
 */
#define EINA_RANDOM_CB(function) ((Eina_Random_Cb)function)

/**
 * @typedef Eina_Each_Cb
 * A callback type used when iterating over a container.
 */
typedef Eina_Bool (*Eina_Each_Cb)(const void *container, void *data, void *fdata);

/**
 * @def EINA_EACH_CB
 * Macro to cast to Eina_Each.
 */
#define EINA_EACH_CB(Function) ((Eina_Each_Cb)Function)

/**
 * @typedef Eina_Free_Cb
 * A callback type used to free data when iterating over a container.
 */
typedef void (*Eina_Free_Cb)(void *data);

/**
 * @def EINA_FREE_CB
 * Macro to cast to Eina_Free_Cb.
 */
#define EINA_FREE_CB(Function) ((Eina_Free_Cb)Function)

/**
 * @def EINA_C_ARRAY_LENGTH
 * Macro to return the array length of a standard c array.
 * For example:
 * int foo[] = { 0, 1, 2, 3 };
 * would return 4 and not 4 * sizeof(int).
 * @since 1.2.0
 */
#define EINA_C_ARRAY_LENGTH(arr) (sizeof(arr) / sizeof((arr)[0]))

/**
 * @}
 */

/**
 * @}
 */

#endif /* EINA_TYPES_H_ */
