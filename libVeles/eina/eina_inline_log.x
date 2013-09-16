/* EINA - EFL data type library
 * Copyright (C) 2002-2008 Gustavo Sverzut Barbieri
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

#ifndef EINA_LOG_INLINE_H_
#define EINA_LOG_INLINE_H_

/**
 * @addtogroup Eina_Log_Group Log
 *
 * @{
 */

/**
 * @brief Checks whenever the given level should be printed out.
 *
 * @param level The level to print
 *
 * This is useful to enable certain blocks of code just when given
 * level is to be used.
 *
 * @code
 * #include <Eina.h>
 *
 * void my_func(void *data)
 * {
 *    if (eina_log_level_check(EINA_LOG_LEVEL_WARN))
 *       expensive_debugging_code(data);
 *
 *    my_func_code(data);
 * }
 * @endcode
 *
 * @return #EINA_TRUE if level is equal or smaller than the current global
 *         logging level.
 */
static inline Eina_Bool
eina_log_level_check(int level)
{
   return eina_log_level_get() >= level;
}

/**
 * @brief Checks whenever the given level should be printed out.
 *
 * @param domain The domain to check
 * @param level The level to print
 *
 * This is useful to enable certain blocks of code just when given
 * level is to be used.
 *
 * @code
 * #include <Eina.h>
 *
 * extern int _my_log_dom;
 *
 * void my_func(void *data)
 * {
 *    if (eina_log_domain_level_check(_my_log_dom, EINA_LOG_LEVEL_WARN))
 *       expensive_debugging_code(data);
 *
 *    my_func_code(data);
 * }
 * @endcode
 *
 * @return #EINA_TRUE if level is equal or smaller than the current
 *         domain logging level.
 */
static inline Eina_Bool
eina_log_domain_level_check(int domain, int level)
{
   int dom_level = eina_log_domain_registered_level_get(domain);
   if (EINA_UNLIKELY(dom_level == EINA_LOG_LEVEL_UNKNOWN))
     return EINA_FALSE;
   return dom_level >= level;
}

/**
 * Function to format the level as a 3 character (+1 null byte) string.
 *
 * This function converts the given level to a known string name (CRI,
 * ERR, WRN, INF or DBG) or a zero-padded 3-character string. In any
 * case the last byte will contain a trailing null byte.
 *
 * If extreme level values are used (greater than 999 and smaller than
 * -99), then the value will just consider the less significant
 * part. This is so uncommon that users should handle this in their
 * code.
 *
 * @param level what level value to use.
 * @param name where to write the actual value.
 *
 * @return pointer to @p name.
 */
static inline const char *
eina_log_level_name_get(int level, char name[4])
{
#define BCPY(A, B, C) \
   do { name[0] = A; name[1] = B; name[2] = C; } while (0)

   if (EINA_UNLIKELY(level < 0))
     {
	name[0] = '-';
	name[1] = '0' + (-level / 10) % 10;
	name[2] = '0' + (-level % 10);
     }
   else if (EINA_UNLIKELY(level >= EINA_LOG_LEVELS))
     {
	name[0] = '0' + (level / 100) % 10;
	name[1] = '0' + (level / 10) % 10;
	name[2] = '0' + level % 10;
     }
   else if (level == 0)
     BCPY('C', 'R', 'I');
   else if (level == 1)
     BCPY('E', 'R', 'R');
   else if (level == 2)
     BCPY('W', 'R', 'N');
   else if (level == 3)
     BCPY('I', 'N', 'F');
   else if (level == 4)
     BCPY('D', 'B', 'G');
   else
     BCPY('?', '?', '?');

   name[3] = '\0';

   return name;
}

/**
 * Function to get recommended color value for level.
 *
 * This function will not check if colors are enabled or not before
 * returning the level color. If you desire such check, use
 * eina_log_level_color_if_enabled_get().
 *
 * @param level what level value to use.
 *
 * @return pointer to null byte terminated ANSI color string to be
 *         used in virtual terminals supporting VT100 color codes.
 *
 * @see eina_log_level_color_if_enabled_get()
 */
static inline const char *
eina_log_level_color_get(int level)
{
   if (level <= 0)
     return EINA_COLOR_LIGHTRED;
   else if (level == 1)
     return EINA_COLOR_RED;
   else if (level == 2)
     return EINA_COLOR_YELLOW;
   else if (level == 3)
     return EINA_COLOR_GREEN;
   else if (level == 4)
     return EINA_COLOR_LIGHTBLUE;
   else
     return EINA_COLOR_BLUE;
}

/**
 * Function to get recommended color value for level, if colors are
 * enabled.
 *
 * This function will check if colors are enabled or not before
 * returning the level color. If colors are disabled, then empty
 * string is returned.
 *
 * @param level what level value to use.
 *
 * @return pointer to null byte terminated ANSI color string to be
 *         used in virtual terminals supporting VT100 color codes. If
 *         colors are disabled, the empty string is returned.
 */
static inline const char *
eina_log_level_color_if_enabled_get(int level)
{
   if (eina_log_color_disable_get())
     return "";
   return eina_log_level_color_get(level);
}

/**
 * @}
 */

#endif /* EINA_LOG_INLINE_H_ */
