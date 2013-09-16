/* EINA - EFL data type library
 * Copyright (C) 2013  Cedric Bail
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

#ifndef EINA_INLINE_PRIVATE_H_
# define EINA_INLINE_PRIVATE_H_


#ifndef _WIN32
#ifndef __USE_POSIX199309
#define __USE_POSIX199309
#endif
# include <time.h>
# include <sys/time.h>
#else
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
# undef WIN32_LEAN_AND_MEAN
#endif /* _WIN2 */

#ifndef _WIN32
typedef struct timespec Eina_Nano_Time;
#else
typedef LARGE_INTEGER Eina_Nano_Time;

extern LARGE_INTEGER _eina_counter_frequency;
#endif

static inline int
_eina_time_get(Eina_Nano_Time *tp)
{
#ifndef _WIN32
# if defined(CLOCK_PROCESS_CPUTIME_ID)
   return clock_gettime(CLOCK_PROCESS_CPUTIME_ID, tp);
# elif defined(CLOCK_PROF)
   return clock_gettime(CLOCK_PROF, tp);
# elif defined(CLOCK_REALTIME)
   return clock_gettime(CLOCK_REALTIME, tp);
# else
   struct timeval tv;

   if (gettimeofday(&tv, NULL))
     return -1;

   tp->tv_sec = tv.tv_sec;
   tp->tv_nsec = tv.tv_usec * 1000L;

   return 0;
# endif
#else
   return QueryPerformanceCounter(tp);
#endif /* _WIN2 */
}

static inline long int
_eina_time_convert(Eina_Nano_Time *tp)
{
  long int r;

#ifndef _WIN32
  r = tp->tv_sec * 1000000000 + tp->tv_nsec;
#else
  r = (long int)(((long long int)tp->QuadPart * 1000000000ll) /
		 (long long int)_eina_counter_frequency.QuadPart);
#endif

  return r;
}

static inline long int
_eina_time_delta(Eina_Nano_Time *start, Eina_Nano_Time *end)
{
  long int r;

#ifndef _WIN32
  r = (end->tv_sec - start->tv_sec) * 1000000000 +
    end->tv_nsec - start->tv_nsec;
#else
  r = (long int)(((long long int)(end->QuadPart - start->QuadPart)
		  * 1000000000LL)
		 / (long long int)_eina_counter_frequency.QuadPart);
#endif

  return r;
}

#endif
