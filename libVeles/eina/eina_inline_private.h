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
#define EINA_INLINE_PRIVATE_H_

#include <time.h>
#include <sys/time.h>

typedef struct timespec Eina_Nano_Time;

static inline int
_eina_time_get(Eina_Nano_Time *tp)
{
   struct timeval tv;

   if (gettimeofday(&tv, NULL))
     return -1;

   tp->tv_sec = tv.tv_sec;
   tp->tv_nsec = tv.tv_usec * 1000L;

   return 0;
}

static inline long int
_eina_time_convert(Eina_Nano_Time *tp)
{
  long int r = tp->tv_sec * 1000000000 + tp->tv_nsec;
  return r;
}

static inline long int
_eina_time_delta(Eina_Nano_Time *start, Eina_Nano_Time *end)
{
  long int r = (end->tv_sec - start->tv_sec) * 1000000000 +
    end->tv_nsec - start->tv_nsec;

  return r;
}

#endif
