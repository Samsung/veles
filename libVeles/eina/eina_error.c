/* EINA - EFL data type library
 * Copyright (C) 2007-2008 Jorge Luis Zapata Muga, Cedric Bail
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

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef HAVE_EVIL
# include <Evil.h>
#endif

#include "eina_config.h"
#include "eina_private.h"


/* undefs EINA_ARG_NONULL() so NULL checks are not compiled out! */
#include "eina_safety_checks.h"
#include "eina_error.h"

/* TODO
 * + add a wrapper for assert?
 * + add common error numbers, messages
 * + add a calltrace of errors, not only store the last error but a list of them
 * and also store the function that set it
 */

/*============================================================================*
*                                  Local                                     *
*============================================================================*/

/**
 * @cond LOCAL
 */

typedef struct _Eina_Error_Message Eina_Error_Message;
struct _Eina_Error_Message
{
   Eina_Bool string_allocated;
   const char *string;
};

static Eina_Error_Message *_eina_errors = NULL;
static size_t _eina_errors_count = 0;
static size_t _eina_errors_allocated = 0;
static Eina_Error _eina_last_error;

static Eina_Error_Message *
_eina_error_msg_alloc(void)
{
   size_t idx;

   if (_eina_errors_count == _eina_errors_allocated)
     {
        void *tmp;
        size_t size;

        if (EINA_UNLIKELY(_eina_errors_allocated == 0))
           size = 24;
        else
           size = _eina_errors_allocated + 8;

        tmp = realloc(_eina_errors, sizeof(Eina_Error_Message) * size);
        if (!tmp)
           return NULL;

        _eina_errors = tmp;
        _eina_errors_allocated = size;
     }

   idx = _eina_errors_count;
   _eina_errors_count++;
   return _eina_errors + idx;
}

/**
 * @endcond
 */


/*============================================================================*
*                                 Global                                     *
*============================================================================*/

/**
 * @cond LOCAL
 */

EAPI Eina_Error EINA_ERROR_OUT_OF_MEMORY = 0;

static const char EINA_ERROR_OUT_OF_MEMORY_STR[] = "Out of memory";

/**
 * @endcond
 */

/**
 * @internal
 * @brief Initialize the error module.
 *
 * @return #EINA_TRUE on success, #EINA_FALSE on failure.
 *
 * This function sets up the error module of Eina. It is called by
 * eina_init().
 *
 * This function registers the error #EINA_ERROR_OUT_OF_MEMORY.
 *
 * @see eina_init()
 */
Eina_Bool
eina_error_init(void)
{
   /* TODO register the eina's basic errors */
   EINA_ERROR_OUT_OF_MEMORY = eina_error_msg_static_register(
         EINA_ERROR_OUT_OF_MEMORY_STR);
   return EINA_TRUE;
}

/**
 * @internal
 * @brief Shut down the error module.
 *
 * @return #EINA_TRUE on success, #EINA_FALSE on failure.
 *
 * This function shuts down the error module set up by
 * eina_error_init(). It is called by eina_shutdown().
 *
 * @see eina_shutdown()
 */
Eina_Bool
eina_error_shutdown(void)
{
   Eina_Error_Message *eem, *eem_end;

   eem = _eina_errors;
   eem_end = eem + _eina_errors_count;

   for (; eem < eem_end; eem++) {
      if (eem->string_allocated) {
         free((char *)eem->string);
         eem->string_allocated = EINA_FALSE;
      }
   }

   free(_eina_errors);
   _eina_errors = NULL;
   _eina_errors_count = 0;
   _eina_errors_allocated = 0;

   return EINA_TRUE;
}

/*============================================================================*
*                                   API                                      *
*============================================================================*/

EAPI Eina_Error
eina_error_msg_register(const char *msg)
{
   Eina_Error_Message *eem;

   EINA_SAFETY_ON_NULL_RETURN_VAL(msg, 0);

   eem = _eina_error_msg_alloc();
   if (!eem)
      return 0;

   eem->string_allocated = EINA_TRUE;
   eem->string = msg;
   if (!eem->string)
     {
        _eina_errors_count--;
        return 0;
     }

   return _eina_errors_count; /* identifier = index + 1 (== _count). */
}

EAPI Eina_Error
eina_error_msg_static_register(const char *msg)
{
   Eina_Error_Message *eem;

   EINA_SAFETY_ON_NULL_RETURN_VAL(msg, 0);

   eem = _eina_error_msg_alloc();
   if (!eem)
      return 0;

   eem->string_allocated = EINA_FALSE;
   eem->string = msg;
   return _eina_errors_count; /* identifier = index + 1 (== _count). */
}

EAPI Eina_Bool
eina_error_msg_modify(Eina_Error error, const char *msg)
{
   Eina_Error_Message *last = _eina_errors + error - 1;

   EINA_SAFETY_ON_NULL_RETURN_VAL(msg, EINA_FALSE);
   if (error < 1)
      return EINA_FALSE;

   if ((size_t)error > _eina_errors_count)
      return EINA_FALSE;

   if (last->string_allocated)
     {
        free((char *)last->string);
        last->string = msg;
        last->string_allocated = EINA_FALSE;
        return EINA_TRUE;
     }

   _eina_errors[error - 1].string = msg;
   return EINA_TRUE;
}

EAPI const char *
eina_error_msg_get(Eina_Error error)
{
   if (error < 1)
      return NULL;

   if ((size_t)error > _eina_errors_count)
      return NULL;

   return _eina_errors[error - 1].string;
}

EAPI Eina_Error
eina_error_get(void)
{
   return _eina_last_error;
}

EAPI void
eina_error_set(Eina_Error err)
{
   _eina_last_error = err;
}

EAPI Eina_Error
eina_error_find(const char *msg)
{
   size_t i;

   EINA_SAFETY_ON_NULL_RETURN_VAL(msg, 0);

   for (i = 0; i < _eina_errors_count; i++)
     {
        if (_eina_errors[i].string_allocated)
          {
             if (_eina_errors[i].string == msg)
               return i + 1;
          }
        if (!strcmp(_eina_errors[i].string, msg))
          return i + 1;
     }
   return 0;
}
