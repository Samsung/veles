/* EINA - EFL data type library
 * Copyright (C) 2012 Cedric Bail
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

#ifndef EINA_THREAD_H_
#define EINA_THREAD_H_

#include "eina_config.h"
#include "eina_types.h"
#include "eina_error.h"

/**
 * @addtogroup Eina_Tools_Group Tools
 *
 * @{
 */

/**
 * @defgroup Eina_Thread_Group Thread
 *
 * Abstracts platform threads, providing an uniform API. It's modeled
 * after POSIX THREADS (pthreads), on Linux they are almost 1:1
 * mapping.
 *
 * @see @ref Eina_Lock_Group for mutex/locking abstraction.
 *
 * @since 1.8
 * @{
 */

/**
 * @var EINA_ERROR_THREAD_CREATION_FAILED
 * Generic error happened and thread couldn't be created.
 * @since 1.8
 */
EAPI extern Eina_Error EINA_ERROR_THREAD_CREATION_FAILED;

/**
 * @var EINA_ERROR_THREAD_CREATION_FAILED_RESOURCES
 * System lacked resources to create thread.
 * @since 1.8
 */
EAPI extern Eina_Error EINA_ERROR_THREAD_CREATION_FAILED_RESOURCES;

/**
 * @var EINA_ERROR_THREAD_CREATION_FAILED_PERMISSIONS
 * System lacked permissions to create thread.
 * @since 1.8
 */
EAPI extern Eina_Error EINA_ERROR_THREAD_CREATION_FAILED_PERMISSIONS;

/**
 * @var EINA_ERROR_THREAD_JOIN_DEADLOCK
 * The system has detected a deadlock situation where both threads
 * would wait each other. Or the thread wanted to wait for itself.
 * @since 1.8
 */
EAPI extern Eina_Error EINA_ERROR_THREAD_JOIN_DEADLOCK;

/**
 * @var EINA_ERROR_THREAD_JOIN_INVALID
 * One of the following happened:
 * @li thread is not a joinable.
 * @li thread does not exist.
 * @li another thread is already waiting for that thread.
 * @since 1.8
 */
EAPI extern Eina_Error EINA_ERROR_THREAD_JOIN_INVALID;


typedef unsigned long int Eina_Thread;

typedef void *(*Eina_Thread_Cb)(void *data, Eina_Thread t);

typedef enum _Eina_Thread_Priority
{
  EINA_THREAD_URGENT,
  EINA_THREAD_NORMAL,
  EINA_THREAD_BACKGROUND,
  EINA_THREAD_IDLE
} Eina_Thread_Priority;

/**
 * Return identifier of the current thread.
 * @return identifier of current thread.
 * @since 1.8
 */
EAPI Eina_Thread eina_thread_self(void) EINA_WARN_UNUSED_RESULT;

/**
 * Check if two thread identifiers are the same.
 * @param t1 first thread identifier to compare.
 * @param t2 second thread identifier to compare.
 * @return #EINA_TRUE if they are equal, #EINA_FALSE otherwise.
 * @since 1.8
 */
EAPI Eina_Bool eina_thread_equal(Eina_Thread t1, Eina_Thread t2) EINA_WARN_UNUSED_RESULT;

/**
 * Create a new thread, setting its priority and affinity.
 *
 * @param t[out] where to return the thread identifier. Must @b not be @c NULL.
 * @param prio thread priority to use, usually #EINA_THREAD_BACKGROUND
 * @param affinity thread affinity to use. To not set affinity use @c -1.
 * @param func function to run in the thread. Must @b not be @c NULL.
 * @param data context data to provide to @a func as first argument.
 * @return #EINA_TRUE if thread was created, #EINA_FALSE on errors.
 * @since 1.8
 */
EAPI Eina_Bool eina_thread_create(Eina_Thread *t,
                                  Eina_Thread_Priority prio, int affinity,
                                  Eina_Thread_Cb func, const void *data) EINA_ARG_NONNULL(1, 4) EINA_WARN_UNUSED_RESULT;

/**
 * Join a currently running thread, waiting until it finishes.
 *
 * This function will block the current thread until @a t
 * finishes. The returned value is the one returned by @a t @c func()
 * and may be @c NULL on errors. See @ref Eina_Error_Group to identify
 * problems.
 *
 * @param t thread identifier to wait.
 * @return value returned by @a t creation function @c func() or
 *         @c NULL on errors. Check error with @ref Eina_Error_Group.
 * @since 1.8
 */
EAPI void *eina_thread_join(Eina_Thread t);

/**
 * @}
 */

/**
 * @}
 */

#endif
