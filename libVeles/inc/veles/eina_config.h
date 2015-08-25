/* EINA - EFL data type library
 * Copyright (C) 2008 Cedric Bail
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

#ifndef EINA_CONFIG_H_
#define EINA_CONFIG_H_

#ifdef HAVE_EXOTIC
# include <Exotic.h>
#endif

#ifdef EINA_MAGIC_DEBUG
# undef EINA_MAGIC_DEBUG
#endif
#define EINA_MAGIC_DEBUG

#ifndef EINA_DEFAULT_MEMPOOL
# undef EINA_DEFAULT_MEMPOOL
#endif


#ifndef EINA_SAFETY_CHECKS
# define EINA_SAFETY_CHECKS
#endif

#ifndef EINA_HAVE_THREADS
#define EINA_HAVE_THREADS
#endif

#ifdef EINA_HAVE_PTHREAD_BARRIER
# undef EINA_HAVE_PTHREAD_BARRIER
#endif
#define EINA_HAVE_PTHREAD_BARRIER

#ifdef EINA_HAVE_DEBUG_THREADS
# undef EINA_HAVE_DEBUG_THREADS
#endif

#ifdef EFL_HAVE_POSIX_THREADS
#undef EFL_HAVE_POSIX_THREADS
#endif
#define EFL_HAVE_POSIX_THREADS

#ifdef EFL_HAVE_THREADS
#undef EFL_HAVE_THREADS
#endif
#define EFL_HAVE_THREADS

#ifdef EFL_HAVE_POSIX_THREADS_SPINLOCK
#undef EFL_HAVE_POSIX_THREADS_SPINLOCK
#endif
#define EFL_HAVE_POSIX_THREADS_SPINLOCK

#ifdef EINA_SIZEOF_WCHAR_T
# undef EINA_SIZEOF_WCHAR_T
#endif
#define EINA_SIZEOF_WCHAR_T 4

#ifdef EINA_CONFIGURE_HAVE_DIRENT_H
# undef EINA_CONFIGURE_HAVE_DIRENT_H
#endif


#ifdef EINA_CONFIGURE_ENABLE_LOG
# undef EINA_CONFIGURE_ENABLE_LOG
#endif
#define EINA_ENABLE_LOG

#ifdef EINA_HAVE_ALLOCA_H
# undef EINA_HAVE_ALLOCA_H
#endif
#define EINA_HAVE_ALLOCA_H

#define HAVE_EXECINFO_H
#define HAVE_BACKTRACE
#define HAVE_BACKTRACE_SYMBOLS

#endif /* EINA_CONFIG_H_ */
