/* EINA - EFL data type library
 * Copyright (C) 2007-2009 Jorge Luis Zapata Muga, Cedric Bail, Andre Dieb
 * Martins
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
#include <unistd.h>
#include <fnmatch.h>
#include <assert.h>
#include <errno.h>
#include <pthread.h>

#include "eina_config.h"
#if defined HAVE_EXECINFO_H && defined HAVE_BACKTRACE && defined HAVE_BACKTRACE_SYMBOLS
# include <execinfo.h>
# define EINA_LOG_BACKTRACE
#endif

#ifdef HAVE_SYSTEMD
# include <systemd/sd-journal.h>
#endif

#include "eina_private.h"
#include "eina_inlist.h"
#include "eina_thread.h"
#include "eina_safety_checks.h"
#include "eina_log.h"
#include "eina_inline_private.h"

/* TODO
 * + printing logs to stdout or stderr can be implemented
 * using a queue, useful for multiple threads printing
 * + add a wrapper for assert?
 */

/*============================================================================*
*                                  Local                                     *
*============================================================================*/

/**
 * @cond LOCAL
 */

#define EINA_LOG_ENV_ABORT "EINA_LOG_ABORT"
#define EINA_LOG_ENV_ABORT_LEVEL "EINA_LOG_ABORT_LEVEL"
#define EINA_LOG_ENV_LEVEL "EINA_LOG_LEVEL"
#define EINA_LOG_ENV_LEVELS "EINA_LOG_LEVELS"
#define EINA_LOG_ENV_LEVELS_GLOB "EINA_LOG_LEVELS_GLOB"
#define EINA_LOG_ENV_COLOR_DISABLE "EINA_LOG_COLOR_DISABLE"
#define EINA_LOG_ENV_FILE_DISABLE "EINA_LOG_FILE_DISABLE"
#define EINA_LOG_ENV_FUNCTION_DISABLE "EINA_LOG_FUNCTION_DISABLE"
#define EINA_LOG_ENV_BACKTRACE "EINA_LOG_BACKTRACE"

#ifdef EINA_ENABLE_LOG

// Structure for storing domain level settings passed from the command line
// that will be matched with application-defined domains.
typedef struct _Eina_Log_Domain_Level_Pending Eina_Log_Domain_Level_Pending;
struct _Eina_Log_Domain_Level_Pending
{
   EINA_INLIST;
   unsigned int level;
   size_t namelen;
   char name[];
};

typedef struct _Eina_Log_Timing Eina_Log_Timing;
struct _Eina_Log_Timing
{
   const char *phase;
   Eina_Nano_Time start;
   Eina_Log_State state;
};

EAPI const char *_eina_log_state_init = "init";
EAPI const char *_eina_log_state_shutdown = "shutdown";

/*
 * List of levels for domains set by the user before the domains are registered,
 * updates the domain levels on the first log and clears itself.
 */
static Eina_Inlist *_pending_list = NULL;
static Eina_Inlist *_glob_list = NULL;

// Disable color flag (can be changed through the env var
// EINA_LOG_ENV_COLOR_DISABLE).
static Eina_Bool _disable_color = EINA_FALSE;
static Eina_Bool _disable_file = EINA_FALSE;
static Eina_Bool _disable_function = EINA_FALSE;
static Eina_Bool _abort_on_critical = EINA_FALSE;
static Eina_Bool _disable_timing = EINA_TRUE;
static int _abort_level_on_critical = EINA_LOG_LEVEL_CRITICAL;

#ifdef EINA_LOG_BACKTRACE
static int _backtrace_level = -1;
#endif

EAPI Eina_Thread
eina_thread_self(void)
{
   return pthread_self();
}

EAPI Eina_Bool
eina_thread_equal(Eina_Thread t1, Eina_Thread t2)
{
   return !!pthread_equal(t1, t2);
}

static Eina_Bool _threads_enabled = EINA_FALSE;
static Eina_Bool _threads_inited = EINA_FALSE;

static Eina_Thread _main_thread;

#  define SELF() eina_thread_self()
#  define IS_MAIN(t)  eina_thread_equal(t, _main_thread)
#  define IS_OTHER(t) EINA_UNLIKELY(!IS_MAIN(t))
#  define CHECK_MAIN(...)                                         \
   do {                                                           \
      if (!IS_MAIN(eina_thread_self())) {                         \
         fprintf(stderr,                                          \
                 "ERR: not main thread! current=%lu, main=%lu\n", \
                 (unsigned long)eina_thread_self(),               \
                 (unsigned long)_main_thread);                    \
         return __VA_ARGS__;                                      \
      }                                                           \
   } while (0)

#ifdef EFL_HAVE_POSIX_THREADS_SPINLOCK

static pthread_spinlock_t _log_lock;

static Eina_Bool _eina_log_spinlock_init(void)
{
   if (pthread_spin_init(&_log_lock, PTHREAD_PROCESS_PRIVATE) == 0)
     return EINA_TRUE;

   fprintf(stderr,
           "ERROR: pthread_spin_init(%p, PTHREAD_PROCESS_PRIVATE): %s\n",
           &_log_lock, strerror(errno));
   return EINA_FALSE;
}

#   define LOG_LOCK()                                                  \
   if (_threads_enabled)                                               \
         do {                                                          \
            if (0) {                                                   \
               fprintf(stderr, "+++LOG LOG_LOCKED!   [%s, %lu]\n",     \
                       __FUNCTION__, (unsigned long)eina_thread_self()); } \
            if (EINA_UNLIKELY(_threads_enabled)) {                     \
               pthread_spin_lock(&_log_lock); }                        \
         } while (0)
#   define LOG_UNLOCK()                                                \
   if (_threads_enabled)                                               \
         do {                                                          \
            if (EINA_UNLIKELY(_threads_enabled)) {                     \
               pthread_spin_unlock(&_log_lock); }                      \
            if (0) {                                                   \
               fprintf(stderr,                                         \
                       "---LOG LOG_UNLOCKED! [%s, %lu]\n",             \
                       __FUNCTION__, (unsigned long)eina_thread_self()); } \
         } while (0)
#   define INIT() _eina_log_spinlock_init()
#   define SHUTDOWN() pthread_spin_destroy(&_log_lock)

#else /* ! EFL_HAVE_POSIX_THREADS_SPINLOCK */

static Eina_Lock _log_mutex;
#   define LOG_LOCK() if(_threads_enabled) {eina_lock_take(&_log_mutex); }
#   define LOG_UNLOCK() if(_threads_enabled) {eina_lock_release(&_log_mutex); }
#   define INIT() eina_lock_new(&_log_mutex)
#   define SHUTDOWN() eina_lock_free(&_log_mutex)

#endif /* ! EFL_HAVE_POSIX_THREADS_SPINLOCK */


// List of domains registered
static Eina_Log_Domain *_log_domains = NULL;
static Eina_Log_Timing *_log_timing = NULL;
static unsigned int _log_domains_count = 0;
static size_t _log_domains_allocated = 0;

// Default function for printing on domains
static Eina_Log_Print_Cb _print_cb = eina_log_print_cb_stderr;
static void *_print_cb_data = NULL;

#ifdef DEBUG
static Eina_Log_Level _log_level = EINA_LOG_LEVEL_DBG;
#elif DEBUG_CRITICAL
static Eina_Log_Level _log_level = EINA_LOG_LEVEL_CRITICAL;
#else
static Eina_Log_Level _log_level = EINA_LOG_LEVEL_ERR;
#endif

/* NOTE: if you change this, also change:
 *   eina_log_print_level_name_get()
 *   eina_log_print_level_name_color_get()
 */
static const char *_names[] = {
   "CRI",
   "ERR",
   "WRN",
   "INF",
   "DBG",
};

#ifdef _WIN32
/* TODO: query win32_def_attr on eina_log_init() */
static int win32_def_attr = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE;

/* NOTE: can't use eina_log from inside this function */
static int
eina_log_win32_color_convert(const char *color, const char **endptr)
{
   const char *p;
   int attr = 0;

   if (endptr) *endptr = color;

   if (color[0] != '\033') return 0;
   if (color[1] != '[') return 0;

   p = color + 2;
   while (1)
     {
        char *end;
        int code = strtol(p, &end, 10);

        if (p == end)
          {
             //fputs("empty color string\n", stderr);
             if (endptr) *endptr = end;
             attr = 0; /* assume it was not color, must end with 'm' */
             break;
          }

        if (code)
          {
             if (code == 0) attr = win32_def_attr;
             else if (code == 1) attr |= FOREGROUND_INTENSITY;
             else if (code == 4) attr |= COMMON_LVB_UNDERSCORE;
             else if (code == 7) attr |= COMMON_LVB_REVERSE_VIDEO;
             else if ((code >= 30) && (code <= 37))
               {
                  /* clear foreground */
                  attr &= ~(FOREGROUND_RED |
                            FOREGROUND_GREEN |
                            FOREGROUND_BLUE);

                  if (code == 31)
                    attr |= FOREGROUND_RED;
                  else if (code == 32)
                    attr |= FOREGROUND_GREEN;
                  else if (code == 33)
                    attr |= FOREGROUND_RED | FOREGROUND_GREEN;
                  else if (code == 34)
                    attr |= FOREGROUND_BLUE;
                  else if (code == 35)
                    attr |= FOREGROUND_RED | FOREGROUND_BLUE;
                  else if (code == 36)
                    attr |= FOREGROUND_GREEN | FOREGROUND_BLUE;
                  else if (code == 37)
                    attr |= FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE;
               }
             else if ((code >= 40) && (code <= 47))
               {
                  /* clear background */
                  attr &= ~(BACKGROUND_RED |
                            BACKGROUND_GREEN |
                            BACKGROUND_BLUE);

                  if (code == 41)
                    attr |= BACKGROUND_RED;
                  else if (code == 42)
                    attr |= BACKGROUND_GREEN;
                  else if (code == 43)
                    attr |= BACKGROUND_RED | BACKGROUND_GREEN;
                  else if (code == 44)
                    attr |= BACKGROUND_BLUE;
                  else if (code == 45)
                    attr |= BACKGROUND_RED | BACKGROUND_BLUE;
                  else if (code == 46)
                    attr |= BACKGROUND_GREEN | BACKGROUND_BLUE;
                  else if (code == 47)
                    attr |= BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE;
               }
          }

        if (*end == 'm')
          {
             if (endptr) *endptr = end + 1;
             break;
          }
        else if (*end == ';')
          p = end + 1;
        else
          {
             //fprintf(stderr, "unexpected char in color string: %s\n", end);
             attr = 0; /* assume it was not color */
             if (endptr) *endptr = end;
             break;
          }
     }

   return attr;
}

static int
eina_log_win32_color_get(const char *color)
{
   return eina_log_win32_color_convert(color, NULL);
}
#endif

static inline unsigned int
eina_log_pid_get(void)
{
   return (unsigned int)getpid();
}

static inline void
eina_log_print_level_name_get(int level, const char **p_name)
{
   static char buf[4];
   /* NOTE: if you change this, also change
    *    eina_log_print_level_name_color_get()
    *    eina_log_level_name_get() (at eina_inline_log.x)
    */
   if (EINA_UNLIKELY(level < 0))
     {
        snprintf(buf, sizeof(buf), "%03d", level);
        *p_name = buf;
     }
   else if (EINA_UNLIKELY(level >= EINA_LOG_LEVELS))
     {
        snprintf(buf, sizeof(buf), "%03d", level);
        *p_name = buf;
     }
   else
      *p_name = _names[level];
}

#ifdef _WIN32
static inline void
eina_log_print_level_name_color_get(int level,
                                    const char **p_name,
                                    int *p_color)
{
   static char buf[4];
   /* NOTE: if you change this, also change:
    *   eina_log_print_level_name_get()
    */
   if (EINA_UNLIKELY(level < 0))
     {
        snprintf(buf, sizeof(buf), "%03d", level);
        *p_name = buf;
     }
   else if (EINA_UNLIKELY(level >= EINA_LOG_LEVELS))
     {
        snprintf(buf, sizeof(buf), "%03d", level);
        *p_name = buf;
     }
   else
      *p_name = _names[level];

   *p_color = eina_log_win32_color_get(eina_log_level_color_get(level));
}
#else
static inline void
eina_log_print_level_name_color_get(int level,
                                    const char **p_name,
                                    const char **p_color)
{
   static char buf[4];
   /* NOTE: if you change this, also change:
    *   eina_log_print_level_name_get()
    */
   if (EINA_UNLIKELY(level < 0))
     {
        snprintf(buf, sizeof(buf), "%03d", level);
        *p_name = buf;
     }
   else if (EINA_UNLIKELY(level >= EINA_LOG_LEVELS))
     {
        snprintf(buf, sizeof(buf), "%03d", level);
        *p_name = buf;
     }
   else
      *p_name = _names[level];

   *p_color = eina_log_level_color_get(level);
}
#endif

#define DECLARE_LEVEL_NAME(level) const char *name; \
   eina_log_print_level_name_get(level, &name)
#ifdef _WIN32
# define DECLARE_LEVEL_NAME_COLOR(level) const char *name; int color; \
   eina_log_print_level_name_color_get(level, &name, &color)
#else
# define DECLARE_LEVEL_NAME_COLOR(level) const char *name, *color; \
   eina_log_print_level_name_color_get(level, &name, &color)
#endif

/** No threads, No color */
static void
eina_log_print_prefix_NOthreads_NOcolor_file_func(FILE *fp,
                                                  const Eina_Log_Domain *d,
                                                  Eina_Log_Level level,
                                                  const char *file,
                                                  const char *fnc,
                                                  int line)
{
   DECLARE_LEVEL_NAME(level);
   fprintf(fp, "%s<%u>:%s %s:%d %s() ", name, eina_log_pid_get(), 
           d->domain_str, file, line, fnc);
}

static void
eina_log_print_prefix_NOthreads_NOcolor_NOfile_func(FILE *fp,
                                                    const Eina_Log_Domain *d,
                                                    Eina_Log_Level level,
                                                    const char *file EINA_UNUSED,
                                                    const char *fnc,
                                                    int line EINA_UNUSED)
{
   DECLARE_LEVEL_NAME(level);
   fprintf(fp, "%s<%u>:%s %s() ", name, eina_log_pid_get(), d->domain_str, 
           fnc);
}

static void
eina_log_print_prefix_NOthreads_NOcolor_file_NOfunc(FILE *fp,
                                                    const Eina_Log_Domain *d,
                                                    Eina_Log_Level level,
                                                    const char *file,
                                                    const char *fnc EINA_UNUSED,
                                                    int line)
{
   DECLARE_LEVEL_NAME(level);
   fprintf(fp, "%s<%u>:%s %s:%d ", name, eina_log_pid_get(), d->domain_str, 
           file, line);
}

/* No threads, color */
static void
eina_log_print_prefix_NOthreads_color_file_func(FILE *fp,
                                                const Eina_Log_Domain *d,
                                                Eina_Log_Level level,
                                                const char *file,
                                                const char *fnc,
                                                int line)
{
   DECLARE_LEVEL_NAME_COLOR(level);
#ifdef _WIN32_WCE
   fprintf(fp, "%s<%u>:%s %s:%d %s() ", name, eina_log_pid_get(), 
           d->domain_str, file, line, fnc);
#elif _WIN32
   SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                           color);
   fprintf(fp, "%s", name);
   SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                           FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
   fprintf(fp, ":");
   SetConsoleTextAttribute(GetStdHandle(
                              STD_OUTPUT_HANDLE),
                           eina_log_win32_color_get(d->domain_str));
   fprintf(fp, "%s", d->name);
   SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                           FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
   fprintf(fp, " %s:%d ", file, line);
   SetConsoleTextAttribute(GetStdHandle(
                              STD_OUTPUT_HANDLE),
                           FOREGROUND_INTENSITY | FOREGROUND_RED |
                           FOREGROUND_GREEN | FOREGROUND_BLUE);
   fprintf(fp, "%s()", fnc);
   SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                           FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
   fprintf(fp, " ");
#else
   fprintf(fp, "%s%s<%u>" EINA_COLOR_RESET ":%s %s:%d "
           EINA_COLOR_HIGH "%s()" EINA_COLOR_RESET " ",
           color, name, eina_log_pid_get(), d->domain_str, file, line, fnc);
#endif
}

static void
eina_log_print_prefix_NOthreads_color_NOfile_func(FILE *fp,
                                                  const Eina_Log_Domain *d,
                                                  Eina_Log_Level level,
                                                  const char *file EINA_UNUSED,
                                                  const char *fnc,
                                                  int line EINA_UNUSED)
{
   DECLARE_LEVEL_NAME_COLOR(level);
#ifdef _WIN32_WCE
   fprintf(fp, "%s<%u>:%s %s() ", name, eina_log_pid_get(), d->domain_str, 
           fnc);
#elif _WIN32
   SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                           color);
   fprintf(fp, "%s", name);
   SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                           FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
   fprintf(fp, ":");
   SetConsoleTextAttribute(GetStdHandle(
                              STD_OUTPUT_HANDLE),
                           eina_log_win32_color_get(d->domain_str));
   fprintf(fp, "%s", d->name);
   SetConsoleTextAttribute(GetStdHandle(
                              STD_OUTPUT_HANDLE),
                           FOREGROUND_INTENSITY | FOREGROUND_RED |
                           FOREGROUND_GREEN | FOREGROUND_BLUE);
   fprintf(fp, "%s()", fnc);
   SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                           FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
   fprintf(fp, " ");
#else
   fprintf(fp, "%s%s<%u>" EINA_COLOR_RESET ":%s "
           EINA_COLOR_HIGH "%s()" EINA_COLOR_RESET " ",
           color, name, eina_log_pid_get(), d->domain_str, fnc);
#endif
}

static void
eina_log_print_prefix_NOthreads_color_file_NOfunc(FILE *fp,
                                                  const Eina_Log_Domain *d,
                                                  Eina_Log_Level level,
                                                  const char *file,
                                                  const char *fnc EINA_UNUSED,
                                                  int line)
{
   DECLARE_LEVEL_NAME_COLOR(level);
#ifdef _WIN32_WCE
   fprintf(fp, "%s<%u>:%s %s:%d ", name, eina_log_pid_get(), d->domain_str, 
           file, line);
#elif _WIN32
   SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                           color);
   fprintf(fp, "%s", name);
   SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                           FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
   fprintf(fp, ":");
   SetConsoleTextAttribute(GetStdHandle(
                              STD_OUTPUT_HANDLE),
                           eina_log_win32_color_get(d->domain_str));
   fprintf(fp, "%s", d->name);
   SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                           FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
   fprintf(fp, " %s:%d ", file, line);
#else
   fprintf(fp, "%s%s<%u>" EINA_COLOR_RESET ":%s %s:%d ",
           color, name, eina_log_pid_get(), d->domain_str, file, line);
#endif
}

/** threads, No color */
static void
eina_log_print_prefix_threads_NOcolor_file_func(FILE *fp,
                                                const Eina_Log_Domain *d,
                                                Eina_Log_Level level,
                                                const char *file,
                                                const char *fnc,
                                                int line)
{
   Eina_Thread cur;

   DECLARE_LEVEL_NAME(level);
   cur = SELF();
   if (IS_OTHER(cur))
     {
        fprintf(fp, "%s<%u>:%s[T:%lu] %s:%d %s() ",
                name, eina_log_pid_get(), d->domain_str, 
                (unsigned long)cur, file, line, fnc);
        return;
     }
   fprintf(fp, "%s<%u>:%s %s:%d %s() ", 
           name, eina_log_pid_get(), d->domain_str, file, line, fnc);
}

static void
eina_log_print_prefix_threads_NOcolor_NOfile_func(FILE *fp,
                                                  const Eina_Log_Domain *d,
                                                  Eina_Log_Level level,
                                                  const char *file EINA_UNUSED,
                                                  const char *fnc,
                                                  int line EINA_UNUSED)
{
   Eina_Thread cur;

   DECLARE_LEVEL_NAME(level);
   cur = SELF();
   if (IS_OTHER(cur))
     {
        fprintf(fp, "%s<%u>:%s[T:%lu] %s() ",
                name, eina_log_pid_get(), d->domain_str, 
                (unsigned long)cur, fnc);
        return;
     }
   fprintf(fp, "%s<%u>:%s %s() ", 
           name, eina_log_pid_get(), d->domain_str, fnc);
}

static void
eina_log_print_prefix_threads_NOcolor_file_NOfunc(FILE *fp,
                                                  const Eina_Log_Domain *d,
                                                  Eina_Log_Level level,
                                                  const char *file,
                                                  const char *fnc EINA_UNUSED,
                                                  int line)
{
   Eina_Thread cur;

   DECLARE_LEVEL_NAME(level);
   cur = SELF();
   if (IS_OTHER(cur))
     {
        fprintf(fp, "%s<%u>:%s[T:%lu] %s:%d ",
                name, eina_log_pid_get(), d->domain_str, (unsigned long)cur, 
                file, line);
        return;
     }
   
   fprintf(fp, "%s<%u>:%s %s:%d ", 
           name, eina_log_pid_get(), d->domain_str, file, line);
}

/* threads, color */
static void
eina_log_print_prefix_threads_color_file_func(FILE *fp,
                                              const Eina_Log_Domain *d,
                                              Eina_Log_Level level,
                                              const char *file,
                                              const char *fnc,
                                              int line)
{
   Eina_Thread cur;

   DECLARE_LEVEL_NAME_COLOR(level);
   cur = SELF();
   if (IS_OTHER(cur))
     {
# ifdef _WIN32
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                                color);
        fprintf(fp, "%s", name);
        SetConsoleTextAttribute(GetStdHandle(
                                   STD_OUTPUT_HANDLE),
                                FOREGROUND_RED | FOREGROUND_GREEN |
                                FOREGROUND_BLUE);
        fprintf(fp, ":");
        SetConsoleTextAttribute(GetStdHandle(
                                   STD_OUTPUT_HANDLE),
                                eina_log_win32_color_get(d->domain_str));
        fprintf(fp, "%s[T:", d->name);
        SetConsoleTextAttribute(GetStdHandle(
                                   STD_OUTPUT_HANDLE),
                                FOREGROUND_RED | FOREGROUND_GREEN |
                                FOREGROUND_BLUE);
        fprintf(fp, "[T:");
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                                FOREGROUND_GREEN | FOREGROUND_BLUE);
        fprintf(fp, "%lu", (unsigned long)cur);
        SetConsoleTextAttribute(GetStdHandle(
                                   STD_OUTPUT_HANDLE),
                                FOREGROUND_RED | FOREGROUND_GREEN |
                                FOREGROUND_BLUE);
        fprintf(fp, "] %s:%d ", file, line);
        SetConsoleTextAttribute(GetStdHandle(
                                   STD_OUTPUT_HANDLE),
                                FOREGROUND_INTENSITY | FOREGROUND_RED |
                                FOREGROUND_GREEN | FOREGROUND_BLUE);
        fprintf(fp, "%s()", fnc);
        SetConsoleTextAttribute(GetStdHandle(
                                   STD_OUTPUT_HANDLE),
                                FOREGROUND_RED | FOREGROUND_GREEN |
                                FOREGROUND_BLUE);
        fprintf(fp, " ");
# else
        fprintf(fp, "%s%s<%u>" EINA_COLOR_RESET ":%s[T:"
                EINA_COLOR_ORANGE "%lu" EINA_COLOR_RESET "] %s:%d "
                EINA_COLOR_HIGH "%s()" EINA_COLOR_RESET " ",
                color, name, eina_log_pid_get() ,d->domain_str, 
                (unsigned long)cur, file, line, fnc);
# endif
        return;
     }

# ifdef _WIN32
   eina_log_print_prefix_NOthreads_color_file_func(fp,
                                                   d,
                                                   level,
                                                   file,
                                                   fnc,
                                                   line);
# else
   fprintf(fp, "%s%s<%u>" EINA_COLOR_RESET ":%s %s:%d "
           EINA_COLOR_HIGH "%s()" EINA_COLOR_RESET " ",
           color, name, eina_log_pid_get(), d->domain_str, file, line, fnc);
# endif
}

static void
eina_log_print_prefix_threads_color_NOfile_func(FILE *fp,
                                                const Eina_Log_Domain *d,
                                                Eina_Log_Level level,
                                                const char *file EINA_UNUSED,
                                                const char *fnc,
                                                int line EINA_UNUSED)
{
   Eina_Thread cur;

   DECLARE_LEVEL_NAME_COLOR(level);
   cur = SELF();
   if (IS_OTHER(cur))
     {
# ifdef _WIN32
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                                color);
        fprintf(fp, "%s", name);
        SetConsoleTextAttribute(GetStdHandle(
                                   STD_OUTPUT_HANDLE),
                                FOREGROUND_RED | FOREGROUND_GREEN |
                                FOREGROUND_BLUE);
        fprintf(fp, ":");
        SetConsoleTextAttribute(GetStdHandle(
                                   STD_OUTPUT_HANDLE),
                                eina_log_win32_color_get(d->domain_str));
        fprintf(fp, "%s[T:", d->name);
        SetConsoleTextAttribute(GetStdHandle(
                                   STD_OUTPUT_HANDLE),
                                FOREGROUND_RED | FOREGROUND_GREEN |
                                FOREGROUND_BLUE);
        fprintf(fp, "[T:");
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                                FOREGROUND_GREEN | FOREGROUND_BLUE);
        fprintf(fp, "%lu", (unsigned long)cur);
        SetConsoleTextAttribute(GetStdHandle(
                                   STD_OUTPUT_HANDLE),
                                FOREGROUND_INTENSITY | FOREGROUND_RED |
                                FOREGROUND_GREEN | FOREGROUND_BLUE);
        fprintf(fp, "%s()", fnc);
        SetConsoleTextAttribute(GetStdHandle(
                                   STD_OUTPUT_HANDLE),
                                FOREGROUND_RED | FOREGROUND_GREEN |
                                FOREGROUND_BLUE);
        fprintf(fp, " ");
# else
        fprintf(fp, "%s%s<%u>" EINA_COLOR_RESET ":%s[T:"
                EINA_COLOR_ORANGE "%lu" EINA_COLOR_RESET "] "
                EINA_COLOR_HIGH "%s()" EINA_COLOR_RESET " ",
                color, name, eina_log_pid_get(), d->domain_str, 
                (unsigned long)cur, fnc);
# endif
        return;
     }

# ifdef _WIN32
   eina_log_print_prefix_NOthreads_color_NOfile_func(fp,
                                                     d,
                                                     level,
                                                     file,
                                                     fnc,
                                                     line);
# else
   fprintf(fp, "%s%s<%u>" EINA_COLOR_RESET ":%s "
           EINA_COLOR_HIGH "%s()" EINA_COLOR_RESET " ",
           color, name, eina_log_pid_get(), d->domain_str, fnc);
# endif
}

static void
eina_log_print_prefix_threads_color_file_NOfunc(FILE *fp,
                                                const Eina_Log_Domain *d,
                                                Eina_Log_Level level,
                                                const char *file,
                                                const char *fnc EINA_UNUSED,
                                                int line)
{
   Eina_Thread cur;

   DECLARE_LEVEL_NAME_COLOR(level);
   cur = SELF();
   if (IS_OTHER(cur))
     {
# ifdef _WIN32
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                                color);
        fprintf(fp, "%s", name);
        SetConsoleTextAttribute(GetStdHandle(
                                   STD_OUTPUT_HANDLE),
                                FOREGROUND_RED | FOREGROUND_GREEN |
                                FOREGROUND_BLUE);
        fprintf(fp, ":");
        SetConsoleTextAttribute(GetStdHandle(
                                   STD_OUTPUT_HANDLE),
                                eina_log_win32_color_get(d->domain_str));
        fprintf(fp, "%s[T:", d->name);
        SetConsoleTextAttribute(GetStdHandle(
                                   STD_OUTPUT_HANDLE),
                                FOREGROUND_RED | FOREGROUND_GREEN |
                                FOREGROUND_BLUE);
        fprintf(fp, "[T:");
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                                FOREGROUND_GREEN | FOREGROUND_BLUE);
        fprintf(fp, "%lu", (unsigned long)cur);
        SetConsoleTextAttribute(GetStdHandle(
                                   STD_OUTPUT_HANDLE),
                                FOREGROUND_RED | FOREGROUND_GREEN |
                                FOREGROUND_BLUE);
        fprintf(fp, "] %s:%d ", file, line);
# else
        fprintf(fp, "%s%s<%u>" EINA_COLOR_RESET ":%s[T:"
                EINA_COLOR_ORANGE "%lu" EINA_COLOR_RESET "] %s:%d ",
                color, name, eina_log_pid_get(), d->domain_str, 
                (unsigned long)cur, file, line);
# endif
        return;
     }

# ifdef _WIN32
   eina_log_print_prefix_NOthreads_color_file_NOfunc(fp,
                                                     d,
                                                     level,
                                                     file,
                                                     fnc,
                                                     line);
# else
        fprintf(fp, "%s%s" EINA_COLOR_RESET ":%s %s:%d ",
           color, name, d->domain_str, file, line);
# endif
}

static void (*_eina_log_print_prefix)(FILE *fp, const Eina_Log_Domain *d,
                                      Eina_Log_Level level, const char *file,
                                      const char *fnc,
                                      int line) =
   eina_log_print_prefix_NOthreads_color_file_func;

static inline void
eina_log_print_prefix_update(void)
{
   if (_disable_file && _disable_function)
     {
        fprintf(stderr, "ERROR: cannot have " EINA_LOG_ENV_FILE_DISABLE " and "
                EINA_LOG_ENV_FUNCTION_DISABLE " set at the same time, will "
                                              "just disable function.\n");
        _disable_file = 0;
     }

#define S(NOthread, NOcolor, NOfile, NOfunc) \
   _eina_log_print_prefix = \
      eina_log_print_prefix_ ## NOthread ## threads_ ## NOcolor ## color_ ## \
      NOfile \
      ## file_ ## NOfunc ## func

   if (_threads_enabled)
     {
        if (_disable_color)
          {
             if (_disable_file)
                S(,NO,NO,);
             else if (_disable_function)
                S(,NO,,NO);
             else
                S(,NO,,);
          }
        else
          {
             if (_disable_file)
                S(,,NO,);
             else if (_disable_function)
                S(,,,NO);
             else
                S(,,,);
          }

        return;
     }

   if (_disable_color)
     {
        if (_disable_file)
                S(NO,NO,NO,);
        else if (_disable_function)
                S(NO,NO,,NO);
        else
                S(NO,NO,,);
     }
   else
     {
        if (_disable_file)
                S(NO,,NO,);
        else if (_disable_function)
                S(NO,,,NO);
        else
                S(NO,,,);
     }

#undef S
}

/*
 * Creates a colored domain name string.
 */
static const char *
eina_log_domain_str_get(const char *name, const char *color)
{
   const char *d;

   if (color)
     {
        size_t name_len;
        size_t color_len;

        name_len = strlen(name);
        color_len = strlen(color);
        d =
           malloc(sizeof(char) *
                  (color_len + name_len + strlen(EINA_COLOR_RESET) + 1));
        if (!d)
           return NULL;

               memcpy((char *)d,                          color, color_len);
               memcpy((char *)(d + color_len),            name,  name_len);
               memcpy((char *)(d + color_len + name_len), EINA_COLOR_RESET,
               strlen(EINA_COLOR_RESET));
        ((char *)d)[color_len + name_len + strlen(EINA_COLOR_RESET)] = '\0';
     }
   else
      d = strdup(name);

   return d;
}

/*
 * Setups a new logging domain to the name and color specified. Note that this
 * constructor acts upon an pre-allocated object.
 */
static Eina_Log_Domain *
eina_log_domain_new(Eina_Log_Domain *d, Eina_Log_Timing *t,
                    const char *name, const char *color)
{
   EINA_SAFETY_ON_NULL_RETURN_VAL(d,    NULL);
   EINA_SAFETY_ON_NULL_RETURN_VAL(name, NULL);

   d->level = EINA_LOG_LEVEL_UNKNOWN;
   d->deleted = EINA_FALSE;

   if ((color) && (!_disable_color))
      d->domain_str = eina_log_domain_str_get(name, color);
   else
      d->domain_str = eina_log_domain_str_get(name, NULL);

   d->name = strdup(name);
   d->namelen = strlen(name);

   t->phase = NULL;

   return d;
}

/*
 * Frees internal strings of a log domain, keeping the log domain itself as a
 * slot for next domain registers.
 */
static void
eina_log_domain_free(Eina_Log_Domain *d)
{
   EINA_SAFETY_ON_NULL_RETURN(d);

   if (d->domain_str)
      free((char *)d->domain_str);

   if (d->name)
      free((char *)d->name);
}

/*
 * Parses domain levels passed through the env var.
 */
static void
eina_log_domain_parse_pendings(void)
{
   const char *start;

   if (!(start = getenv(EINA_LOG_ENV_LEVELS)))
      return;

   // name1:level1,name2:level2,name3:level3,...
   while (1)
     {
        Eina_Log_Domain_Level_Pending *p;
        char *end = NULL;
        char *tmp = NULL;
        long int level;

        end = strchr(start, ':');
        if (!end)
           break;

        // Parse level, keep going if failed
        level = strtol((char *)(end + 1), &tmp, 10);
        if (tmp == (end + 1))
           goto parse_end;

        // Parse name
        p = malloc(sizeof(Eina_Log_Domain_Level_Pending) + end - start + 1);
        if (!p)
           break;

        p->namelen = end - start;
        memcpy((char *)p->name, start, end - start);
        ((char *)p->name)[end - start] = '\0';
        p->level = level;

        _pending_list = eina_inlist_append(_pending_list, EINA_INLIST_GET(p));

parse_end:
        start = strchr(tmp, ',');
        if (start)
           start++;
        else
           break;
     }
}

static void
eina_log_domain_parse_pending_globs(void)
{
   const char *start;

   if (!(start = getenv(EINA_LOG_ENV_LEVELS_GLOB)))
      return;

   // name1:level1,name2:level2,name3:level3,...
   while (1)
     {
        Eina_Log_Domain_Level_Pending *p;
        char *end = NULL;
        char *tmp = NULL;
        long int level;

        end = strchr(start, ':');
        if (!end)
           break;

        // Parse level, keep going if failed
        level = strtol((char *)(end + 1), &tmp, 10);
        if (tmp == (end + 1))
           goto parse_end;

        // Parse name
        p = malloc(sizeof(Eina_Log_Domain_Level_Pending) + end - start + 1);
        if (!p)
           break;

        p->namelen = 0; /* not that useful */
        memcpy((char *)p->name, start, end - start);
        ((char *)p->name)[end - start] = '\0';
        p->level = level;

        _glob_list = eina_inlist_append(_glob_list, EINA_INLIST_GET(p));

parse_end:
        start = strchr(tmp, ',');
        if (start)
           start++;
        else
           break;
     }
}

static inline int
eina_log_domain_register_unlocked(const char *name, const char *color)
{
   Eina_Log_Domain_Level_Pending *pending = NULL;
   size_t namelen;
   unsigned int i;

   for (i = 0; i < _log_domains_count; i++)
     {
        if (_log_domains[i].deleted)
          {
             // Found a flagged slot, free domain_str and replace slot
             eina_log_domain_new(&_log_domains[i], &_log_timing[i], name, color);
             goto finish_register;
          }
     }

   if (_log_domains_count >= _log_domains_allocated)
     {
        Eina_Log_Domain *tmp;
        Eina_Log_Timing *tim;
        size_t size;

        if (!_log_domains)
           // special case for init, eina itself will allocate a dozen of domains
           size = 24;
        else
           // grow 8 buckets to minimize reallocs
           size = _log_domains_allocated + 8;

        tmp = realloc(_log_domains, sizeof(Eina_Log_Domain) * size);
        tim = realloc(_log_timing, sizeof (Eina_Log_Timing) * size);

        if (tmp && tim)
          {
             // Success!
             _log_domains = tmp;
             _log_timing = tim;
             _log_domains_allocated = size;
          }
        else
          {
             free(tmp);
             free(tim);
             return -1;
          }
     }

   // Use an allocated slot
   eina_log_domain_new(&_log_domains[i], &_log_timing[i], name, color);
   _log_domains_count++;

finish_register:
   namelen = _log_domains[i].namelen;

   EINA_INLIST_FOREACH(_pending_list, pending)
   {
      if ((namelen == pending->namelen) && (strcmp(pending->name, name) == 0))
        {
           _log_domains[i].level = pending->level;
           break;
        }
   }

   if (_log_domains[i].level == EINA_LOG_LEVEL_UNKNOWN)
     {
        EINA_INLIST_FOREACH(_glob_list, pending)
        {
           if (!fnmatch(pending->name, name, 0))
             {
                _log_domains[i].level = pending->level;
                break;
             }
        }
     }

   // Check if level is still UNKNOWN, set it to global
   if (_log_domains[i].level == EINA_LOG_LEVEL_UNKNOWN)
      _log_domains[i].level = _log_level;

   eina_log_timing(i, EINA_LOG_STATE_START, EINA_LOG_STATE_INIT);

   return i;
}

static inline Eina_Bool
eina_log_term_color_supported(const char *term)
{
   const char *tail;
   size_t len;

   if (!term)
      return EINA_FALSE;

   len = strlen(term);
   tail = term + 1;
   switch (term[0])
     {
      /* list of known to support color terminals,
       * take from gentoo's portage.
       */

      case 'x': /* xterm and xterm-(256)color */
         return ((strncmp(tail, "term", sizeof("term") - 1) == 0) &&
                 ((tail[sizeof("term") - 1] == '\0') ||
                  (strcmp(term + len - sizeof("color") + 1, "color") == 0)));

      case 'E': /* Eterm */
      case 'a': /* aterm */
      case 'k': /* kterm */
         return (strcmp(tail, "term") == 0);

      case 'r': /* xrvt or rxvt-unicode */
         return ((strncmp(tail, "xvt", sizeof("xvt") - 1) == 0) &&
                 ((tail[sizeof("xvt") - 1] == '\0') ||
                  (strcmp(tail + sizeof("xvt") - 1, "-unicode") == 0)));

      case 's': /* screen */
         return ((strncmp(tail, "creen", sizeof("creen") - 1) == 0) &&
                 ((tail[sizeof("creen") - 1] == '\0') ||
                  (strcmp(term + len - sizeof("color") + 1, "color") == 0)));

      case 'g': /* gnome */
         return (strcmp(tail, "nome") == 0);

      case 'i': /* interix */
         return (strcmp(tail, "nterix") == 0);

      default:
         return EINA_FALSE;
     }
}

static inline void
eina_log_domain_unregister_unlocked(int domain)
{
   Eina_Log_Domain *d;

   if ((unsigned int)domain >= _log_domains_count)
      return;

   eina_log_timing(domain, EINA_LOG_STATE_STOP, EINA_LOG_STATE_SHUTDOWN);

   d = &_log_domains[domain];
   eina_log_domain_free(d);
   d->deleted = 1;
}

static inline void
eina_log_print_unlocked(int domain,
                        Eina_Log_Level level,
                        const char *file,
                        const char *fnc,
                        int line,
                        const char *fmt,
                        va_list args)
{
   Eina_Log_Domain *d;

#ifdef EINA_SAFETY_CHECKS
   if (EINA_UNLIKELY((unsigned int)domain >= _log_domains_count) ||
       EINA_UNLIKELY(domain < 0))
     {
        if (file && fnc && fmt)
           fprintf(
              stderr,
              "CRI: %s:%d %s() eina_log_print() unknown domain %d, original message format '%s'\n",
              file,
              line,
              fnc,
              domain,
              fmt);
        else
           fprintf(
              stderr,
              "CRI: eina_log_print() unknown domain %d, original message format '%s'\n",
              domain,
              fmt ? fmt : "");

        if (_abort_on_critical)
           abort();

        return;
     }

#endif
   d = _log_domains + domain;
#ifdef EINA_SAFETY_CHECKS
   if (EINA_UNLIKELY(d->deleted))
     {
           fprintf(stderr,
                "ERR: eina_log_print() domain %d is deleted\n",
                domain);
        return;
     }

#endif

   if (level > d->level)
      return;

#ifdef _WIN32
   {
      char *wfmt;
      char *tmp;

      wfmt = strdup(fmt);
      if (!wfmt)
        {
           fprintf(stderr, "ERR: %s: can not allocate memory\n", __FUNCTION__);
           return;
        }

      tmp = wfmt;
      while (strchr(tmp, '%'))
        {
           tmp++;
           if (*tmp == 'z')
              *tmp = 'I';
        }
      _print_cb(d, level, file, fnc, line, wfmt, _print_cb_data, args);
      free(wfmt);
   }
#else
   _print_cb(d, level, file, fnc, line, fmt, _print_cb_data, args);
#endif

   if (EINA_UNLIKELY(_abort_on_critical) &&
       EINA_UNLIKELY(level <= _abort_level_on_critical))
      abort();
}

#endif

/**
 * @endcond
 */


/*============================================================================*
*                                 Global                                     *
*============================================================================*/

/**
 * @internal
 * @brief Initialize the log module.
 *
 * @return #EINA_TRUE on success, #EINA_FALSE on failure.
 *
 * This function sets up the log module of Eina. It is called by
 * eina_init().
 *
 * @see eina_init()
 *
 * @warning Not-MT: just call this function from main thread! The
 *          place where this function was called the first time is
 *          considered the main thread.
 */
Eina_Bool
eina_log_init(void)
{
#ifdef EINA_ENABLE_LOG
   const char *level, *tmp;
   int color_disable;

   assert((sizeof(_names) / sizeof(_names[0])) == EINA_LOG_LEVELS);

#ifdef EINA_LOG_BACKTRACE
   if ((tmp = getenv(EINA_LOG_ENV_BACKTRACE)))
     _backtrace_level = atoi(tmp);
#endif

   if ((tmp = getenv(EINA_LOG_ENV_COLOR_DISABLE)))
      color_disable = atoi(tmp);
   else
      color_disable = -1;

   /* Check if color is explicitly disabled */
   if (color_disable == 1)
      _disable_color = EINA_TRUE;

#ifndef _WIN32
   /* color was not explicitly disabled or enabled, guess it */
   else if (color_disable == -1)
     {
        if (!eina_log_term_color_supported(getenv("TERM")))
           _disable_color = EINA_TRUE;
        else
          {
             /* if not a terminal, but redirected to a file, disable color */
             int fd;

             if (_print_cb == eina_log_print_cb_stderr)
                fd = STDERR_FILENO;
             else if (_print_cb == eina_log_print_cb_stdout)
                fd = STDOUT_FILENO;
             else
                fd = -1;

             if ((fd >= 0) && (!isatty(fd)))
                _disable_color = EINA_TRUE;
          }
     }
#endif

#ifdef HAVE_SYSTEMD
   if (getenv("NOTIFY_SOCKET"))
      _print_cb = eina_log_print_cb_journald;
#endif

   if (getenv("EINA_LOG_TIMING"))
     _disable_timing = EINA_FALSE;

   if ((tmp = getenv(EINA_LOG_ENV_FILE_DISABLE)) && (atoi(tmp) == 1))
      _disable_file = EINA_TRUE;

   if ((tmp = getenv(EINA_LOG_ENV_FUNCTION_DISABLE)) && (atoi(tmp) == 1))
      _disable_function = EINA_TRUE;

   if ((tmp = getenv(EINA_LOG_ENV_ABORT)) && (atoi(tmp) == 1))
      _abort_on_critical = EINA_TRUE;

   if ((tmp = getenv(EINA_LOG_ENV_ABORT_LEVEL)))
      _abort_level_on_critical = atoi(tmp);

   eina_log_print_prefix_update();

   // Global log level
   if ((level = getenv(EINA_LOG_ENV_LEVEL)))
      _log_level = atoi(level);
#ifdef HAVE_SYSTEMD
   else if (getenv("NOTIFY_SOCKET") && (_print_cb == eina_log_print_cb_journald))
      _log_level = EINA_LOG_LEVEL_INFO;
#endif

   // Register UNKNOWN domain, the default logger
   EINA_LOG_DOMAIN_GLOBAL = eina_log_domain_register("", NULL);

   if (EINA_LOG_DOMAIN_GLOBAL < 0)
     {
        fprintf(stderr, "Failed to create global logging domain.\n");
        return EINA_FALSE;
     }

   // Parse pending domains passed through EINA_LOG_LEVELS_GLOB
   eina_log_domain_parse_pending_globs();

   // Parse pending domains passed through EINA_LOG_LEVELS
   eina_log_domain_parse_pendings();

   eina_log_timing(EINA_LOG_DOMAIN_GLOBAL,
                   EINA_LOG_STATE_STOP,
                   EINA_LOG_STATE_INIT);

#endif
   return EINA_TRUE;
}

/**
 * @internal
 * @brief Shut down the log module.
 *
 * @return #EINA_TRUE on success, #EINA_FALSE on failure.
 *
 * This function shuts down the log module set up by
 * eina_log_init(). It is called by eina_shutdown().
 *
 * @see eina_shutdown()
 *
 * @warning Not-MT: just call this function from main thread! The
 *          place where eina_log_init() (eina_init()) was called the
 *          first time is considered the main thread.
 */
Eina_Bool
eina_log_shutdown(void)
{
#ifdef EINA_ENABLE_LOG
   Eina_Inlist *tmp;

   eina_log_timing(EINA_LOG_DOMAIN_GLOBAL,
                   EINA_LOG_STATE_START,
                   EINA_LOG_STATE_SHUTDOWN);

   while (_log_domains_count--)
     {
        if (_log_domains[_log_domains_count].deleted)
           continue;

        eina_log_domain_free(&_log_domains[_log_domains_count]);
     }

   free(_log_domains);
   free(_log_timing);

   _log_timing = NULL;
   _log_domains = NULL;
   _log_domains_count = 0;
   _log_domains_allocated = 0;

   while (_glob_list)
     {
        tmp = _glob_list;
        _glob_list = _glob_list->next;
        free(tmp);
     }

   while (_pending_list)
     {
        tmp = _pending_list;
        _pending_list = _pending_list->next;
        free(tmp);
     }

#endif
   return EINA_TRUE;
}

/**
 * @internal
 * @brief Activate the log mutex.
 *
 * This function activate the mutex in the eina log module. It is called by
 * eina_threads_init().
 *
 * @see eina_threads_init()
 */
void
eina_log_threads_init(void)
{
#ifdef EINA_ENABLE_LOG
   if (_threads_inited) return;
   _main_thread = SELF();
   if (!INIT()) return;
   _threads_inited = EINA_TRUE;
#endif
}

/**
 * @internal
 * @brief Shut down the log mutex.
 *
 * This function shuts down the mutex in the log module.
 * It is called by eina_threads_shutdown().
 *
 * @see eina_threads_shutdown()
 */
void
eina_log_threads_shutdown(void)
{
#ifdef EINA_ENABLE_LOG
   if (!_threads_inited) return;
   CHECK_MAIN();
   SHUTDOWN();
   _threads_enabled = EINA_FALSE;
   _threads_inited = EINA_FALSE;
#endif
}

/*============================================================================*
*                                   API                                      *
*============================================================================*/

/**
 * @cond LOCAL
 */

EAPI int EINA_LOG_DOMAIN_GLOBAL = 0;

/**
 * @endcond
 */

EAPI void
eina_log_threads_enable(void)
{
#ifdef EINA_ENABLE_LOG
   if (_threads_enabled) return;
   if (!_threads_inited) eina_log_threads_init();
   _threads_enabled = EINA_TRUE;
   eina_log_print_prefix_update();
#endif
}

EAPI void
eina_log_print_cb_set(Eina_Log_Print_Cb cb, void *data)
{
#ifdef EINA_ENABLE_LOG
   LOG_LOCK();
   _print_cb = cb;
   _print_cb_data = data;
   eina_log_print_prefix_update();
   LOG_UNLOCK();
#else
   (void) cb;
   (void) data;
#endif
}

EAPI void
eina_log_level_set(int level)
{
#ifdef EINA_ENABLE_LOG
   _log_level = level;
   if (EINA_LIKELY((EINA_LOG_DOMAIN_GLOBAL >= 0) &&
                   ((unsigned int)EINA_LOG_DOMAIN_GLOBAL < _log_domains_count)))
      _log_domains[EINA_LOG_DOMAIN_GLOBAL].level = level;
#else
   (void) level;
#endif
}

EAPI int
eina_log_level_get(void)
{
#ifdef EINA_ENABLE_LOG
   return _log_level;
#else
   return 0;
#endif
}

EAPI Eina_Bool
eina_log_main_thread_check(void)
{
#ifdef EINA_ENABLE_LOG
   return ((!_threads_enabled) || IS_MAIN(SELF()));
#else
   return EINA_TRUE;
#endif
}

EAPI void
eina_log_color_disable_set(Eina_Bool disabled)
{
#ifdef EINA_ENABLE_LOG
   _disable_color = disabled;
#else
   (void) disabled;
#endif
}

EAPI Eina_Bool
eina_log_color_disable_get(void)
{
#ifdef EINA_ENABLE_LOG
   return _disable_color;
#else
   return EINA_TRUE;
#endif
}

EAPI void
eina_log_file_disable_set(Eina_Bool disabled)
{
#ifdef EINA_ENABLE_LOG
   _disable_file = disabled;
#else
   (void) disabled;
#endif
}

EAPI Eina_Bool
eina_log_file_disable_get(void)
{
#ifdef EINA_ENABLE_LOG
   return _disable_file;
#else
   return EINA_TRUE;
#endif
}

EAPI void
eina_log_function_disable_set(Eina_Bool disabled)
{
#ifdef EINA_ENABLE_LOG
   _disable_function = disabled;
#else
   (void) disabled;
#endif
}

EAPI Eina_Bool
eina_log_function_disable_get(void)
{
#ifdef EINA_ENABLE_LOG
   return _disable_function;
#else
   return EINA_TRUE;
#endif
}

EAPI void
eina_log_abort_on_critical_set(Eina_Bool abort_on_critical)
{
#ifdef EINA_ENABLE_LOG
   _abort_on_critical = abort_on_critical;
#else
   (void) abort_on_critical;
#endif
}

EAPI Eina_Bool
eina_log_abort_on_critical_get(void)
{
#ifdef EINA_ENABLE_LOG
   return _abort_on_critical;
#else
   return EINA_FALSE;
#endif
}

EAPI void
eina_log_abort_on_critical_level_set(int critical_level)
{
#ifdef EINA_ENABLE_LOG
   _abort_level_on_critical = critical_level;
#else
   (void) critical_level;
#endif
}

EAPI int
eina_log_abort_on_critical_level_get(void)
{
#ifdef EINA_ENABLE_LOG
   return _abort_level_on_critical;
#else
   return 0;
#endif
}

EAPI int
eina_log_domain_register(const char *name, const char *color)
{
#ifdef EINA_ENABLE_LOG
   int r;

   EINA_SAFETY_ON_NULL_RETURN_VAL(name, -1);

   LOG_LOCK();
   r = eina_log_domain_register_unlocked(name, color);
   LOG_UNLOCK();
   return r;
#else
   (void) name;
   (void) color;
   return 0;
#endif
}

EAPI void
eina_log_domain_unregister(int domain)
{
#ifdef EINA_ENABLE_LOG
   EINA_SAFETY_ON_FALSE_RETURN(domain >= 0);
   LOG_LOCK();
   eina_log_domain_unregister_unlocked(domain);
   LOG_UNLOCK();
#else
   (void) domain;
#endif
}

EAPI void
eina_log_domain_level_set(const char *domain_name, int level)
{
#ifdef EINA_ENABLE_LOG
   Eina_Log_Domain_Level_Pending *pending;
   size_t namelen;
   unsigned int i;

   EINA_SAFETY_ON_NULL_RETURN(domain_name);

   namelen = strlen(domain_name);

   for (i = 0; i < _log_domains_count; i++)
     {
        if (_log_domains[i].deleted)
           continue;

        if ((namelen != _log_domains[i].namelen) ||
            (strcmp(_log_domains[i].name, domain_name) != 0))
           continue;

        _log_domains[i].level = level;
        return;
     }

   EINA_INLIST_FOREACH(_pending_list, pending)
   {
      if ((namelen == pending->namelen) &&
          (strcmp(pending->name, domain_name) == 0))
        {
           pending->level = level;
           return;
        }
   }

   pending = malloc(sizeof(Eina_Log_Domain_Level_Pending) + namelen + 1);
   if (!pending)
      return;

   pending->level = level;
   pending->namelen = namelen;
   memcpy(pending->name, domain_name, namelen + 1);

   _pending_list = eina_inlist_append(_pending_list, EINA_INLIST_GET(pending));
#else
   (void) domain_name;
   (void) level;
#endif
}

EAPI int
eina_log_domain_level_get(const char *domain_name)
{
#ifdef EINA_ENABLE_LOG
   Eina_Log_Domain_Level_Pending *pending;
   size_t namelen;
   unsigned int i;

   EINA_SAFETY_ON_NULL_RETURN_VAL(domain_name, EINA_LOG_LEVEL_UNKNOWN);

   namelen = strlen(domain_name);

   for (i = 0; i < _log_domains_count; i++)
     {
        if (_log_domains[i].deleted)
           continue;

        if ((namelen != _log_domains[i].namelen) ||
            (strcmp(_log_domains[i].name, domain_name) != 0))
           continue;

        return _log_domains[i].level;
     }

   EINA_INLIST_FOREACH(_pending_list, pending)
   {
      if ((namelen == pending->namelen) &&
          (strcmp(pending->name, domain_name) == 0))
         return pending->level;
   }

   EINA_INLIST_FOREACH(_glob_list, pending)
   {
      if (!fnmatch(pending->name, domain_name, 0))
         return pending->level;
   }

   return _log_level;
#else
   (void) domain_name;
   return 0;
#endif
}

EAPI int
eina_log_domain_registered_level_get(int domain)
{
#ifdef EINA_ENABLE_LOG
   EINA_SAFETY_ON_FALSE_RETURN_VAL(domain >= 0, EINA_LOG_LEVEL_UNKNOWN);
   EINA_SAFETY_ON_FALSE_RETURN_VAL((unsigned int)domain < _log_domains_count,
                                   EINA_LOG_LEVEL_UNKNOWN);
   EINA_SAFETY_ON_TRUE_RETURN_VAL(_log_domains[domain].deleted,
                                  EINA_LOG_LEVEL_UNKNOWN);
   return _log_domains[domain].level;
#else
   (void) domain;
   return 0;
#endif
}

#ifdef EINA_LOG_BACKTRACE
# define DISPLAY_BACKTRACE(File, Level)			\
  if (EINA_UNLIKELY(Level < _backtrace_level))		\
    {							\
      void *bt[256];					\
      char **strings;					\
      int btlen;					\
      int i;						\
      							\
      btlen = backtrace((void **)bt, 256);		\
      strings = backtrace_symbols((void **)bt, btlen);	\
      fprintf(File, "*** Backtrace ***\n");		\
      for (i = 0; i < btlen; ++i)			\
	fprintf(File, "%s\n", strings[i]);		\
      free(strings);					\
    }
#else
# define DISPLAY_BACKTRACE(File, Level)
#endif

EAPI void
eina_log_print_cb_stderr(const Eina_Log_Domain *d,
                         Eina_Log_Level level,
                         const char *file,
                         const char *fnc,
                         int line,
                         const char *fmt,
                         EINA_UNUSED void *data,
                         va_list args)
{
#ifdef EINA_ENABLE_LOG
   _eina_log_print_prefix(stderr, d, level, file, fnc, line);
   vfprintf(stderr, fmt, args);
   putc('\n', stderr);
   DISPLAY_BACKTRACE(stderr, level);
#else
   (void) d;
   (void) level;
   (void) file;
   (void) fnc;
   (void) line;
   (void) fmt;
   (void) data;
   (void) args;
#endif
}

EAPI void
eina_log_print_cb_stdout(const Eina_Log_Domain *d,
                         Eina_Log_Level level,
                         const char *file,
                         const char *fnc,
                         int line,
                         const char *fmt,
                         EINA_UNUSED void *data,
                         va_list args)
{
#ifdef EINA_ENABLE_LOG
   _eina_log_print_prefix(stdout, d, level, file, fnc, line);
   vprintf(fmt, args);
   putchar('\n');
   DISPLAY_BACKTRACE(stdout, level);
#else
   (void) d;
   (void) level;
   (void) file;
   (void) fnc;
   (void) line;
   (void) fmt;
   (void) data;
   (void) args;
#endif
}

EAPI void
eina_log_print_cb_journald(const Eina_Log_Domain *d,
			   Eina_Log_Level level,
			   const char *file,
			   const char *fnc,
			   int line,
			   const char *fmt,
			   void *data EINA_UNUSED,
			   va_list args)
{
#ifdef HAVE_SYSTEMD
   char buf[12];
   char *tmp;
   Eina_Thread cur;

   vasprintf(&tmp, fmt, args);

   eina_convert_itoa(line, buf);

   cur = SELF();

#ifdef EINA_LOG_BACKTRACE
   if (EINA_LIKELY(level >= _backtrace_level))
#endif
     sd_journal_send_with_location(file, buf, fnc,
				   "PRIORITY=%i", level,
				   "MESSAGE=%s", tmp,
				   "EFL_DOMAIN=%s", d->domain_str,
				   "THREAD=%lu", cur,
				   NULL);
#ifdef EINA_LOG_BACKTRACE
   else
     {
        Eina_Strbuf *bts;
	char **strings;
        void *bt[256];
	int btlen;
	int i;

	btlen = backtrace((void **)bt, 256);
	strings = backtrace_symbols((void **)bt, btlen);

	bts = eina_strbuf_new();
	for (i = 0; i < btlen; i++)
	  if (i + 1 == btlen)
	    eina_strbuf_append_printf(bts, "[%s]", strings[i]);
	  else
	    eina_strbuf_append_printf(bts, "[%s], ", strings[i]);

	sd_journal_send_with_location(file, buf, fnc,
				      "PRIORITY=%i", level,
				      "MESSAGE=%s", tmp,
				      "EFL_DOMAIN=%s", d->domain_str,
				      "THREAD=%lu", cur,
				      "BACKTRACE=%s", eina_strbuf_string_get(bts),
				      NULL);
	eina_strbuf_free(bts);
	free(strings);
     }
#endif

   free(tmp);

#else
   eina_log_print_cb_stderr(d, level, file, fnc, line, fmt, data, args);
#endif
}

EAPI void
eina_log_print_cb_file(const Eina_Log_Domain *d,
                       EINA_UNUSED Eina_Log_Level level,
                       const char *file,
                       const char *fnc,
                       int line,
                       const char *fmt,
                       void *data,
                       va_list args)
{
#ifdef EINA_ENABLE_LOG
   EINA_SAFETY_ON_NULL_RETURN(data);
   FILE *f = data;
   if (_threads_enabled)
     {
        Eina_Thread cur;

        cur = SELF();
        if (IS_OTHER(cur))
          {
             fprintf(f, "%s[T:%lu] %s:%d %s() ", d->name, (unsigned long)cur,
	        file, line, fnc);
             goto end;
          }
     }

   fprintf(f, "%s<%u> %s:%d %s() ", d->name, eina_log_pid_get(), 
           file, line, fnc);
   DISPLAY_BACKTRACE(f, level);

end:
   vfprintf(f, fmt, args);
   putc('\n', f);
#else
   (void) d;
   (void) file;
   (void) fnc;
   (void) line;
   (void) fmt;
   (void) data;
   (void) args;
#endif
}

EAPI void
eina_log_print(int domain, Eina_Log_Level level, const char *file,
               const char *fnc, int line, const char *fmt, ...)
{
#ifdef EINA_ENABLE_LOG
   va_list args;

#ifdef EINA_SAFETY_CHECKS
   if (EINA_UNLIKELY(!file))
     {
        fputs("ERR: eina_log_print() file == NULL\n", stderr);
        return;
     }

   if (EINA_UNLIKELY(!fnc))
     {
        fputs("ERR: eina_log_print() fnc == NULL\n", stderr);
        return;
     }

   if (EINA_UNLIKELY(!fmt))
     {
        fputs("ERR: eina_log_print() fmt == NULL\n", stderr);
        return;
     }

#endif
   va_start(args, fmt);
   LOG_LOCK();
   eina_log_print_unlocked(domain, level, file, fnc, line, fmt, args);
   LOG_UNLOCK();
   va_end(args);
#else
   (void) domain;
   (void) level;
   (void) file;
   (void) fnc;
   (void) line;
   (void) fmt;
#endif
}

EAPI void
eina_log_vprint(int domain, Eina_Log_Level level, const char *file,
                const char *fnc, int line, const char *fmt, va_list args)
{
#ifdef EINA_ENABLE_LOG

#ifdef EINA_SAFETY_CHECKS
   if (EINA_UNLIKELY(!file))
     {
        fputs("ERR: eina_log_print() file == NULL\n", stderr);
        return;
     }

   if (EINA_UNLIKELY(!fnc))
     {
        fputs("ERR: eina_log_print() fnc == NULL\n", stderr);
        return;
     }

   if (EINA_UNLIKELY(!fmt))
     {
        fputs("ERR: eina_log_print() fmt == NULL\n", stderr);
        return;
     }

#endif
   LOG_LOCK();
   eina_log_print_unlocked(domain, level, file, fnc, line, fmt, args);
   LOG_UNLOCK();
#else
   (void) domain;
   (void) level;
   (void) file;
   (void) fnc;
   (void) line;
   (void) fmt;
   (void) args;
#endif
}

EAPI void
eina_log_console_color_set(FILE *fp, const char *color)
{
#ifdef EINA_ENABLE_LOG

   EINA_SAFETY_ON_NULL_RETURN(fp);
   EINA_SAFETY_ON_NULL_RETURN(color);
   if (_disable_color) return;

#ifdef _WIN32
   int attr = eina_log_win32_color_convert(color, NULL);
   HANDLE *handle;

   if (!attr) return;

   if (fp == stderr)
     handle = GetStdHandle(STD_ERROR_HANDLE);
   else if (fp == stdout)
     handle = GetStdHandle(STD_OUTPUT_HANDLE);
   else
     {
        /* Do we have a way to convert FILE* to HANDLE?
         * Should we use it?
         */
        return;
     }
   SetConsoleTextAttribute(handle, attr);
#else
   fputs(color, fp);
#endif

#else
   (void)color;
#endif
}

EAPI void
eina_log_timing(int domain,
                Eina_Log_State state,
                const char *phase)
{
   //Eina_Log_Domain *d;
   Eina_Log_Timing *t;

   if (_disable_timing) return;

   //d = _log_domains + domain;
   t = _log_timing + domain;
#ifdef EINA_SAFETY_CHECKS
   if (EINA_UNLIKELY(d->deleted))
     {
        fprintf(stderr,
                "ERR: eina_log_print() domain %d is deleted\n",
                domain);
        return;
     }
#endif

   if (!t->phase && state == EINA_LOG_STATE_STOP)
     return;

   if (t->phase == EINA_LOG_STATE_INIT &&
       phase == EINA_LOG_STATE_SHUTDOWN)
     return;

   if (state == EINA_LOG_STATE_START &&
       t->phase &&
       strcmp(t->phase, phase)) // Different phase
     {
        fprintf(stderr, "%s vs %s\n", t->phase, phase);
        eina_log_timing(domain, EINA_LOG_STATE_STOP, t->phase);
     }

   switch (state)
     {
      case EINA_LOG_STATE_START:
        {
           _eina_time_get(&t->start);
           t->phase = phase;
           break;
        }
      case EINA_LOG_STATE_STOP:
        {
           Eina_Nano_Time end;
           long int r;

           _eina_time_get(&end);
           r = _eina_time_delta(&t->start, &end);
           EINA_LOG_DOM_INFO(domain, "%s timing: %li", t->phase, r);

           t->phase = NULL;
           break;
        }
     }
}
