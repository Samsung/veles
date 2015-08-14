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

#ifndef EINA_LOG_H_
#define EINA_LOG_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/types.h>

#if __GNUC__ >= 4
#pragma GCC visibility push(default)
#endif
#ifdef __cplusplus
extern "C" {
#endif

#include "eina_types.h"

#define EINA_COLOR_LIGHTRED     "\033[31;1m"
#define EINA_COLOR_RED          "\033[31m"
#define EINA_COLOR_LIGHTBLUE    "\033[34;1m"
#define EINA_COLOR_BLUE         "\033[34m"
#define EINA_COLOR_GREEN        "\033[32;1m"
#define EINA_COLOR_YELLOW       "\033[33;1m"
#define EINA_COLOR_ORANGE       "\033[0;33m"
#define EINA_COLOR_WHITE        "\033[37;1m"
#define EINA_COLOR_LIGHTCYAN    "\033[36;1m"
#define EINA_COLOR_CYAN         "\033[36m"
#define EINA_COLOR_RESET        "\033[0m"
#define EINA_COLOR_HIGH         "\033[1m"

Eina_Bool eina_log_init(void);
Eina_Bool eina_log_shutdown(void);

/**
 * @page tutorial_log_page Log Tutorial
 *
 * @section tutorial_log_introduction Introduction
 *
 * The Eina Log module provides logging facilities for libraries and
 * applications. It provides colored logging, basic logging levels (error,
 * warning, debug, info, critical) and loggers - called logging domains -
 * which will be covered on next sections.
 *
 * @section tutorial_log_basic_usage Basic Usage
 *
 * Log messages can be displayed using the following macros:
 *
 * @li EINA_LOG_ERR(),
 * @li EINA_LOG_INFO(),
 * @li EINA_LOG_WARN(),
 * @li EINA_LOG_DBG().
 *
 * Here is an example:
 *
 * <removed>
 *
 * If you compiled Eina without debug mode, execution will yield only one log
 * message, which is "argument is negative".
 *
 * Here we introduce the concept of logging domains (or loggers), which might
 * already be familiar to readers. It is basically a way to separate a set of
 * log messages into a context (e.g. a module) and provide a way of controlling
 * this set as a whole.
 *
 * For example, suppose you have 3 different modules on your application and you
 * want to get logging only from one of them (e.g. create some sort of filter).
 * For achieving that, all you need to do is create a logging domain for each
 * module so that all logging inside a module can be considered as a whole.
 *
 * Logging domains are specified by a name, color applied to the name and the
 * level. The first two (name and color) are set through code, that is, inside
 * your application/module/library.
 *
 * The level is used for controlling which messages should appear. It
 * specifies the lowest level that should be displayed (e.g. a message
 * with level 11 being logged on a domain with level set to 10 would be
 * displayed, while a message with level 9 wouldn't).
 *
 * The domain level is set during runtime (in contrast with the name and
 * color) through the environment variable EINA_LOG_LEVELS. This variable
 * expects a list in the form domain_name1:level1,domain_name2:level2,... . For
 * example:
 *
 * @verbatim EINA_LOG_LEVELS=mymodule1:5,mymodule2:2,mymodule3:0 ./myapp@endverbatim
 *
 * This line would set mymodule1 level to 5, mymodule2 level to 2 and mymodule3
 * level to 0.
 *
 * There's also a global logger to which EINA_LOG_(ERR, DBG, INFO, CRIT, WARN)
 * macros do log on. It is a logger that is created internally by Eina Log with
 * an empty name and can be used for general logging (where logging domains do
 * not apply).
 *
 * Since this global logger doesn't have a name, you can't set its level through
 * EINA_LOG_LEVELS variable. Here we introduce a second environment variable
 * that is a bit more special: EINA_LOG_LEVEL.
 *
 * This variable specifies the level of the global logging domain and the level
 * of domains that haven't been set through EINA_LOG_LEVELS. Here's an example:
 *
 * @verbatim EINA_LOG_LEVEL=3 EINA_LOG_LEVELS=module1:10,module3:2 ./myapp@endverbatim
 *
 * Supposing you have modules named "module1", "module2" and "module3", this
 * line would result in module1 with level 10, module2 with level 3 and module3
 * with level 2. Note that module2's level wasn't specified, so it's level is
 * set to the global level. This way we can easily apply filters to multiple
 * domains with only one parameter (EINA_LOG_LEVEL=num).
 *
 * The global level (EINA_LOG_LEVEL) can also be set through code, using
 * eina_log_level_set() function.
 *
 * While developing your libraries or applications, you may notice that
 * EINA_LOG_DOM_(ERR, DBG, INFO, CRIT, WARN) macros also print out
 * messages from eina itself. Here we introduce another environment variable
 * that is a bit more special: EINA_LOG_LEVELS_GLOB.
 *
 * This variable allows you to disable the logging of any/all code in eina itself.
 * This is useful when developing your libraries or applications so that you can
 * see your own domain's messages easier without having to sift through a lot of
 * internal eina debug messages. Here's an example:
 *
 * @verbatim EINA_LOG_LEVEL=3 EINA_LOG_LEVELS_GLOB=eina_*:0 ./myapp@endverbatim
 *
 * This will disable eina_log output from all internal eina code thus allowing
 * you to see your own domain messages easier.
 *
 * @section tutorial_log_advanced_display Advanced usage of print callbacks
 *
 * The log module allows the user to change the way
 * eina_log_print() displays the messages. It suffices to pass to
 * eina_log_print_cb_set() the function used to display the
 * message. That  function must be of type #Eina_Log_Print_Cb. As a
 * custom data can be passed to that callback, powerful display
 * messages can be displayed.
 *
 * It is suggested to not use __FILE__, __FUNCTION__ or __LINE__ when
 * writing that callback, but when defining macros (like
 * EINA_LOG_ERR() and other macros).
 *
 * Here is an example of custom callback, whose behavior can be
 * changed at runtime:
 *
 * <removed>
 * <removed>
 * <removed>
 */

/**
 * @addtogroup Eina_Log_Group Log
 *
 * @brief Full-featured logging system.
 *
 * Eina provides eina_log_print(), a standard function to manage all
 * logging messages. This function may be called directly or using the
 * helper macros such as EINA_LOG_DBG(), EINA_LOG_ERR() or those that
 * take a specific domain as argument EINA_LOG_DOM_DBG(),
 * EINA_LOG_DOM_ERR().  Internally, eina_log_print() will call the
 * function defined with eina_log_print_cb_set(), that defaults to
 * eina_log_print_cb_stderr(), but may be changed to do whatever you
 * need, such as networking or syslog logging.
 *
 * The logging system is thread safe once initialized with
 * eina_log_threads_enable(). The thread that calls this function
 * first is considered "main thread" and other threads will have their
 * thread id (pthread_self()) printed in the log message so it is easy
 * to detect from where it is coming.
 *
 * Log domains is the Eina way to differentiate messages. There might
 * be different domains to represent different modules, different
 * feature-set, different categories and so on. Filtering can be
 * applied to domain names by means of @c EINA_LOG_LEVELS environment
 * variable or eina_log_domain_level_set().
 *
 * The different logging levels serve to customize the amount of
 * debugging one want to take and may be used to automatically call
 * abort() once some given level message is printed. This is
 * controlled by environment variable @c EINA_LOG_ABORT and the level
 * to be considered critical with @c EINA_LOG_ABORT_LEVEL. These can
 * be changed with eina_log_abort_on_critical_set() and
 * eina_log_abort_on_critical_level_set().
 *
 * The default maximum level to print is defined by environment
 * variable @c EINA_LOG_LEVEL, but may be set per-domain with @c
 * EINA_LOG_LEVELS. It will default to #EINA_LOG_ERR. This can be
 * changed with eina_log_level_set().
 *
 * To use the log system Eina must be initialized with eina_init() and
 * later shut down with eina_shutdown(). Here is a straightforward
 * example:
 *
 * <removed>
 *
 * Compile this code with the following command:
 *
 * @verbatim gcc -Wall -o eina_log_01 eina_log_01.c `pkg-config --cflags --libs eina`@endverbatim
 *
 * Now execute the program with:
 *
 * @verbatim EINA_LOG_LEVEL=2 ./eina_log_01@endverbatim
 *
 * You should see a message displayed in the terminal.
 *
 * For more information, you can look at the @ref tutorial_log_page.
 *
 * <removed>
 */

/**
 * @addtogroup Eina_Tools_Group Tools
 *
 * @{
 */

/**
 * @defgroup Eina_Log_Group Log
 *
 * @{
 */

/**
 * EINA_LOG_DOMAIN_GLOBAL is the general purpose log domain to be
 * used, it is always registered and available everywhere.
 */
EAPI extern int EINA_LOG_DOMAIN_GLOBAL;

#ifndef EINA_LOG_DOMAIN_DEFAULT

/**
 * @def EINA_LOG_DOMAIN_DEFAULT
 * This macro defines the domain to use with the macros EINA_LOG_DOM_DBG(),
 * EINA_LOG_DOM_INFO(), EINA_LOG_DOM_WARN(), EINA_LOG_DOM_ERR() and
 * EINA_LOG_DOM_CRIT().
 *
 * If not defined prior to the inclusion of this header, then it
 * defaults to #EINA_LOG_DOMAIN_GLOBAL.
 *
 * @note One may like to redefine this in its code to avoid typing too
 *       much. In this case the recommended way is:
 *
 * @code
 * #include <Eina.h>
 * #undef EINA_LOG_DOMAIN_DEFAULT
 * #define EINA_LOG_DOMAIN_DEFAULT _log_dom
 * static int _log_dom = -1;
 *
 * int main(void)
 * {
 *    eina_init();
 *    _log_dom = eina_log_domain_register("mydom", EINA_COLOR_CYAN);
 *    EINA_LOG_ERR("using my own domain");
 *    return 0;
 * }
 * @endcode
 *
 * @warning If one defines the domain prior to inclusion of this
 *          header, the defined log domain symbol must be defined
 *          prior as well, otherwise the inlined functions defined by
 *          Eina will fail to find the symbol, causing build failure.
 *
 * @code
 * #define EINA_LOG_DOMAIN_DEFAULT _log_dom
 * static int _log_dom = -1; // must come before inclusion of Eina.h!
 * #include <Eina.h>
 *
 * int main(void)
 * {
 *    eina_init();
 *    _log_dom = eina_log_domain_register("mydom", EINA_COLOR_CYAN);
 *    EINA_LOG_ERR("using my own domain");
 *    return 0;
 * }
 * @endcode
 *
 */
# define EINA_LOG_DOMAIN_DEFAULT EINA_LOG_DOMAIN_GLOBAL

#endif /* EINA_LOG_DOMAIN_DEFAULT */

/**
 * @def EINA_LOG(DOM, LEVEL, fmt, ...)
 * Logs a message on the specified domain, level and format.
 *
 * @note if @c EINA_LOG_LEVEL_MAXIMUM is defined, then messages larger
 *       than this value will be ignored regardless of current domain
 *       level, the eina_log_print() is not even called! Most
 *       compilers will just detect the two integers make the branch
 *       impossible and remove the branch and function call all
 *       together. Take this as optimization tip and possible remove
 *       debug messages from binaries to be deployed, saving on hot
 *       paths. Never define @c EINA_LOG_LEVEL_MAXIMUM on public
 *       header files.
 */
#ifdef EINA_ENABLE_LOG
# ifdef EINA_LOG_LEVEL_MAXIMUM
# define EINA_LOG(DOM, LEVEL, fmt, ...)					\
  do {									\
     if (LEVEL <= EINA_LOG_LEVEL_MAXIMUM) {				\
        eina_log_print(DOM, LEVEL, __FILE__, __FUNCTION__, __LINE__,	\
                       fmt, ## __VA_ARGS__); }				\
  } while (0)
# else
# define EINA_LOG(DOM, LEVEL, fmt, ...) \
  eina_log_print(DOM,                   \
                 LEVEL,                 \
                 __FILE__,              \
                 __FUNCTION__,          \
                 __LINE__,              \
                 fmt,                   \
                 ## __VA_ARGS__)
# endif
#else
#define EINA_LOG(DOM, LEVEL, fmt, ...)          \
  do { (void) DOM; (void) LEVEL; (void) fmt; } while (0)
#endif

/**
 * @def EINA_LOG_DOM_CRIT(DOM, fmt, ...)
 * Logs a message with level CRITICAL on the specified domain and format.
 */
#define EINA_LOG_DOM_CRIT(DOM, fmt, ...) \
  EINA_LOG(DOM, EINA_LOG_LEVEL_CRITICAL, fmt, ## __VA_ARGS__)

/**
 * @def EINA_LOG_DOM_ERR(DOM, fmt, ...)
 * Logs a message with level ERROR on the specified domain and format.
 */
#define EINA_LOG_DOM_ERR(DOM, fmt, ...) \
  EINA_LOG(DOM, EINA_LOG_LEVEL_ERR, fmt, ## __VA_ARGS__)

/**
 * @def EINA_LOG_DOM_INFO(DOM, fmt, ...)
 * Logs a message with level INFO on the specified domain and format.
 */
#define EINA_LOG_DOM_INFO(DOM, fmt, ...) \
  EINA_LOG(DOM, EINA_LOG_LEVEL_INFO, fmt, ## __VA_ARGS__)

/**
 * @def EINA_LOG_DOM_DBG(DOM, fmt, ...)
 * Logs a message with level DEBUG on the specified domain and format.
 */
#define EINA_LOG_DOM_DBG(DOM, fmt, ...) \
  EINA_LOG(DOM, EINA_LOG_LEVEL_DBG, fmt, ## __VA_ARGS__)

/**
 * @def EINA_LOG_DOM_WARN(DOM, fmt, ...)
 * Logs a message with level WARN on the specified domain and format.
 */
#define EINA_LOG_DOM_WARN(DOM, fmt, ...) \
  EINA_LOG(DOM, EINA_LOG_LEVEL_WARN, fmt, ## __VA_ARGS__)

/**
 * @def EINA_LOG_CRIT(fmt, ...)
 * Logs a message with level CRITICAL on the default domain with the specified
 * format.
 */
#define EINA_LOG_CRIT(fmt, ...)     \
  EINA_LOG(EINA_LOG_DOMAIN_DEFAULT, \
           EINA_LOG_LEVEL_CRITICAL, \
           fmt,                     \
           ## __VA_ARGS__)

/**
 * @def EINA_LOG_ERR(fmt, ...)
 * Logs a message with level ERROR on the default domain with the specified
 * format.
 */
#define EINA_LOG_ERR(fmt, ...) \
  EINA_LOG(EINA_LOG_DOMAIN_DEFAULT, EINA_LOG_LEVEL_ERR, fmt, ## __VA_ARGS__)

/**
 * @def EINA_LOG_INFO(fmt, ...)
 * Logs a message with level INFO on the default domain with the specified
 * format.
 */
#define EINA_LOG_INFO(fmt, ...) \
  EINA_LOG(EINA_LOG_DOMAIN_DEFAULT, EINA_LOG_LEVEL_INFO, fmt, ## __VA_ARGS__)

/**
 * @def EINA_LOG_WARN(fmt, ...)
 * Logs a message with level WARN on the default domain with the specified
 * format.
 */
#define EINA_LOG_WARN(fmt, ...) \
  EINA_LOG(EINA_LOG_DOMAIN_DEFAULT, EINA_LOG_LEVEL_WARN, fmt, ## __VA_ARGS__)

/**
 * @def EINA_LOG_DBG(fmt, ...)
 * Logs a message with level DEBUG on the default domain with the specified
 * format.
 */
#define EINA_LOG_DBG(fmt, ...) \
  EINA_LOG(EINA_LOG_DOMAIN_DEFAULT, EINA_LOG_LEVEL_DBG, fmt, ## __VA_ARGS__)

/**
 * @typedef Eina_Log_Domain
 * The domain used for logging.
 */
typedef struct _Eina_Log_Domain Eina_Log_Domain;

/**
 * @struct _Eina_Log_Domain
 * The domain used for logging.
 */
struct _Eina_Log_Domain
{
   int         level; /**< Max level to log */
   const char *domain_str; /**< Formatted string with color to print */
   const char *name; /**< Domain name */
   size_t      namelen; /**< strlen(name) */

   /* Private */
   Eina_Bool   deleted : 1; /**< Flags deletion of domain, a free slot */
};

/**
 * Enable logging module to handle threads.
 *
 * There is no disable option on purpose, if it is enabled, there is
 * no way back until you call the last eina_shutdown().
 *
 * There is no function to retrieve if threads are enabled as one is
 * not supposed to know this from outside.
 *
 * After this call is executed at least once, if Eina was compiled
 * with threads support then logging will lock around debug messages
 * and threads that are not the main thread will have its identifier
 * printed.
 *
 * The main thread is considered the thread where the first
 * eina_init() was called.
 */
EAPI void eina_log_threads_enable(void);

/**
 * @enum _Eina_Log_Level
 * List of available logging levels.
 */
typedef enum _Eina_Log_Level
{
   EINA_LOG_LEVEL_CRITICAL, /**< Critical log level */
   EINA_LOG_LEVEL_ERR, /**< Error log level */
   EINA_LOG_LEVEL_WARN, /**< Warning log level */
   EINA_LOG_LEVEL_INFO, /**< Information log level */
   EINA_LOG_LEVEL_DBG, /**< Debug log level */
   EINA_LOG_LEVELS, /**< Count of default log levels */
   EINA_LOG_LEVEL_UNKNOWN = (-2147483647 - 1) /**< Unknown level */
} Eina_Log_Level;

/**
 * @typedef Eina_Log_Print_Cb
 * Type for print callbacks.
 */
typedef void (*Eina_Log_Print_Cb)(const Eina_Log_Domain *d,
                                  Eina_Log_Level level,
                                  const char *file, const char *fnc, int line,
                                  const char *fmt, void *data, va_list args);

typedef enum _Eina_Log_State
{
  EINA_LOG_STATE_START,
  EINA_LOG_STATE_STOP
} Eina_Log_State;

/*
 * Customization
 */

/**
 * Sets logging method to use.
 *
 * @param cb The callback to call when printing a log.
 * @param data The data to pass to the callback.
 *
 * By default, eina_log_print_cb_stderr() is used.
 *
 * @note MT: safe to call from any thread.
 *
 * @note MT: given function @a cb will be called protected by mutex.
 *       This means you're safe from other calls but you should never
 *       call eina_log_print(), directly or indirectly.
 */
EAPI void eina_log_print_cb_set(Eina_Log_Print_Cb cb, void *data) EINA_ARG_NONNULL(1);


/**
 * @brief Set the default log level.
 *
 * @param level The log level.
 *
 * This function sets the log level @p level. It is used in
 * eina_log_print().
 *
 * @note this is initially set to envvar EINA_LOG_LEVEL by eina_init().
 *
 * @see eina_log_level_get()
 */
EAPI void eina_log_level_set(int level);

/**
 * @brief Get the default log level.
 *
 * @return the log level that limits eina_log_print().
 *
 * @see eina_log_level_set()
 */
EAPI int  eina_log_level_get(void) EINA_WARN_UNUSED_RESULT;

static inline Eina_Bool eina_log_level_check(int level);

/**
 * Checks if current thread is the main thread.
 *
 * @return #EINA_TRUE if threads were enabled and the current thread
 *         is the one that called eina_log_threads_init(). If there is
 *         no thread support (compiled with --disable-pthreads) or
 *         they were not enabled, then #EINA_TRUE is also
 *         returned. The only case where #EINA_FALSE is returned is
 *         when threads were successfully enabled but the current
 *         thread is not the main (one that called
 *         eina_log_threads_init()).
 */
EAPI Eina_Bool          eina_log_main_thread_check(void) EINA_CONST EINA_WARN_UNUSED_RESULT;


/**
 * @brief Set if color logging should be disabled.
 *
 * @param disabled if #EINA_TRUE, color logging should be disabled.
 *
 * @note this is initially set to envvar EINA_LOG_COLOR_DISABLE by eina_init().
 *
 * @see eina_log_color_disable_get()
 */
EAPI void               eina_log_color_disable_set(Eina_Bool disabled);

/**
 * @brief Get if color logging should be disabled.
 *
 * @return if #EINA_TRUE, color logging should be disabled.
 *
 * @see eina_log_color_disable_set()
 */
EAPI Eina_Bool          eina_log_color_disable_get(void) EINA_WARN_UNUSED_RESULT;

/**
 * @brief Set if originating file name logging should be disabled.
 *
 * @param disabled if #EINA_TRUE, file name logging should be disabled.
 *
 * @note this is initially set to envvar EINA_LOG_FILE_DISABLE by eina_init().
 *
 * @see eina_log_file_disable_get()
 */
EAPI void               eina_log_file_disable_set(Eina_Bool disabled);

/**
 * @brief Get if originating file name logging should be disabled.
 *
 * @return if #EINA_TRUE, file name logging should be disabled.
 *
 * @see eina_log_file_disable_set()
 */
EAPI Eina_Bool          eina_log_file_disable_get(void) EINA_WARN_UNUSED_RESULT;

/**
 * @brief Set if originating function name logging should be disabled.
 *
 * @param disabled if #EINA_TRUE, function name logging should be disabled.
 *
 * @note this is initially set to envvar EINA_LOG_FUNCTION_DISABLE by
 *       eina_init().
 *
 * @see eina_log_function_disable_get()
 */
EAPI void               eina_log_function_disable_set(Eina_Bool disabled);

/**
 * @brief Get if originating function name logging should be disabled.
 *
 * @return if #EINA_TRUE, function name logging should be disabled.
 *
 * @see eina_log_function_disable_set()
 */
EAPI Eina_Bool          eina_log_function_disable_get(void) EINA_WARN_UNUSED_RESULT;

/**
 * @brief Set if critical messages should abort the program.
 *
 * @param abort_on_critical if #EINA_TRUE, messages with level equal
 *        or smaller than eina_log_abort_on_critical_level_get() will
 *        abort the program.
 *
 * @note this is initially set to envvar EINA_LOG_ABORT by
 *       eina_init().
 *
 * @see eina_log_abort_on_critical_get()
 * @see eina_log_abort_on_critical_level_set()
 */
EAPI void               eina_log_abort_on_critical_set(Eina_Bool abort_on_critical);

/**
 * @brief Get if critical messages should abort the program.
 *
 * @return if #EINA_TRUE, any messages with level equal or smaller
 *         than eina_log_abort_on_critical_level_get() will abort the
 *         program.
 *
 * @see eina_log_abort_on_critical_set()
 * @see eina_log_abort_on_critical_level_set()
 */
EAPI Eina_Bool          eina_log_abort_on_critical_get(void) EINA_WARN_UNUSED_RESULT;

/**
 * @brief Set level that triggers abort if abort-on-critical is set.
 *
 * @param critical_level levels equal or smaller than the given value
 *        will trigger program abortion if
 *        eina_log_abort_on_critical_get() returns #EINA_TRUE.
 *
 * @note this is initially set to envvar EINA_LOG_ABORT_LEVEL by
 *       eina_init().
 *
 * @see eina_log_abort_on_critical_level_get()
 * @see eina_log_abort_on_critical_get()
 */
EAPI void               eina_log_abort_on_critical_level_set(int critical_level);

/**
 * @brief Get level that triggers abort if abort-on-critical is set.
 *
 * @return critical level equal or smaller than value will trigger
 *        program abortion if eina_log_abort_on_critical_get()
 *        returns #EINA_TRUE.
 *
 * @see eina_log_abort_on_critical_level_set()
 * @see eina_log_abort_on_critical_get()
 */
EAPI int                eina_log_abort_on_critical_level_get(void) EINA_WARN_UNUSED_RESULT;


/**
 * Set the domain level given its name.
 *
 * This call has the same effect as setting
 * EINA_LOG_LEVELS=&lt;@p domain_name&gt;:&lt;@p level&gt;
 *
 * @param domain_name domain name to change the level. It may be of a
 *        still not registered domain. If the domain is not registered
 *        yet, it will be saved as a pending set and applied upon
 *        registration.
 * @param level level to use to limit eina_log_print() for given domain.
 */
EAPI void               eina_log_domain_level_set(const char *domain_name, int level) EINA_ARG_NONNULL(1);

/**
 * Get the domain level given its name.
 *
 * @param domain_name domain name to retrieve the level. It may be of
 *        a still not registered domain. If the domain is not
 *        registered yet, but there is a pending value, either from
 *        eina_log_domain_level_set(),EINA_LOG_LEVELS environment
 *        variable or from EINA_LOG_LEVELS_GLOB, these are
 *        returned. If nothing else was found, then the global/default
 *        level (eina_log_level_get()) is returned.
 *
 * @return level to use to limit eina_log_print() for given
 *         domain. On error (@p domain_name == NULL),
 *         EINA_LOG_LEVEL_UNKNOWN is returned.
 *
 * @see eina_log_domain_level_set()
 * @see eina_log_domain_registered_level_get()
 */
EAPI int                eina_log_domain_level_get(const char *domain_name) EINA_WARN_UNUSED_RESULT EINA_ARG_NONNULL(1);

/**
 * Get the domain level given its identifier.
 *
 * @param domain identifier, so it must be previously registered with
 *        eina_log_domain_register(). It's a much faster version of
 *        eina_log_domain_level_get(), but relies on domain being
 *        present.
 *
 * @return #EINA_TRUE if level should be printed, #EINA_FALSE if not.
 *         (domain's level is greater or equal @a level).
 */
EAPI int                eina_log_domain_registered_level_get(int domain) EINA_WARN_UNUSED_RESULT;

static inline Eina_Bool eina_log_domain_level_check(int domain, int level);

/*
 * Logging domains
 */

/**
 * @param name Domain name
 * @param color Color of the domain name
 *
 * @return Domain index that will be used as the DOMAIN parameter on log
 *         macros. A negative return value means an log occurred.
 *
 * @note MT: safe to call from any thread.
 */
EAPI int  eina_log_domain_register(const char *name, const char *color) EINA_ARG_NONNULL(1);

/**
 * Forget about a logging domain registered by eina_log_domain_register()
 *
 * @param domain domain identifier as reported by eina_log_domain_register(),
 *        must be >= 0.
 *
 * @note MT: safe to call from any thread.
 */
EAPI void eina_log_domain_unregister(int domain);

/*
 * Logging functions.
 */

/**
 * Print out log message using given domain and level.
 *
 * @note Usually you'll not use this function directly but the helper
 *       macros EINA_LOG(), EINA_LOG_DOM_CRIT(), EINA_LOG_CRIT() and
 *       so on. See eina_log.h
 *
 * @param domain logging domain to use or @c EINA_LOG_DOMAIN_GLOBAL if
 *        you registered none. It is recommended that modules and
 *        applications have their own logging domain.
 * @param level message level, those with level greater than user
 *        specified value (eina_log_level_set() or environment
 *        variables EINA_LOG_LEVEL, EINA_LOG_LEVELS) will be ignored.
 * @param file filename that originated the call, must @b not be @c NULL.
 * @param function function that originated the call, must @b not be @c NULL.
 * @param line originating line in @a file.
 * @param fmt printf-like format to use. Should not provide trailing
 *        '\n' as it is automatically included.
 *
 * @note MT: this function may be called from different threads if
 *       eina_log_threads_enable() was called before.
 */
EAPI void eina_log_print(int            domain,
                         Eina_Log_Level level,
                         const char    *file,
                         const char    *function,
                         int            line,
                         const char    *fmt,
                         ...) EINA_ARG_NONNULL(3, 4, 6) EINA_PRINTF(6, 7) EINA_NOINSTRUMENT;

/**
 * Print out log message using given domain and level.
 *
 * @note Usually you'll not use this function directly but the helper
 *       macros EINA_LOG(), EINA_LOG_DOM_CRIT(), EINA_LOG_CRIT() and
 *       so on. See eina_log.h
 *
 * @param domain logging domain to use or @c EINA_LOG_DOMAIN_GLOBAL if
 *        you registered none. It is recommended that modules and
 *        applications have their own logging domain.
 * @param level message level, those with level greater than user
 *        specified value (eina_log_level_set() or environment
 *        variables EINA_LOG_LEVEL, EINA_LOG_LEVELS) will be ignored.
 * @param file filename that originated the call, must @b not be @c NULL.
 * @param fnc function that originated the call, must @b not be @c NULL.
 * @param line originating line in @a file.
 * @param fmt printf-like format to use. Should not provide trailing
 *        '\n' as it is automatically included.
 * @param args the arguments needed by the format.
 *
 * @note MT: this function may be called from different threads if
 *       eina_log_threads_enable() was called before.
 *
 * @see eina_log_print()
 */
EAPI void eina_log_vprint(int            domain,
                          Eina_Log_Level level,
                          const char    *file,
                          const char    *fnc,
                          int            line,
                          const char    *fmt,
                          va_list        args) EINA_ARG_NONNULL(3, 4, 6) EINA_NOINSTRUMENT;

/*
 * Logging methods (change how logging is done).
 */

/**
 * @brief Alternative logging method, this will output to standard output stream.
 *
 * @param d The domain.
 * @param level The level.
 * @param file The file which is logged.
 * @param fnc The function which is logged.
 * @param line The line which is logged.
 * @param fmt The ouptut format to use.
 * @param data Not used.
 * @param args The arguments needed by the format.
 *
 * This method will colorize output based on domain provided color and
 * message logging level. To disable color, set environment variable
 * EINA_LOG_COLOR_DISABLE=1. Similarly, to disable file and line
 * information, set EINA_LOG_FILE_DISABLE=1 or
 * EINA_LOG_FUNCTION_DISABLE=1 to avoid function name in output. It is
 * not acceptable to have both EINA_LOG_FILE_DISABLE and
 * EINA_LOG_FUNCTION_DISABLE at the same time, in this case just
 * EINA_LOG_FUNCTION_DISABLE will be considered and file information
 * will be printed anyways.
 *
 * @note MT: if threads are enabled, this function is called within locks.
 * @note MT: Threads different from main thread will have thread id
 *       appended to domain name.
 */
EAPI void eina_log_print_cb_stdout(const Eina_Log_Domain *d,
                                   Eina_Log_Level         level,
                                   const char            *file,
                                   const char            *fnc,
                                   int                    line,
                                   const char            *fmt,
                                   void                  *data,
                                   va_list                args);

/**
 * @brief Default logging method, this will output to standard error stream.
 *
 * @param d The domain.
 * @param level The level.
 * @param file The file which is logged.
 * @param fnc The function which is logged.
 * @param line The line which is logged.
 * @param fmt The ouptut format to use.
 * @param data Not used.
 * @param args The arguments needed by the format.
 *
 * This method will colorize output based on domain provided color and
 * message logging level.
 *
 * To disable color, set environment variable
 * EINA_LOG_COLOR_DISABLE=1. To enable color, even if directing to a
 * file or when using a non-supported color terminal, use
 * EINA_LOG_COLOR_DISABLE=0. If EINA_LOG_COLOR_DISABLE is unset (or
 * -1), then Eina will disable color if terminal ($TERM) is
 * unsupported or if redirecting to a file.

   . Similarly, to disable file and line
 * information, set EINA_LOG_FILE_DISABLE=1 or
 * EINA_LOG_FUNCTION_DISABLE=1 to avoid function name in output. It is
 * not acceptable to have both EINA_LOG_FILE_DISABLE and
 * EINA_LOG_FUNCTION_DISABLE at the same time, in this case just
 * EINA_LOG_FUNCTION_DISABLE will be considered and file information
 * will be printed anyways.
 *
 * @note MT: if threads are enabled, this function is called within locks.
 * @note MT: Threads different from main thread will have thread id
 *       appended to domain name.
 */
EAPI void eina_log_print_cb_stderr(const Eina_Log_Domain *d,
                                   Eina_Log_Level         level,
                                   const char            *file,
                                   const char            *fnc,
                                   int                    line,
                                   const char            *fmt,
                                   void                  *data,
                                   va_list                args);

/**
 * Alternative logging method, this will output to given file stream.
 *
 * @param d The domain.
 * @param level Not used.
 * @param file The file which is logged.
 * @param fnc The function which is logged.
 * @param line The line which is logged.
 * @param fmt The ouptut format to use.
 * @param data The file which will store the output (as a FILE *).
 * @param args The arguments needed by the format.
 *
 * This method will never output color.
 *
 * @note MT: if threads are enabled, this function is called within locks.
 * @note MT: Threads different from main thread will have thread id
 *       appended to domain name.
 */
EAPI void eina_log_print_cb_file(const Eina_Log_Domain *d,
                                 Eina_Log_Level         level,
                                 const char            *file,
                                 const char            *fnc,
                                 int                    line,
                                 const char            *fmt,
                                 void                  *data,
                                 va_list                args);


/**
 * Alternative logging method, this will output to systemd journal.
 *
 * @param d The domain.
 * @param level Not used.
 * @param file The file which is logged.
 * @param fnc The function which is logged.
 * @param line The line which is logged.
 * @param fmt The ouptut format to use.
 * @param data The file which will store the output (as a FILE *).
 * @param args The arguments needed by the format.
 *
 * This method will never output color.
 *
 * @note if systemd journal is not there it will display error on stderr.
 * @note if the process has been started by systemd this will be the default logging method.
 *
 * @since 1.8
 */
EAPI void eina_log_print_cb_journald(const Eina_Log_Domain *d,
				     Eina_Log_Level level,
				     const char *file,
				     const char *fnc,
				     int line,
				     const char *fmt,
				     void *data,
				     va_list args);

/**
 * Configure console color of given file.
 *
 * @param fp file to configure console color (usually stderr or stdout).
 * @param color a VT color code such as #EINA_COLOR_RED or #EINA_COLOR_RESET.
 *
 * @note if color is disabled, nothing is done. See
 *       eina_log_color_disable_get()
 * @note on windows, both @a fp and @a color is converted automatically.
 *
 * @since 1.7
 */
EAPI void eina_log_console_color_set(FILE *fp,
                                     const char *color) EINA_ARG_NONNULL(1, 2);

extern EAPI const char *_eina_log_state_init;
extern EAPI const char *_eina_log_state_shutdown;

#define EINA_LOG_STATE_INIT _eina_log_state_init
#define EINA_LOG_STATE_SHUTDOWN _eina_log_state_shutdown

/**
 * @brief Start or stop the timing of a phase.
 *
 * @param d The domain.
 * @param state State indicating if we are starting or stopping a phase.
 * @param phase The name of the phase to be used in the log.
 *
 * @note One domain can be in only one phase at a time.
 * @note If you change the name of the phase, it is assumed that
 *    the previous phase has stopped.
 * @note The phase name should be available for all the life of the timing.
 * @since 1.8
 */
EAPI void eina_log_timing(int domain,
                          Eina_Log_State state,
                          const char *phase) EINA_ARG_NONNULL(3);

#include "eina_inline_log.x"

/**
 * @}
 */

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
#if __GNUC__ >= 4
#pragma GCC visibility pop
#endif

#endif /* EINA_LOG_H_ */
