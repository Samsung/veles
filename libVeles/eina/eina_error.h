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

#ifndef EINA_ERROR_H_
#define EINA_ERROR_H_

#include <stdarg.h>

#include "eina_types.h"


/**
 * @page tutorial_error_page Error Tutorial
 *
 * @section tutorial_error_registering_msg Registering messages
 *
 * The error module can provide a system that mimics the errno system
 * of the C standard library. It consists in 2 parts:
 *
 * @li a way of registering new messages with
 * eina_error_msg_register() and eina_error_msg_get(),
 * @li a way of setting / getting last error message with
 * eina_error_set() / eina_error_get().
 *
 * So one has to fisrt register all the error messages that a program
 * or a lib should manage. Then, when an error can occur, use
 * eina_error_set(), and when errors are managed, use
 * eina_error_get(). If eina_error_set() is used to set an error, do
 * not forget to call before eina_error_set(), to remove previous set
 * errors.
 *
 * Here is an example of use:
 *
 * @include eina_error_01.c
 *
 * Of course, instead of printf(), eina_log_print() can be used to
 * have beautiful error messages.
 */

/**
 * @addtogroup Eina_Error_Group Error
 *
 * @brief These functions provide error management for projects.
 *
 * The Eina error module provides a way to manage errors in a simple but
 * powerful way in libraries and modules. It is also used in Eina itself.
 * Similar to libC's @c errno and strerror() facilities, this is extensible and
 * recommended for other libraries and applications.
 *
 * A simple example of how to use this can be seen @ref tutorial_error_page
 * "here".
 */

/**
 * @addtogroup Eina_Tools_Group Tools
 *
 * @{
 */

/**
 * @defgroup Eina_Error_Group Error
 *
 * @{
 */

/**
 * @typedef Eina_Error
 * Error type.
 */
typedef int Eina_Error;

/**
 * @var EINA_ERROR_OUT_OF_MEMORY
 * Error identifier corresponding to a lack of memory.
 */

EAPI extern Eina_Error EINA_ERROR_OUT_OF_MEMORY;

/**
 * @brief Register a new error type.
 *
 * @param msg The description of the error. It will be duplicated using
 *        eina_stringshare_add().
 * @return The unique number identifier for this error.
 *
 * This function stores in a list the error message described by
 * @p msg. The returned value is a unique identifier greater or equal
 * than 1. The description can be retrieve later by passing to
 * eina_error_msg_get() the returned value.
 *
 * @see eina_error_msg_static_register()
 */
EAPI Eina_Error  eina_error_msg_register(const char *msg) EINA_ARG_NONNULL(1) EINA_WARN_UNUSED_RESULT;

/**
 * @brief Register a new error type, statically allocated message.
 *
 * @param msg The description of the error. This string will not be
 *        duplicated and thus the given pointer should live during
 *        usage of eina_error.
 * @return The unique number identifier for this error.
 *
 * This function stores in a list the error message described by
 * @p msg. The returned value is a unique identifier greater or equal
 * than 1. The description can be retrieve later by passing to
 * eina_error_msg_get() the returned value.
 *
 * @see eina_error_msg_register()
 */
EAPI Eina_Error  eina_error_msg_static_register(const char *msg) EINA_ARG_NONNULL(1) EINA_WARN_UNUSED_RESULT;

/**
 * @brief Change the message of an already registered message
 *
 * @param error The Eina_Error to change the message of
 * @param msg The description of the error. This string will be
 * duplicated only if the error was registered with @ref eina_error_msg_register
 * otherwise it must remain intact for the duration.
 * @return #EINA_TRUE if successful, #EINA_FALSE on error.
 *
 * This function modifies the message associated with @p error and changes
 * it to @p msg.  If the error was previously registered by @ref eina_error_msg_static_register
 * then the string will not be duplicated, otherwise the previous message
 * will be unrefed and @p msg copied.
 *
 * @see eina_error_msg_register()
 */
EAPI Eina_Bool   eina_error_msg_modify(Eina_Error  error,
                                       const char *msg) EINA_ARG_NONNULL(2);

/**
 * @brief Return the last set error.
 *
 * @return The last error.
 *
 * This function returns the last error set by eina_error_set(). The
 * description of the message is returned by eina_error_msg_get().
 */
EAPI Eina_Error  eina_error_get(void);

/**
 * @brief Set the last error.
 *
 * @param err The error identifier.
 *
 * This function sets the last error identifier. The last error can be
 * retrieved with eina_error_get().
 *
 * @note This is also used to clear previous errors, in that case @p err should
 * be @c 0.
 */
EAPI void        eina_error_set(Eina_Error err);

/**
 * @brief Return the description of the given an error number.
 *
 * @param error The error number.
 * @return The description of the error.
 *
 * This function returns the description of an error that has been
 * registered with eina_error_msg_register(). If an incorrect error is
 * given, then @c NULL is returned.
 */
EAPI const char *eina_error_msg_get(Eina_Error error) EINA_PURE;

/**
 * @brief Find the #Eina_Error corresponding to a message string
 * @param msg The error message string to match (NOT @c NULL)
 * @return The #Eina_Error matching @p msg, or 0 on failure
 * This function attempts to match @p msg with its corresponding #Eina_Error value.
 * If no such value is found, 0 is returned.
 */
EAPI Eina_Error  eina_error_find(const char *msg) EINA_ARG_NONNULL(1) EINA_PURE;

/**
 * @}
 */

/**
 * @}
 */

#endif /* EINA_ERROR_H_ */
