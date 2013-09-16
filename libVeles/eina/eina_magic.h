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

#ifndef EINA_MAGIC_H_
#define EINA_MAGIC_H_

#include "eina_config.h"
#include "eina_types.h"
#include "eina_error.h"

/**
 * @page eina_magic_example_01_page
 * @dontinclude eina_magic_01.c
 *
 * Whenever using Eina we must include it:
 * @skipline #include
 *
 * For this example we are going to define two classes, person and pilot, and
 * since every pilot is a person we use inheritance. To be type safe we are
 * going to add EINA_MAGIC to our classes:
 * @until struct _pilot pilot
 * @note The values of BASETYPE_MAGIC and SUBTYPE_MAGIC have no meaning, the
 * only important thing about them is that they be unique.
 *
 * Here we have a function to create a perso given a name, nothing too fancy:
 * @until }
 *
 * And now the counterpart, a function the free a person.
 * @until {
 * Before we start releasing resources we check that the pointer we were given
 * actually points to a person, and if not we will print an error message and
 * quit:
 * @until }
 * @note EINA_MAGIC_FAIL is a macro that make's it easy to print an appropriate
 * (and consistent) error message.
 * Now knowing that ptr is indeed of type person we prooced to set EINA_MAGIC to
 * EINA_MAGIC_NONE and free alocated memory:
 * @until }
 * @note Setting EINA_MAGIC to EINA_MAGIC_NONE is important to prevent the
 * struct from being used after freed.
 *
 * Now we have our function to create a pilot, this one is a little more complex
 * because we need to set EINA_MAGIC for the pilot and pilot->base, this is very
 * important so that checking the EINA_MAGIC of (person*)my_pilot will work:
 * @until }
 *
 * The function to free a pilot is not too different from the one that frees a
 * person:
 * @until }
 * @until }
 *
 * We also create functions to print a person or a pilot that check the type of
 * the pointers they receive:
 * @until }
 * @until }
 *
 * And on to our main function where we declare some variables and initialize
 * Eina:
 * @until eina_init
 *
 * For Eina to be able to provide more informative error messages we are going
 * to give names to our EINA_MAGIC types:
 * @until string_set
 *
 * Since our types won't live longer than the scope of the current function we
 * can set the name without eina making a copy of the string:
 * @until static_set
 *
 * Now we create a person, a pilot and print both as persons:
 * @until person *
 *
 * Now we try to print both as pilots, which will obvisouly not work since base
 * is not a pilot:
 * @until pilot(sub
 *
 * That's all folks:
 * @until }
 *
 * See full source @ref eina_magic_example_01_c "here".
 */
/**
 * @page eina_magic_example_01_c Eina_Magic
 * @include eina_magic_01.c
 * @example eina_magic_01.c
 */
/**
 * @addtogroup Eina_Tools_Group Tools
 *
 * @{
 */
/**
 * @defgroup Eina_Magic_Group Magic
 *
 * @brief Eina_Magic provides run-time type-checking.
 *
 * C is a weak statically typed language, in other words, it will just check for
 * types during compile time and any cast will make the compiler believe the 
 * type is correct.
 *
 * In real world code we often need to deal with casts, either explicit or
 * implicit by means of @c void*. We also need to resort to casts when doing
 * inheritance in C.
 *
 * Eina_Magic give us a way to do casts and still be certain of the type we are
 * opearting on.
 *
 * @note It should be noted that it is considered good practice to @b disable
 * Eina_Magic for production code. The reasoning is that any Eina_Magic errors
 * should have been caught during testing and therefore there is no reason to
 * incur the performance downside of Eina_Magic.
 *
 * An @ref eina_magic_example_01_page "example" should elucidate matters.
 *
 * @{
 */

/**
 * An abstract type for a magic number.
 */
typedef unsigned int Eina_Magic;

/**
 * @brief Return the string associated to the given magic identifier.
 *
 * @param magic The magic identifier.
 * @return The string associated to the identifier.
 *
 * This function returns the string associated to @p magic. Even if none are
 * found this function still returns non @c NULL, in this case an identifier
 * such as "(none)", "(undefined)" or "(unknown)".
 *
 * The following identifiers may be returned whenever magic is
 * invalid, with their meanings:
 *
 *   - (none): no magic was registered exists at all.
 *   - (undefined): magic was registered and found, but no string associated.
 *   - (unknown): magic was not found in the registry.
 *
 * @warning The returned value must not be freed.
 */
EAPI const char *eina_magic_string_get(Eina_Magic magic) EINA_WARN_UNUSED_RESULT;
/**
 * @brief Set the string associated to the given magic identifier.
 *
 * @param magic The magic identifier.
 * @param magic_name The string associated to the identifier, must not
 *        be @c NULL.
 *
 * @return #EINA_TRUE on success, #EINA_FALSE on failure.
 *
 * This function sets the string @p magic_name to @p magic. It is not
 * checked if number or string are already set, in which case you will end with
 * duplicates. Internally, eina will make a copy of @p magic_name.
 *
 * @see eina_magic_string_static_set()
 */
EAPI Eina_Bool   eina_magic_string_set(Eina_Magic  magic,
                                       const char *magic_name) EINA_ARG_NONNULL(2);

/**
 * @brief Set the string associated to the given magic identifier.
 *
 * @param magic The magic identifier.
 * @param magic_name The string associated to the identifier, must not be
 *        @c NULL.
 *
 * @return #EINA_TRUE on success, #EINA_FALSE on failure.
 *
 * This function sets the string @p magic_name to @p magic. It is not checked if
 * number or string are already set, in which case you might end with
 * duplicates. Eina will @b not make a copy of @p magic_name, this means that
 * @p magic_name has to be a valid pointer for as long as @p magic is used.
 *
 * @see eina_magic_string_set()
 */
EAPI Eina_Bool   eina_magic_string_static_set(Eina_Magic  magic,
                                              const char *magic_name) EINA_ARG_NONNULL(2);

/**
 * @def EINA_MAGIC_NONE
 * Random value for specifying that a structure using the magic
 * feature has already been freed. It is used by eina_magic_fail().
 *
 * If the magic feature of Eina is disabled, #EINA_MAGIC_NONE is just
 * @c 0.
 */
#define EINA_MAGIC_NONE 0x1234fedc

/**
 * @var EINA_ERROR_MAGIC_FAILED
 * Error identifier corresponding to magic check failure.
 */
EAPI extern Eina_Error EINA_ERROR_MAGIC_FAILED;


#ifdef EINA_MAGIC_DEBUG

/**
 * @def EINA_MAGIC
 * Declaration of a variable of type #Eina_Magic. To put in a structure
 * when one wants to use the magic feature of Eina with the functions
 * of that structure, like that:
 *
 * @code
 * struct Foo
 * {
 *    int i;
 *
 *    EINA_MAGIC
 * };
 * @endcode
 *
 * If the magic feature of Eina is disabled, #EINA_MAGIC does nothing.
 */
#define EINA_MAGIC Eina_Magic __magic;

/**
 * @def EINA_MAGIC_SET(d, m)
 * Set the magic number of @p d to @p m. @p d must be a valid pointer
 * to a structure holding an Eina magic number declaration.
 * Use #EINA_MAGIC to add such declaration.
 *
 * If the magic feature of Eina is disabled, #EINA_MAGIC_CHECK is just
 * the value @c 0.
 */
#define EINA_MAGIC_SET(d, m)   (d)->__magic = (m)

/**
 * @def EINA_MAGIC_CHECK(d, m)
 * Test if @p d is @c NULL or not, and if not @c NULL, if
 * @p d->__eina_magic is equal to @p m. @p d must be a structure that
 * holds an Eina magic number declaration. Use #EINA_MAGIC to add such
 * declaration.
 *
 * If the magic feature of Eina is disabled, #EINA_MAGIC_CHECK is just
 * the value @c 1.
 */
#define EINA_MAGIC_CHECK(d, m) (EINA_LIKELY((d) && ((d)->__magic == (m))))

/**
 * @def EINA_MAGIC_FAIL(d, m)
 * Call eina_magic_fail() with the parameters @p d, @p d->__magic, @p
 * m, __FILE__, __FUNCTION__ and __LINE__. @p d must be a structure that
 * holds an Eina magic number declaration. Use #EINA_MAGIC to add such
 * declaration.
 *
 * If the magic feature of Eina is disabled, #EINA_MAGIC_FAIL does
 * nothing.
 */
#define EINA_MAGIC_FAIL(d, m)             \
  eina_magic_fail((void *)(d),            \
                  (d) ? (d)->__magic : 0, \
                  (m),                    \
                  __FILE__,               \
                  __FUNCTION__,           \
                  __LINE__);

/**
 * @brief Display a message or abort if a magic check failed.
 *
 * @param d The checked data pointer.
 * @param m The magic identifer to check.
 * @param req_m The requested magic identifier to check.
 * @param file The file in which the magic check failed.
 * @param fnc The function in which the magic check failed.
 * @param line The line at which the magic check failed.
 *
 * @warning You should @b strongly consider using @ref EINA_MAGIC_FAIL(d, m)
 * instead.
 *
 * This function displays an error message if a magic check has
 * failed, using the following logic in the following order:
 * @li If @p d is @c NULL, a message warns about a @c NULL pointer.
 * @li Otherwise, if @p m is equal to #EINA_MAGIC_NONE, a message
 * warns about a handle that was already freed.
 * @li Otherwise, if @p m is equal to @p req_m, a message warns about
 * a handle that is of wrong type.
 * @li Otherwise, a message warns you about ab-using that function...
 *
 * If the environment variable EINA_LOG_ABORT is set, abort() is
 * called and the program stops. It is useful for debugging programs
 * with gdb.
 */
EAPI void eina_magic_fail(void *d, Eina_Magic m, Eina_Magic req_m,
                          const char *file, const char *fnc,
                          int line) EINA_ARG_NONNULL(4, 5);

#else

/**
 * @cond LOCAL
 */

#define EINA_MAGIC
#define EINA_MAGIC_SET(d, m)                          ((void)0)
#define EINA_MAGIC_CHECK(d, m)                        (1)
#define EINA_MAGIC_FAIL(d, m)                         ((void)0)

#define eina_magic_fail(d, m, req_m, file, fnx, line) ((void)0)

/**
 * @endcond
 */

#endif

/**
 * @}
 */

/**
 * @}
 */

#endif /* EINA_MAGIC_H_ */
