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

#ifndef EINA_ITERATOR_H__
#define EINA_ITERATOR_H__

#include "eina_config.h"

#include "eina_types.h"
#include "eina_magic.h"

/**
 * @page eina_iterator_example_page Eina_Iterator usage
 * @dontinclude eina_iterator_01.c
 *
 * As always when using eina we need to include it:
 * @skip #include
 * @until Eina.h
 *
 * Here we a declare an unimpressive @ref Eina_Each_Cb "function" that prints
 * some data:
 * @until }
 * @note Returning EINA_TRUE is important so we don't stop iterating over the
 * container.
 *
 * And here a more interesting function, it uses an iterator to print the
 * contents of a container. What's interesting about it is that it doesn't care
 * the type of container, it works for anything that can provide an iterator:
 * @until }
 *
 * And on to our main function were we declare some variables and initialize
 * eina, nothing too special:
 * @until eina_init
 *
 * Next we populate both an array and a list with our strings, for more details
 * see @ref eina_list_01_example_page and @ref eina_array_01_example_page :
 * @until }
 *
 * And now we create an array and because the first element of the container
 * doesn't interest us we skip it:
 * @until iterator_next
 *
 * Having our iterator now pointing to interesting data we go ahead and print:
 * @until print_eina_container
 *
 * As always once data with a structure we free it, but just because we can we
 * do it by asking the iterator for it's container, and then of course free the
 * iterator itself:
 * @until eina_iterator_free
 *
 * But so far you're not impressed in @ref eina_array_01_example_page an array is
 * also printed, so now we go to the cool stuff and use an iterator to do same
 * stuff to a list:
 * @until eina_iterator_free
 * @note The only significant diference to the block above is in the
 * function used to create the iterator.
 *
 * And now we free the list and shut eina down:
 * @until }
 */

/**
 * @page eina_iterator_01_c Eina_Iterator usage
 * @page eina_iterator_01_c Eina_Iterator usage
 *
 * @include eina_iterator_01.c
 * @example eina_iterator_01.c
 */

/**
 * @addtogroup Eina_Iterator_Group Iterator Functions
 *
 * @brief These functions manage iterators on containers.
 *
 * These functions allow to access elements of a container in a
 * generic way, without knowing which container is used (a bit like
 * iterators in the C++ STL). Iterators only allows sequential access
 * (that is, from an element to the next one). For random access, see
 * @ref Eina_Accessor_Group.
 *
 * Getting an iterator to access elements of a given container is done through
 * the functions of that particular container. There is no function to create
 * a generic iterator as iterators absolutely depend on the container. This
 * means you won't find iterator creation function here, those can be found on
 * the documentation of the container type you're using. Though created with
 * container specific functions iterators are always deleted with the same
 * function: eina_iterator_free().
 *
 * To get the data and iterate, use eina_iterator_next(). To call a function on
 * all the elements of a container, use eina_iterator_foreach().
 * 
 * Here an @ref eina_iterator_example_page "example"
 */

/**
 * @addtogroup Eina_Content_Access_Group Content Access
 *
 * @{
 */

/**
 * @defgroup Eina_Iterator_Group Iterator Functions
 *
 * @{
 */

/**
 * @typedef Eina_Iterator
 * Abstract type for iterators.
 */
typedef struct _Eina_Iterator Eina_Iterator;

/**
 * @typedef Eina_Iterator_Next_Callback
 * Type for a callback that returns the next element in a container.
 */
typedef Eina_Bool           (*Eina_Iterator_Next_Callback)(Eina_Iterator *it, void **data);

/**
 * @typedef Eina_Iterator_Get_Container_Callback
 * Type for a callback that returns the container.
 */
typedef void               *(*Eina_Iterator_Get_Container_Callback)(Eina_Iterator *it);

/**
 * @typedef Eina_Iterator_Free_Callback
 * Type for a callback that frees the container.
 */
typedef void                (*Eina_Iterator_Free_Callback)(Eina_Iterator *it);

/**
 * @typedef Eina_Iterator_Lock_Callback
 * Type for a callback that lock the container.
 */
typedef Eina_Bool           (*Eina_Iterator_Lock_Callback)(Eina_Iterator *it);

/**
 * @struct _Eina_Iterator
 * structure of an iterator
 *
 * If creating an iterator remember to set the type using @ref EINA_MAGIC_SET.
 */
struct _Eina_Iterator
{
#define EINA_ITERATOR_VERSION 1
   int                                  version; /**< Version of the Iterator API. */

   Eina_Iterator_Next_Callback          next          EINA_ARG_NONNULL(1, 2) EINA_WARN_UNUSED_RESULT; /**< Callback called when a next element is requested. */
   Eina_Iterator_Get_Container_Callback get_container EINA_ARG_NONNULL(1) EINA_WARN_UNUSED_RESULT; /**< Callback called when the container is requested. */
   Eina_Iterator_Free_Callback          free          EINA_ARG_NONNULL(1); /**< Callback called when the container is freed. */

   Eina_Iterator_Lock_Callback          lock          EINA_WARN_UNUSED_RESULT; /**< Callback called when the container is locked. */
   Eina_Iterator_Lock_Callback          unlock        EINA_WARN_UNUSED_RESULT; /**< Callback called when the container is unlocked. */

#define EINA_MAGIC_ITERATOR 0x98761233
   EINA_MAGIC
};

/**
 * @def FUNC_ITERATOR_NEXT(Function)
 * Helper macro to cast @p Function to a Eina_Iterator_Next_Callback.
 */
#define FUNC_ITERATOR_NEXT(Function)          ((Eina_Iterator_Next_Callback)Function)

/**
 * @def FUNC_ITERATOR_GET_CONTAINER(Function)
 * Helper macro to cast @p Function to a Eina_Iterator_Get_Container_Callback.
 */
#define FUNC_ITERATOR_GET_CONTAINER(Function) ((Eina_Iterator_Get_Container_Callback)Function)

/**
 * @def FUNC_ITERATOR_FREE(Function)
 * Helper macro to cast @p Function to a Eina_Iterator_Free_Callback.
 */
#define FUNC_ITERATOR_FREE(Function)          ((Eina_Iterator_Free_Callback)Function)

/**
 * @def FUNC_ITERATOR_LOCK(Function)
 * Helper macro to cast @p Function to a Eina_Iterator_Lock_Callback.
 */
#define FUNC_ITERATOR_LOCK(Function)          ((Eina_Iterator_Lock_Callback)Function)


/**
 * @brief Free an iterator.
 *
 * @param iterator The iterator to free.
 *
 * This function frees @p iterator if it is not @c NULL;
 */
EAPI void      eina_iterator_free(Eina_Iterator *iterator);


/**
 * @brief Return the container of an iterator.
 *
 * @param iterator The iterator.
 * @return The container which created the iterator.
 *
 * This function returns the container which created @p iterator. If
 * @p iterator is @c NULL, this function returns @c NULL.
 */
EAPI void     *eina_iterator_container_get(Eina_Iterator *iterator) EINA_ARG_NONNULL(1) EINA_PURE;

/**
 * @brief Return the value of the current element and go to the next one.
 *
 * @param iterator The iterator.
 * @param data The data of the element.
 * @return #EINA_TRUE on success, #EINA_FALSE otherwise.
 *
 * This function returns the value of the current element pointed by
 * @p iterator in @p data, then goes to the next element. If @p
 * iterator is @c NULL or if a problem occurred, #EINA_FALSE is
 * returned, otherwise #EINA_TRUE is returned.
 */
EAPI Eina_Bool eina_iterator_next(Eina_Iterator *iterator,
                                  void         **data) EINA_ARG_NONNULL(2) EINA_WARN_UNUSED_RESULT;


/**
 * @brief Iterate over the container and execute a callback on each element.
 *
 * @param iterator The iterator.
 * @param callback The callback called on each iteration.
 * @param fdata The data passed to the callback.
 *
 * This function iterates over the elements pointed by @p iterator,
 * beginning from the current element. For Each element, the callback
 * @p cb is called with the data @p fdata. If @p iterator is @c NULL,
 * the function returns immediately. Also, if @p cb returns #EINA_FALSE,
 * the iteration stops at that point, if @p cb returns #EINA_TRUE
 * the iteration continues.
 */
EAPI void eina_iterator_foreach(Eina_Iterator *iterator,
                                Eina_Each_Cb   callback,
                                const void    *fdata) EINA_ARG_NONNULL(2);


/**
 * @brief Lock the container of the iterator.
 *
 * @param iterator The iterator.
 * @return #EINA_TRUE on success, #EINA_FALSE otherwise.
 *
 * If the container of the @p iterator permits it, it will be locked. When a
 * container is locked calling eina_iterator_foreach() on it will return
 * immediately. If @p iterator is @c NULL or if a problem occurred, #EINA_FALSE
 * is returned, otherwise #EINA_TRUE is returned. If the container isn't
 * lockable, it will return #EINA_TRUE.
 *
 * @warning None of the existing eina data structures are lockable.
 */
EAPI Eina_Bool eina_iterator_lock(Eina_Iterator *iterator) EINA_ARG_NONNULL(1);

/**
 * @brief Unlock the container of the iterator.
 *
 * @param iterator The iterator.
 * @return #EINA_TRUE on success, #EINA_FALSE otherwise.
 *
 * If the container of the @p iterator permits it and was previously
 * locked, it will be unlocked. If @p iterator is @c NULL or if a
 * problem occurred, #EINA_FALSE is returned, otherwise #EINA_TRUE
 * is returned. If the container is not lockable, it will
 * return #EINA_TRUE.
 *
 * @warning None of the existing eina data structures are lockable.
 */
EAPI Eina_Bool eina_iterator_unlock(Eina_Iterator *iterator) EINA_ARG_NONNULL(1);

/**
 * @def EINA_ITERATOR_FOREACH
 * @brief Macro to iterate over all elements easily.
 *
 * @param itr The iterator to use.
 * @param data Where to store * data, must be a pointer support getting
 *        its address since * eina_iterator_next() requires a pointer
 *        to pointer!
 *
 * This macro is a convenient way to use iterators, very similar to
 * EINA_LIST_FOREACH().
 *
 * This macro can be used for freeing the data of a list, like in the
 * following example. It has the same goal as the one documented in
 * EINA_LIST_FOREACH(), but using iterators:
 *
 * @code
 * Eina_List     *list;
 * Eina_Iterator *itr;
 * char          *data;
 *
 * // list is already filled,
 * // its elements are just duplicated strings
 *
 * itr = eina_list_iterator_new(list);
 * EINA_ITERATOR_FOREACH(itr, data)
 *   free(data);
 * eina_iterator_free(itr);
 * eina_list_free(list);
 * @endcode
 *
 * @note this example is not optimal algorithm to release a list since
 *    it will walk the list twice, but it serves as an example. For
 *    optimized version use EINA_LIST_FREE()
 *
 * @warning The order in which the elements will be traversed depends on the
 * underlying container and @b shouldn't be relied upon.
 *
 * @warning unless explicitly stated in functions returning iterators,
 *    do not modify the iterated object while you walk it, in this
 *    example using lists, do not remove list nodes or you might
 *    crash!  This is not a limitiation of iterators themselves,
 *    rather in the iterators implementations to keep them as simple
 *    and fast as possible.
 */
#define EINA_ITERATOR_FOREACH(itr,                                   \
                              data) while (eina_iterator_next((itr), \
                                                              (void **)(void *)&(data)))

/**
 * @}
 */

/**
 * @}
 */

#endif
