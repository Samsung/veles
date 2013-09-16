/* EINA - EFL data type library
 * Copyright (C) 2002-2008 Carsten Haitzler, Vincent Torri
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

#include <stdlib.h>
#include <assert.h>

#include <stdio.h>

#include "eina_config.h"
#include "eina_private.h"
#include "eina_error.h"
#include "eina_log.h"

/* undefs EINA_ARG_NONULL() so NULL checks are not compiled out! */
#include "eina_safety_checks.h"
#include "eina_inlist.h"

/* FIXME: TODO please, refactor this :) */

/*============================================================================*
 *                                  Local                                     *
 *============================================================================*/

/**
 * @cond LOCAL
 */

#define EINA_INLIST_SORT_STACK_SIZE 32

typedef struct _Eina_Iterator_Inlist Eina_Iterator_Inlist;
typedef struct _Eina_Accessor_Inlist Eina_Accessor_Inlist;

struct _Eina_Iterator_Inlist
{
   Eina_Iterator iterator;
   const Eina_Inlist *head;
   const Eina_Inlist *current;
};

struct _Eina_Accessor_Inlist
{
   Eina_Accessor accessor;

   const Eina_Inlist *head;
   const Eina_Inlist *current;

   unsigned int index;
};

struct _Eina_Inlist_Sorted_State
{
   Eina_Inlist *jump_table[EINA_INLIST_JUMP_SIZE];

   unsigned short jump_limit;
   int jump_div;

   int inserted;
};

static Eina_Bool
eina_inlist_iterator_next(Eina_Iterator_Inlist *it, void **data)
{
   if (!it->current)
     return EINA_FALSE;

   if (data)
     *data = (void *)it->current;

   it->current = it->current->next;

   return EINA_TRUE;
}

static Eina_Inlist *
eina_inlist_iterator_get_container(Eina_Iterator_Inlist *it)
{
   return (Eina_Inlist *)it->head;
}

static void
eina_inlist_iterator_free(Eina_Iterator_Inlist *it)
{
   free(it);
}

static Eina_Bool
eina_inlist_accessor_get_at(Eina_Accessor_Inlist *it,
                            unsigned int idx,
                            void **data)
{
   const Eina_Inlist *over;
   unsigned int middle;
   unsigned int i;

   if (it->index == idx)
     over = it->current;
   else if (idx > it->index)
     /* Looking after current. */
     for (i = it->index, over = it->current;
          i < idx && over;
          ++i, over = over->next)
       ;
   else
     {
        middle = it->index >> 1;

        if (idx > middle)
          /* Looking backward from current. */
          for (i = it->index, over = it->current;
               i > idx && over;
               --i, over = over->prev)
            ;
        else
          /* Looking from the start. */
          for (i = 0, over = it->head;
               i < idx && over;
               ++i, over = over->next)
            ;
     }

   if (!over)
     return EINA_FALSE;

   it->current = over;
   it->index = idx;

   if (data)
     *data = (void *)over;

   return EINA_TRUE;
}

static Eina_Inlist *
eina_inlist_accessor_get_container(Eina_Accessor_Inlist *it)
{
   return (Eina_Inlist *)it->head;
}

static void
eina_inlist_accessor_free(Eina_Accessor_Inlist *it)
{
   free(it);
}

static Eina_Inlist *
eina_inlist_sort_merge(Eina_Inlist *a, Eina_Inlist *b, Eina_Compare_Cb func)
{
   Eina_Inlist *first, *last;

   if (func(a, b) < 0)
     a = (last = first = a)->next;
   else
     b = (last = first = b)->next;

   while (a && b)
     if (func(a, b) < 0)
       a = (last = last->next = a)->next;
     else
       b = (last = last->next = b)->next;

   last->next = a ? a : b;

   return first;
}

static Eina_Inlist *
eina_inlist_sort_rebuild_prev(Eina_Inlist *list)
{
   Eina_Inlist *prev = NULL;

   for (; list; list = list->next)
     {
        list->prev = prev;
        prev = list;
     }

   return prev;
}

static void
_eina_inlist_sorted_state_compact(Eina_Inlist_Sorted_State *state)
{
   unsigned short i, j;

   /* compress the jump table */
   state->jump_div *= 2;
   state->jump_limit /= 2;

   for (i = 2, j = 1;
        i < EINA_INLIST_JUMP_SIZE;
        i += 2, j++)
     state->jump_table[j] = state->jump_table[i];
}

/**
 * @endcond
 */


/*============================================================================*
 *                                 Global                                     *
 *============================================================================*/

/*============================================================================*
 *                                   API                                      *
 *============================================================================*/

EAPI Eina_Inlist *
eina_inlist_append(Eina_Inlist *list, Eina_Inlist *new_l)
{
   Eina_Inlist *l;

   EINA_SAFETY_ON_NULL_RETURN_VAL(new_l, list);

   new_l->next = NULL;
   if (!list)
     {
        new_l->prev = NULL;
        new_l->last = new_l;
        return new_l;
     }

   if (list->last)
     l = list->last;
   else
     for (l = list; (l) && (l->next); l = l->next)
       ;

   l->next = new_l;
   new_l->prev = l;
   list->last = new_l;
   return list;
}

EAPI Eina_Inlist *
eina_inlist_prepend(Eina_Inlist *list, Eina_Inlist *new_l)
{
   EINA_SAFETY_ON_NULL_RETURN_VAL(new_l, list);

   new_l->prev = NULL;
   if (!list)
     {
        new_l->next = NULL;
        new_l->last = new_l;
        return new_l;
     }

   new_l->next = list;
   list->prev = new_l;
   new_l->last = list->last;
   list->last = NULL;
   return new_l;
}

EAPI Eina_Inlist *
eina_inlist_append_relative(Eina_Inlist *list,
                            Eina_Inlist *new_l,
                            Eina_Inlist *relative)
{
   EINA_SAFETY_ON_NULL_RETURN_VAL(new_l, list);

   if (relative)
     {
        if (relative->next)
          {
             new_l->next = relative->next;
             relative->next->prev = new_l;
          }
        else
          new_l->next = NULL;

        relative->next = new_l;
        new_l->prev = relative;
        if (!new_l->next)
          list->last = new_l;

        return list;
     }

   return eina_inlist_append(list, new_l);
}

EAPI Eina_Inlist *
eina_inlist_prepend_relative(Eina_Inlist *list,
                             Eina_Inlist *new_l,
                             Eina_Inlist *relative)
{
   EINA_SAFETY_ON_NULL_RETURN_VAL(new_l, list);

   if (relative)
     {
        new_l->prev = relative->prev;
        new_l->next = relative;
        relative->prev = new_l;
        if (new_l->prev)
          {
             new_l->prev->next = new_l;
             /* new_l->next could not be NULL, as it was set to 'relative' */
             assert(new_l->next);
             return list;
          }
        else
          {
             /* new_l->next could not be NULL, as it was set to 'relative' */
             assert(new_l->next);

             new_l->last = list->last;
             list->last = NULL;
             return new_l;
          }
     }

   return eina_inlist_prepend(list, new_l);
}

EAPI Eina_Inlist *
eina_inlist_remove(Eina_Inlist *list, Eina_Inlist *item)
{
   Eina_Inlist *return_l;

   /* checkme */
   EINA_SAFETY_ON_NULL_RETURN_VAL(list, NULL);
   EINA_SAFETY_ON_NULL_RETURN_VAL(item, list);
   if (EINA_UNLIKELY((item != list) && (!item->prev) && (!item->next)))
     {
        //eina_error_set(EINA_ERROR_SAFETY_FAILED);
        EINA_LOG_ERR("safety check failed: item %p does not appear to be part of an inlist!", item);
        return list;
     }

   if (item->next)
     item->next->prev = item->prev;

   if (item->prev)
     {
        item->prev->next = item->next;
        return_l = list;
     }
   else
     {
        return_l = item->next;
        if (return_l)
          return_l->last = list->last;
     }

   if (item == list->last)
     list->last = item->prev;

   item->next = NULL;
   item->prev = NULL;
   return return_l;
}

EAPI Eina_Inlist *
eina_inlist_promote(Eina_Inlist *list, Eina_Inlist *item)
{
   EINA_SAFETY_ON_NULL_RETURN_VAL(list, NULL);
   EINA_SAFETY_ON_NULL_RETURN_VAL(item, list);

   if (item == list)
     return list;

   if (item->next)
     item->next->prev = item->prev;

   item->prev->next = item->next;

   if (list->last == item)
     list->last = item->prev;

   item->next = list;
   item->prev = NULL;
   item->last = list->last;

   list->prev = item;
   list->last = NULL;

   return item;
}

EAPI Eina_Inlist *
eina_inlist_demote(Eina_Inlist *list, Eina_Inlist *item)
{
   Eina_Inlist *l;

   EINA_SAFETY_ON_NULL_RETURN_VAL(list, NULL);
   EINA_SAFETY_ON_NULL_RETURN_VAL(item, list);

   if (list->last == item)
     return list;

   if (!list->last)
     {
        for (l = list; l->next; l = l->next)
          ;
        list->last = l;
     }

   l = list;
   if (item->prev)
     item->prev->next = item->next;
   else
     l = item->next;

   item->next->prev = item->prev;

   list->last->next = item;
   item->prev = list->last;
   item->next = NULL;

   l->last = item;
   return l;
}

EAPI Eina_Inlist *
eina_inlist_find(Eina_Inlist *list, Eina_Inlist *item)
{
   Eina_Inlist *l;

   EINA_SAFETY_ON_NULL_RETURN_VAL(item, NULL);

   for (l = list; l; l = l->next)
     {
        if (l == item)
          return item;
     }
   return NULL;
}

EAPI unsigned int
eina_inlist_count(const Eina_Inlist *list)
{
   const Eina_Inlist *l;
   unsigned int i = 0;

   for (l = list; l; l = l->next)
     i++;

   return i;
}

EAPI int
eina_inlist_sorted_state_init(Eina_Inlist_Sorted_State *state, Eina_Inlist *list)
{
   Eina_Inlist *ct = NULL;
   int count = 0;
   int jump_count = 1;

   /*
    * prepare a jump table to avoid doing unnecessary rewalk
    * of the inlist as much as possible.
    */
   for (ct = list; ct; ct = ct->next, jump_count++, count++)
     {
        if (jump_count == state->jump_div)
          {
             if (state->jump_limit == EINA_INLIST_JUMP_SIZE)
               {
                  _eina_inlist_sorted_state_compact(state);
               }

             state->jump_table[state->jump_limit] = ct;
             state->jump_limit++;
             jump_count = 0;
          }
     }

   state->inserted = count;
   return count;
}

EAPI Eina_Inlist_Sorted_State *
eina_inlist_sorted_state_new(void)
{
   Eina_Inlist_Sorted_State *r;

   r = calloc(1, sizeof (Eina_Inlist_Sorted_State));
   if (!r) return NULL;

   r->jump_div = 1;

   return r;
}

EAPI void
eina_inlist_sorted_state_free(Eina_Inlist_Sorted_State *state)
{
   free(state);
}

static void
_eina_inlist_sorted_state_insert(Eina_Inlist_Sorted_State *state,
                                 unsigned short idx,
                                 int offset)
{
   Eina_Inlist *last;
   int jump_count;
   int start;

   state->inserted++;

   if (offset != 0) idx++;
   for (; idx < state->jump_limit; idx++)
     {
        state->jump_table[idx] = state->jump_table[idx]->prev;
     }

   start = state->jump_limit - 3;
   if (start < 0)
     start = 0;

   last = state->jump_table[start];
   start++;

   /* Correctly rebuild end of list */
   for (jump_count = 0; last->next != NULL; last = last->next, jump_count++)
     {
        if (jump_count == state->jump_div)
          {
             if (state->jump_limit == start)
               {
                  if (state->jump_limit == EINA_INLIST_JUMP_SIZE)
                    {
                       _eina_inlist_sorted_state_compact(state);
                       start = state->jump_limit - 1;
                       continue ;
                    }
                  else
                    {
                       state->jump_limit++;
                    }
               }

             state->jump_table[start++] = last;
             jump_count = 0;
          }
     }
}

EAPI Eina_Inlist *
eina_inlist_sorted_insert(Eina_Inlist *list,
                          Eina_Inlist *item,
                          Eina_Compare_Cb func)
{
   Eina_Inlist *ct = NULL;
   Eina_Inlist_Sorted_State state;
   int cmp = 0;
   int inf, sup;
   int cur = 0;
   int count;

   EINA_SAFETY_ON_NULL_RETURN_VAL(item, list);
   EINA_SAFETY_ON_NULL_RETURN_VAL(func, list);

   if (!list) return eina_inlist_append(NULL, item);

   if (!list->next)
     {
        cmp = func(list, item);

        if (cmp < 0)
          return eina_inlist_append(list, item);
        return eina_inlist_prepend(list, item);
     }

   state.jump_div = 1;
   state.jump_limit = 0;
   count = eina_inlist_sorted_state_init(&state, list);

   /*
    * now do a dychotomic search directly inside the jump_table.
    */
   inf = 0;
   sup = state.jump_limit - 1;
   cur = 0;
   ct = state.jump_table[cur];
   cmp = func(ct, item);

   while (inf <= sup)
     {
        cur = inf + ((sup - inf) >> 1);
        ct = state.jump_table[cur];

        cmp = func(ct, item);
        if (cmp == 0)
          break ;
        else if (cmp < 0)
          inf = cur + 1;
        else if (cmp > 0)
          {
             if (cur > 0)
               sup = cur - 1;
             else
               break;
          }
        else
          break;
     }

   /* If at the beginning of the table and cmp < 0,
    * insert just after the head */
   if (cur == 0 && cmp > 0)
     return eina_inlist_prepend_relative(list, item, ct);

   /* If at the end of the table and cmp >= 0,
    * just append the item to the list */
   if (cmp < 0 && ct == list->last)
     return eina_inlist_append(list, item);

   /*
    * Now do a dychotomic search between two entries inside the jump_table
    */
   cur *= state.jump_div;
   inf = cur - state.jump_div - 1;
   sup = cur + state.jump_div + 1;

   if (sup > count - 1) sup = count - 1;
   if (inf < 0) inf = 0;

   while (inf <= sup)
     {
        int tmp = cur;

        cur = inf + ((sup - inf) >> 1);
        if (tmp < cur)
          for (; tmp != cur; tmp++, ct = ct->next);
        else if (tmp > cur)
          for (; tmp != cur; tmp--, ct = ct->prev);

        cmp = func(ct, item);
        if (cmp == 0)
          break ;
        else if (cmp < 0)
          inf = cur + 1;
        else if (cmp > 0)
          {
             if (cur > 0)
               sup = cur - 1;
             else
               break;
          }
        else
          break;
     }

   if (cmp <= 0)
     return eina_inlist_append_relative(list, item, ct);
   return eina_inlist_prepend_relative(list, item, ct);
}

EAPI Eina_Inlist *
eina_inlist_sorted_state_insert(Eina_Inlist *list,
                                Eina_Inlist *item,
                                Eina_Compare_Cb func,
                                Eina_Inlist_Sorted_State *state)
{
   Eina_Inlist *ct = NULL;
   int cmp = 0;
   int inf, sup;
   int cur = 0;
   int count;
   unsigned short head;
   unsigned int offset;

   if (!list)
     {
        state->inserted = 1;
        state->jump_limit = 1;
        state->jump_table[0] = item;
        return eina_inlist_append(NULL, item);
     }

   if (!list->next)
     {
        cmp = func(list, item);

        state->jump_limit = 2;
        state->inserted = 2;

        if (cmp < 0)
          {
             state->jump_table[1] = item;
             return eina_inlist_append(list, item);
          }
        state->jump_table[1] = state->jump_table[0];
        state->jump_table[0] = item;
        return eina_inlist_prepend(list, item);
     }

   count = state->inserted;

   /*
    * now do a dychotomic search directly inside the jump_table.
    */
   inf = 0;
   sup = state->jump_limit - 1;
   cur = 0;
   ct = state->jump_table[cur];
   cmp = func(ct, item);

   while (inf <= sup)
     {
        cur = inf + ((sup - inf) >> 1);
        ct = state->jump_table[cur];

        cmp = func(ct, item);
        if (cmp == 0)
          break ;
        else if (cmp < 0)
          inf = cur + 1;
        else if (cmp > 0)
          {
             if (cur > 0)
               sup = cur - 1;
             else
               break;
          }
        else
          break;
     }

   /* If at the beginning of the table and cmp < 0,
    * insert just after the head */
   if (cur == 0 && cmp > 0)
     {
        ct = eina_inlist_prepend_relative(list, item, ct);
        _eina_inlist_sorted_state_insert(state, 0, 0);
        return ct;
     }

   /* If at the end of the table and cmp >= 0,
    * just append the item to the list */
   if (cmp < 0 && ct == list->last)
     {
        ct = eina_inlist_append(list, item);
        _eina_inlist_sorted_state_insert(state, state->jump_limit - 1, 1);
        return ct;
     }

   /*
    * Now do a dychotomic search between two entries inside the jump_table
    */
   cur *= state->jump_div;
   inf = cur - state->jump_div - 1;
   sup = cur + state->jump_div + 1;

   if (sup > count - 1) sup = count - 1;
   if (inf < 0) inf = 0;

   while (inf <= sup)
     {
        int tmp = cur;

        cur = inf + ((sup - inf) >> 1);
        if (tmp < cur)
          for (; tmp != cur; tmp++, ct = ct->next);
        else if (tmp > cur)
          for (; tmp != cur; tmp--, ct = ct->prev);

        cmp = func(ct, item);
        if (cmp == 0)
          break ;
        else if (cmp < 0)
          inf = cur + 1;
        else if (cmp > 0)
          {
             if (cur > 0)
               sup = cur - 1;
             else
               break;
          }
        else
          break;
     }

   if (cmp <= 0)
     {
        cur++;

        ct = eina_inlist_append_relative(list, item, ct);
     }
   else
     {
        ct = eina_inlist_prepend_relative(list, item, ct);
     }

   head = cur / state->jump_div;
   offset = cur % state->jump_div;

   _eina_inlist_sorted_state_insert(state, head, offset);
   return ct;
}

EAPI Eina_Inlist *
eina_inlist_sort(Eina_Inlist *head, Eina_Compare_Cb func)
{
   unsigned int i = 0;
   unsigned int n = 0;
   Eina_Inlist *tail = head;
   Eina_Inlist *unsort = NULL;
   Eina_Inlist *stack[EINA_INLIST_SORT_STACK_SIZE];

   EINA_SAFETY_ON_NULL_RETURN_VAL(head, NULL);
   EINA_SAFETY_ON_NULL_RETURN_VAL(func, head);

   while (tail)
     {
        unsigned int idx, tmp;

        Eina_Inlist *a = tail;
        Eina_Inlist *b = tail->next;

        if (!b)
          {
             stack[i++] = a;
             break;
          }

        tail = b->next;

        if (func(a, b) < 0)
          ((stack[i++] = a)->next = b)->next = 0;
        else
          ((stack[i++] = b)->next = a)->next = 0;

        tmp = n++;
        for (idx = n ^ tmp; idx &= idx - 1; i--)
          stack[i - 2] = eina_inlist_sort_merge(stack[i - 2], stack[i - 1], func);
     }

   while (i-- > 1)
     stack[i - 1] = eina_inlist_sort_merge(stack[i - 1], stack[i], func);

   head = stack[0];
   tail = eina_inlist_sort_rebuild_prev(head);

   if (unsort)
     {
        tail->next = unsort;
        unsort->prev = tail;
     }

   head->last = tail;

   return head;

}

EAPI Eina_Iterator *
eina_inlist_iterator_new(const Eina_Inlist *list)
{
   Eina_Iterator_Inlist *it;

   //eina_error_set(0);
   it = calloc(1, sizeof (Eina_Iterator_Inlist));
   if (!it)
     {
        //eina_error_set(EINA_ERROR_OUT_OF_MEMORY);
        EINA_LOG_ERR("out of memory");
        return NULL;
     }

   it->head = list;
   it->current = list;

   it->iterator.version = EINA_ITERATOR_VERSION;
   it->iterator.next = FUNC_ITERATOR_NEXT(eina_inlist_iterator_next);
   it->iterator.get_container = FUNC_ITERATOR_GET_CONTAINER(
      eina_inlist_iterator_get_container);
   it->iterator.free = FUNC_ITERATOR_FREE(eina_inlist_iterator_free);

   EINA_MAGIC_SET(&it->iterator, EINA_MAGIC_ITERATOR);

   return &it->iterator;
}

EAPI Eina_Accessor *
eina_inlist_accessor_new(const Eina_Inlist *list)
{
   Eina_Accessor_Inlist *ac;

   //eina_error_set(0);
   ac = calloc(1, sizeof (Eina_Accessor_Inlist));
   if (!ac)
     {
        //eina_error_set(EINA_ERROR_OUT_OF_MEMORY);
        EINA_LOG_ERR("out of memory");
        return NULL;
     }

   ac->head = list;
   ac->current = list;
   ac->index = 0;

   ac->accessor.version = EINA_ACCESSOR_VERSION;
   ac->accessor.get_at = FUNC_ACCESSOR_GET_AT(eina_inlist_accessor_get_at);
   ac->accessor.get_container = FUNC_ACCESSOR_GET_CONTAINER(
      eina_inlist_accessor_get_container);
   ac->accessor.free = FUNC_ACCESSOR_FREE(eina_inlist_accessor_free);

   EINA_MAGIC_SET(&ac->accessor, EINA_MAGIC_ACCESSOR);

   return &ac->accessor;
}
