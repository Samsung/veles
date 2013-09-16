/* EINA - EFL data type library
 * Copyright (C) 2013 Jérémy Zurcher
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

#ifndef EINA_INLIST_INLINE_H_
#define EINA_INLIST_INLINE_H_

static inline Eina_Inlist *
eina_inlist_first(const Eina_Inlist *list)
{
   Eina_Inlist *l;

   if (!list) return NULL;

   for (l = (Eina_Inlist*)list; l->prev; l = l->prev);

   return l;
}

static inline Eina_Inlist *
eina_inlist_last(const Eina_Inlist *list)
{
   Eina_Inlist *l;

   if (!list) return NULL;

   for (l = (Eina_Inlist*)list; l->next; l = l->next);

   return l;
}

#endif /* EINA_INLIST_INLINE_H_ */
