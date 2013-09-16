/* EINA - EFL data type library
 * Copyright (C) 2008 Carsten Haitzler, Vincent Torri, Jorge Luis Zapata Muga
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

#ifndef EINA_PRIVATE_H_
#define EINA_PRIVATE_H_

#include <stdarg.h>

#include "eina_magic.h"
#include "eina_iterator.h"
#include "eina_accessor.h"

#ifndef MIN
# define MIN(x, y) (((x) > (y)) ? (y) : (x))
#endif

#ifndef MAX
# define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif

#ifndef ABS
# define ABS(x) ((x) < 0 ? -(x) : (x))
#endif

#ifndef CLAMP
# define CLAMP(x, min, \
               max) (((x) > (max)) ? (max) : (((x) < (min)) ? (min) : (x)))
#endif

#define EINA_INLIST_JUMP_SIZE 256

#define READBUFSIZ 65536

#define EINA_LOG_COLOR_DEFAULT "\033[36m"

/* eina magic types */
#define EINA_MAGIC_SHARE 0x98761234
#define EINA_MAGIC_SHARE_HEAD 0x98761235
#define EINA_MAGIC_STRINGSHARE_NODE 0x98761254
#define EINA_MAGIC_USTRINGSHARE_NODE 0x98761255
#define EINA_MAGIC_BINSHARE_NODE 0x98761256

#define EINA_MAGIC_LIST 0x98761237
#define EINA_MAGIC_LIST_ITERATOR 0x98761238
#define EINA_MAGIC_LIST_ACCESSOR 0x98761239
#define EINA_MAGIC_LIST_ACCOUNTING 0x9876123a

#define EINA_MAGIC_ARRAY 0x9876123b
#define EINA_MAGIC_ARRAY_ITERATOR 0x9876123c
#define EINA_MAGIC_ARRAY_ACCESSOR 0x9876123d

#define EINA_MAGIC_HASH 0x9876123e
#define EINA_MAGIC_HASH_ITERATOR 0x9876123f

#define EINA_MAGIC_TILER 0x98761240
#define EINA_MAGIC_TILER_ITERATOR 0x98761241

#define EINA_MAGIC_MATRIXSPARSE 0x98761242
#define EINA_MAGIC_MATRIXSPARSE_ROW 0x98761243
#define EINA_MAGIC_MATRIXSPARSE_CELL 0x98761244
#define EINA_MAGIC_MATRIXSPARSE_ITERATOR 0x98761245
#define EINA_MAGIC_MATRIXSPARSE_ROW_ITERATOR 0x98761246
#define EINA_MAGIC_MATRIXSPARSE_ROW_ACCESSOR 0x98761247
#define EINA_MAGIC_MATRIXSPARSE_CELL_ITERATOR 0x98761248
#define EINA_MAGIC_MATRIXSPARSE_CELL_ACCESSOR 0x98761249

#define EINA_MAGIC_STRBUF 0x98761250
#define EINA_MAGIC_USTRBUF 0x98761257
#define EINA_MAGIC_BINBUF 0x98761258

#define EINA_MAGIC_QUADTREE 0x98761251
#define EINA_MAGIC_QUADTREE_ROOT 0x98761252
#define EINA_MAGIC_QUADTREE_ITEM 0x98761253

#define EINA_MAGIC_SIMPLE_XML_TAG 0x98761260
#define EINA_MAGIC_SIMPLE_XML_DATA 0x98761261
#define EINA_MAGIC_SIMPLE_XML_ATTRIBUTE 0x98761262

#define EINA_MAGIC_INARRAY 0x98761270
#define EINA_MAGIC_INARRAY_ITERATOR 0x98761271
#define EINA_MAGIC_INARRAY_ACCESSOR 0x98761272

#define EINA_MAGIC_MODEL 0x98761280

#define EINA_MAGIC_CLASS 0x9877CB30

/* undef the following, we want out version */
#undef FREE
#define FREE(ptr)				\
  do {						\
     free(ptr);					\
     ptr = NULL;				\
  } while(0);

#undef IF_FREE
#define IF_FREE(ptr)				\
  do {						\
     if (ptr) {					\
	free(ptr);				\
	ptr = NULL;				\
     }						\
  } while(0);

#undef IF_FN_DEL
#define IF_FN_DEL(_fn, ptr)			\
  do {						\
     if (ptr) {					\
	_fn(ptr);				\
	ptr = NULL;				\
     }						\
  } while(0);

#define MAGIC_FREE(ptr)				\
  do {						\
     if (ptr) {					\
	EINA_MAGIC_SET(ptr, EINA_MAGIC_NONE);	\
	FREE(ptr);				\
     }						\
  } while(0);

#ifdef EFL_HAVE_THREADS
extern Eina_Bool _threads_activated;

void eina_share_common_threads_init(void);
void eina_share_common_threads_shutdown(void);
void eina_log_threads_init(void);
void eina_log_threads_shutdown(void);
#endif

void eina_cpu_count_internal(void);

void eina_file_mmap_faulty(void *addr, long page_size);

#include "eina_inline_private.h"

#endif /* EINA_PRIVATE_H_ */

