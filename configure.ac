#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.61)
AC_INIT([helios-base],[2023.03],[hidehisaakiyama@users.noreply.github.com])
AC_CONFIG_SRCDIR([config.h.in])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([m4])

# --------------------------------------------------------------
AM_INIT_AUTOMAKE([gnu check-news])

AC_LANG([C++])

# --------------------------------------------------------------
# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_RANLIB
AC_PROG_LN_S

# ----------------------------------------------------------
# check C++
AX_CXX_COMPILE_STDCXX_17(noext)

# ----------------------------------------------------------
# check boost
AX_BOOST_BASE([1.38.0])
AX_BOOST_SYSTEM
CPPFLAGS="$CPPFLAGS $BOOST_CPPFLAGS"
LDFLAGS="$LDFLAGS $BOOST_LDFLAGS"
LIBS="$BOOST_SYSTEM_LIB $LIBS"

# check librcsc path
AC_SUBST(LIBRCSC_LIBDIR)
LIBRCSC_LIBDIR="/usr/local/lib"
librcsc_prefix=""
AC_ARG_WITH(librcsc,
            [  --with-librcsc=PREFIX prefix where librcsc is installed (optional)],
            librcsc_prefix="$withval",
            librcsc_prefix="")
if test x$librcsc_prefix != x; then
  CPPFLAGS="$CPPFLAGS -I$librcsc_prefix/include"
  LDFLAGS="$LDFLAGS -L$librcsc_prefix/lib"
  LIBRCSC_LIBDIR="$librcsc_prefix/lib"
else
  for librcsc_path_tmp in $HOME/.local $HOME/local $HOME/rcss /opt/robocup /opt /opt/local ; do
    if test -d "$librcsc_path_tmp/include/rcsc" && test -r "$librcsc_path_tmp/include/rcsc/types.h" ; then
      CPPFLAGS="$CPPFLAGS -I$librcsc_path_tmp/include"
      LDFLAGS="$LDFLAGS -L$librcsc_path_tmp/lib"
      LIBRCSC_LIBDIR="$librcsc_path_tmp/lib"
      break;
    fi
  done
fi


# --------------------------------------------------------------
# Checks for libraries.
AC_CHECK_LIB([m], [cos],
             [LIBS="-lm $LIBS"],
             [AC_MSG_ERROR([*** -lm not found! ***])])
AC_CHECK_LIB([z], [deflate])

AC_CHECK_LIB([rcsc], [main],
             [LIBS="-lrcsc $LIBS"],
             [AC_MSG_ERROR([*** -lrcsc not found! ***])])

# --------------------------------------------------------------
# Checks for header files.
AC_CHECK_HEADERS([rcsc/types.h])

# --------------------------------------------------------------
# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T
AC_HEADER_TIME

# --------------------------------------------------------------
# Add original options
AC_ARG_ENABLE(debug,
              AC_HELP_STRING([--enable-debug],
                             [turn on debugging (default=no)]))
if test "x$enable_debug" = "xyes"; then
#  AC_DEFINE([DEBUG], [1], [debugging flag])
  CFLAGS="-DDEBUG $CFLAGS"
  CXXFLAGS="-DDEBUG $CXXFLAGS"
  echo "enabled debug"
fi

# --------------------------------------------------------------
# Checks for library functions.
AC_HEADER_STDC
AC_CHECK_FUNCS([floor pow sqrt strerror])

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 src/player/Makefile
                 src/coach/Makefile
                 src/trainer/Makefile
                 src/start.sh
                 src/keepaway.sh
                 src/train.sh],
                 [test -f src/start.sh && chmod +x src/start.sh]
                 [test -f src/keepaway.sh && chmod +x src/keepaway.sh]
                 [test -f src/train.sh && chmod +x src/train.sh])
AC_OUTPUT
