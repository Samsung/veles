bjam.launch:
	$(AM_V_GEN)BOOST_BUILD_USER_CONFIG="$(abs_builddir)/user-config.jam" $(BJAM) \
	--toolset=gcc-host --build-dir="$(abs_builddir)/libs" --stagedir="$(abs_builddir)/stage" \
	-j$$(getconf _NPROCESSORS_ONLN) link=static \
        cxxflags="$(AM_CXXFLAGS) $(CXXFLAGS) $(AM_CPPFLAGS) $(CPPFLAGS) -fPIC \
        -Wno-unused-local-typedefs -Wno-old-style-cast -Wno-unused-parameter" \
	linkflags="$(LDFLAGS)" "$(srcdir)"
	$(AM_V_at)echo stamp > bjam.launch

CLEANFILES = bjam.launch

all-local: bjam.launch	

install-data-local: bjam.launch

uninstall-local:

clean-local:
	$(AM_V_at)rm -rf $(builddir)/libs $(builddir)/stage

