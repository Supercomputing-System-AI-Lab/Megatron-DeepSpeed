# New stack only (2026-11-04, BUILD.md step 7): load Triton's native library with RTLD_DEEPBIND before anything else does.
# Why: the ROCm 7.14.1 HIP runtime (TheRock) pulls libamd_comgr -> libLLVM.so.23.0git into the process with its LLVM symbols
# exported; libtriton.so (triton-rocm 3.7.1) carries its own LLVM, and without deep binding its references resolve to the
# already-loaded LLVM 23 symbols: `import triton` dies with a segmentation fault in the library's loader (login-node test
# 2026-11-04 01:5x; the same crash the toy met in job 5841280). With RTLD_DEEPBIND libtriton binds to its own copy first.
import os as _os, sys as _sys
if _os.environ.get('NEWSTACK_TRITON_DEEPBIND', '1') == '1':
    _f = _sys.getdlopenflags()
    try:
        _sys.setdlopenflags(_f | _os.RTLD_DEEPBIND)
        import triton._C.libtriton  # noqa: F401
    except Exception as _e:  # never stop the interpreter here; the later import shows the real error
        print(f'[sitecustomize] triton deep-bind preload failed: {_e!r}', file=_sys.stderr)
    finally:
        _sys.setdlopenflags(_f)
