def ensure_cppad_or_warn():
    try:
        import cppad_py  # noqa
    except Exception:
        print("[ADMISER] WARNING: cppad_py not found. "
              "Please run tools/build_cppad_py.sh to build & install it for your environment.")
