"""Stable script/import entry point for the shared Algorithms runtime."""
if __name__ == "__main__":
    from algorithms.runtime.cli import cli_main
    cli_main()
else:
    # Preserve the longstanding import/patch surface while implementation lives
    # in one place. Research utilities can migrate to the named runtime services.
    import sys
    from algorithms.runtime import job
    sys.modules[__name__] = job
