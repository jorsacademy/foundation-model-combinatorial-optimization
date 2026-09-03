"""Compatibility entry point for `fmco research`."""

from fmco.cli import main

raise SystemExit(main(["research", *(__import__("sys").argv[1:])]))
