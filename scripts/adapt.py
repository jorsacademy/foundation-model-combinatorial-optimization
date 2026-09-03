"""Compatibility entry point for `fmco adapt`."""

from fmco.cli import main

raise SystemExit(main(["adapt", *(__import__("sys").argv[1:])]))
