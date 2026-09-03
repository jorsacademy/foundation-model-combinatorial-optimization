"""Compatibility entry point for `fmco benchmark`."""

from fmco.cli import main

raise SystemExit(main(["benchmark", *(__import__("sys").argv[1:])]))
