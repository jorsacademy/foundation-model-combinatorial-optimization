"""Compatibility entry point for `fmco collect`."""

from fmco.cli import main

raise SystemExit(main(["collect", *(__import__("sys").argv[1:])]))
