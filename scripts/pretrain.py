"""Compatibility entry point for `fmco pretrain`."""

from fmco.cli import main

raise SystemExit(main(["pretrain", *(__import__("sys").argv[1:])]))
