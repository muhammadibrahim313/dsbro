"""Tests for top-level dsbro package helpers."""

from __future__ import annotations

import dsbro


def test_version_returns_semver_like_string():
    assert dsbro.version() == "0.2.0"


def test_about_returns_project_metadata(capsys):
    about_text = dsbro.about()

    captured = capsys.readouterr()
    assert "Your Data Science Bro" in about_text
    assert "dsbro 0.2.0" in captured.out
    assert "Homepage: https://ibrahimqasmi.com" in captured.out
    assert "GitHub: https://github.com/muhammadibrahim313/dsbro" in captured.out


def test_help_supports_module_and_function_queries(capsys):
    full_help = dsbro.help()
    module_help = dsbro.help("utils")
    function_help = dsbro.help("setup")

    captured = capsys.readouterr()
    assert "dsbro available modules:" in full_help
    assert "text cleaning and nlp-style helpers".lower() in full_help.lower()
    assert "dsbro.utils" in module_help
    assert "Configure a notebook-friendly dsbro environment." in function_help
    assert captured.out
