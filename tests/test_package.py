import trustpipe


def test_package_exposes_installed_version() -> None:
    assert trustpipe.__version__ == "0.1.0"
