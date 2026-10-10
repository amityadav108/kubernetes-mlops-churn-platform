from importlib.metadata import version


def test_required_packages_are_installed() -> None:
    required_packages = {
        "fastapi": "0.115.12",
        "pandas": "2.2.3",
        "scikit-learn": "1.6.1",
        "pytest": "8.3.5",
    }

    for package, expected_version in required_packages.items():
        assert version(package) == expected_version
