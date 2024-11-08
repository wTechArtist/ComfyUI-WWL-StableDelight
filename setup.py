from pathlib import Path
from setuptools import setup, find_packages

setup_path = Path(__file__).parent
README = (setup_path / "README.md").read_text(encoding="utf-8")

def split_requirements(requirements):
    install_requires = []
    dependency_links = []
    for requirement in requirements:
        if requirement.startswith("git+"):
            dependency_links.append(requirement)
        else:
            install_requires.append(requirement)
    return install_requires, dependency_links

with open("./requirements.txt", "r") as f:
    requirements = f.read().splitlines()

install_requires, dependency_links = split_requirements(requirements)

setup(
    name = "stabledelight",
    packages=find_packages(),
    description=README,
    long_description=README,
    install_requires=install_requires,
    entry_points={
        "comfyui_nodes": [
            "stabledelight = stabledelight.nodes:NODE_CLASS_MAPPINGS",
        ],
    }
)
