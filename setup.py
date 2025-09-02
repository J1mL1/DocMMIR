from setuptools import setup, find_packages

setup(
    name="docmmir",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # Treats /src as the root package
    install_requires=[
        "torch",
        "pytorch-lightning",
        "open-clip-torch",
        "transformers",
        "torchvision",
        "numpy",
    ],
)
