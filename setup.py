from setuptools import setup, find_packages

setup(
    name="esot500syn",
    version="0.1.0",
    description="A tool for generating event-based VOT datasets in simulation.",
    
    # tell setuptools to find packages in 'src' directory
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    install_requires=[
        "mani-skill==3.0.0b21",
    ],

    entry_points={
        "console_scripts": [
            "esot500syn-gen = esot500syn.main:main_cli",
        ],
    },
)