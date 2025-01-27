import glob
import os
from setuptools import setup, find_packages, Extension


cloneroot = os.path.dirname(__file__)


with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f.readlines() if not line.startswith("#")]

setup(
    name                 = "covid19sim",
    version              = "68c0b7ef4a3e41f41d20e6cd679b87fe3a38b6af",
    url                  = "https://github.com/aubreymcleod/COVI-AgentSim",
    description          = "Simulation of COVID-19 spread.",
    long_description     = "Simulation of COVID-19 spread.",
    classifiers          = [
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    zip_safe             = False,
    python_requires      = '>=3.7.4',
    install_requires     = requirements,
    extras_require       = {
        "ctt": [
            "ctt @ git+https://github.com/mila-iqia/COVI-ML@bunchacrunch#egg=ctt",
        ],
        "ctt-tf": [
            "ctt[tensorflow] @ git+https://github.com/mila-iqia/COVI-ML@master#egg=ctt"
        ],
    },
    packages             = find_packages("src"),
    package_dir          = {'': 'src'},
    ext_modules          = [
        Extension("covid19sim.native._native",
                  glob.glob(os.path.join(cloneroot, "src", "covid19sim", "native", "**", "*.c"),
                            recursive=True),
                  include_dirs=[os.path.join(cloneroot, "src", "covid19sim", "native")],
                  define_macros=[("PY_SSIZE_T_CLEAN", None),],
        ),
    ],
)
