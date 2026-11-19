"""Package physical source roots without changing dependency configuration."""
from pathlib import Path

from setuptools import Distribution, find_packages, setup

ROOT = Path(__file__).parent
PACKAGE_DIRS = {
    "snd": "BA-MCHRO/snd",
    "m_asro": "BA-MCHRO/m_asro",
    "eoh": "BA-MCHRO/eoh",
    "ba_eoh_s": "BA-MCHRO/ba-eoh-s",
}


class SourceLayout(Distribution):
    """Resolve package locations after the existing pyproject metadata is read."""

    def parse_config_files(self, *args, **kwargs):
        super().parse_config_files(*args, **kwargs)
        self.packages = []
        self.package_dir = dict(PACKAGE_DIRS)
        self.include_package_data = False
        self.package_data = {"eoh": ["operator_prompts.txt"]}
        for name, folder in PACKAGE_DIRS.items():
            self.packages.append(name)
            self.packages.extend(f"{name}.{sub}" for sub in find_packages(str(ROOT / folder)))


setup(distclass=SourceLayout)
