from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='bottomline_project_backend',
      version="1.0",
      description="API for the back-end of Bottom___ project",
      packages=find_packages(),
      install_requires=requirements,
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      #scripts=['scripts/company_reputation_analyser-run'],
      zip_safe=False)
