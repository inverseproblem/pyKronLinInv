
from setuptools import setup

setup(name='kronlininv',
      version='0.1',
      python_requires='>=3',
      description='Kronecker-product based linear inversion',
      url='http://github.com/inverseproblem/pykronlininv.git',
      author='Andrea Zunino',
      author_email='inverseproblem@users.noreply.github.com',
      license='GPL3',
      packages=['kronlininv'],
      install_requires=[
          'numpy','scipy','numba','collections'
      ],
      zip_safe=False)
