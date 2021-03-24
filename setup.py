from setuptools import setup, find_packages

setup(
      name='friendly-guacamole',
      version='0.0.1',
      description='Microbiome-friendly scikit-learn style data'
                  'preprocessing and other utilities.',
      long_description=open('README.md').read(),
      license='LICENSE',
      author='George Armstrong',
      author_email='garmstro@eng.ucsd.edu',
      url='https://github.com/gwarmstrong/friendly-guacamole',
      packages=find_packages(),
      install_requires=[
            'numpy',
            'scikit-learn>=0.24',
            'biom-format',
            'matplotlib',
            'scipy',
            'pandas',
            'unifrac',
      ],
      extras_require={
            'test': [
                  'nose',
                  'flake8',
            ]
      }
)
