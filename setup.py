from setuptools import setup

setup(name = "FFPE", 
      url = "https://github.com/ksjiang/FFPE", 
      author = "Kyle Jiang", 
      author_email = "siyu.jiang81@gmail.com", 
      packages = ["FFPE"], 
      install_requires = ["numpy", "pandas"], 
      version = "0.0.1", 
      license = "MIT", 
      description = "A collection of parsers for fast file parsing of electrochemical data.", 
      long_description = open("./README.md", 'r').read())