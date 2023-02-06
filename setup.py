from os import path

from setuptools import setup


#import weightwatcher as ww
#try:
#    import pypandoc
#    readme = pypandoc.convert('README.md', 'rst')
#    readme = readme.replace("\r","")
#except OSError as e:
#    # pypandoc failed, use the short description as long description
#    readme = ww.__description__
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    readme = f.read()

class _metadata(object):
    def __init__(self):
        """
        Read the __init__.py vars into attrs
        """
        for line in open('weightwatcher/__init__.py'):
            if line.startswith('__') and "__all__" not in line:
                setattr(self,
                        line.split()[0],
                        eval(line.split('=')[1]))

ww = _metadata()

setup(
    name = ww.__name__,
    version = ww.__version__,
    url = ww.__url__,
    project_urls={
        "Documentation": "https://calculationconsulting.com/",
        "Code": "https://github.com/calculatedcontent/weightwatcher",
        "Issue tracker": "https://github.com/calculatedcontent/weightwatcher/issues",
    },
    license = ww.__license__,
    author = ww.__author__,
    author_email = ww.__email__,
    maintainer = ww.__author__,
    maintainer_email = ww.__email__,
    description = ww.__description__,
    long_description = readme,
    long_description_content_type="text/markdown",
    packages = ["weightwatcher"],
    include_package_data = True,
    test_suite = 'tests',
    python_requires = ">= 3.3",
    install_requires = ['numpy',
                        'pandas',
                        'matplotlib',
                        'matplotlib-inline',
                        'powerlaw',
                        'scikit-learn',
                        'tqdm'],
    entry_points = '''
        [console_scripts]
        weightwatcher=weightwatcher:main
    ''',
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
    ],
    keywords = "Deep Learning Keras Tensorflow pytorch Deep Learning DNN Neural Networks",
)
