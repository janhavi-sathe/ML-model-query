from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setup(name="aicoach",
      version="0.1.0",
      author="Sangwon Seo",
      author_email="sangwon.seo@rice.edu",
      description="Code for AI Coach experiments and results",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(include=["aicoach"]),
      python_requires='>=3.8',
      install_requires=[
          'numpy', 'tqdm', 'scipy', 'sparse', 'torch', 'termcolor',
          'tensorboard', "stable-baselines3", 'matplotlib', 'click',
          'opencv-python', 'seaborn'
      ])
