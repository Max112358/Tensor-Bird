from setuptools import setup, find_packages

setup(
    name="flappy-bird-ai",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pygame>=2.5.0',
        'neat-python>=0.92',
    ],
    author="Your Name",
    description="A Flappy Bird AI implementation using NEAT",
    python_requires='>=3.6',
)
#use
# pip install . 
#to install. dont forget the dot