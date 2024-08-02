from setuptools import setup, find_packages

setup(
    name="emocon",
    version="0.1.0",
    author="eliird",
    author_email="irdali1996@gmail.com",
    description="Emotion Detection From Videos Using Contrastive Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/eliird/emocon",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.18.0',
        'opencv-python',
        'torchaudio',
        'torch',
        'torchvision',
        'transformers',
        'librosa', 'moviepy', 'facenet_pytorch',
    ],
    entry_points={
        'console_scripts': [
        ],
    },
    include_package_data=True,
    package_data={

    },
)
