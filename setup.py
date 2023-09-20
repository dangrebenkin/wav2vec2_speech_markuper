from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess


class PostInstallCommand(install):
    def run(self):
        subprocess.call(['pip', 'install', '-r', 'requirements.txt'])
        install.run(self)


setup(
    name='wav2vec2_speech_markuper',
    version='0.1',
    description='Automatic generation of speech dataset markup with use of Wav2Vec2 ASR models',
    url='https://github.com/dangrebenkin/wav2vec2_speech_markuper',
    author='Daniel Grebenkin',
    author_email='d.grebenkin@g.nsu.ru',
    license='Apache License Version 2.0',
    keywords=['speech-recognition', 'speech-to-text', 'audio-segmentation', 'forced-alignment', 'wav2vec2'],
    packages=find_packages(),
    python_requires=r'>=3.8.0',
    cmdclass={
        'install': PostInstallCommand,
    },
)
