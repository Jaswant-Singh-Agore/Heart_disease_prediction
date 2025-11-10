from setuptools import find_packages, setup

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path):
    """
    Returns a list of requirements from a requirements.txt file.
    """
    with open(file_path, encoding="utf-8") as file_obj:
        requirements = [req.strip() for req in file_obj.readlines()]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='Heart_disease_classification',
    version='0.1',
    author='Jaswant',
    description='A heart disease classification package using TensorFlow',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    python_requires='>=3.9',
)
