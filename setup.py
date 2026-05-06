from setuptools import find_packages, setup

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> list:
    '''Reads the requirements from a file and returns them as a list.'''
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)


    return requirements

setup(
    name="churn_prediction",
    version="0.1",
    author="Hritik",
    author_email='ritikmodanwal101@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)