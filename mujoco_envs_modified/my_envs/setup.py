from setuptools import setup, find_packages
import sys, os.path

extras = {
  'mujoco': ['mujoco_py>=1.50', 'imageio'],
}

all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras['all'] = all_deps

setup(name='my_envs',
      version='0.0.1',
      description='Custom environments.',
      install_requires=[
          'numpy>=1.10.4',
      ],
      extras_require=extras,
      package_data={'my_envs': [
        'envs/assets/*.xml'
        ]
      },
)