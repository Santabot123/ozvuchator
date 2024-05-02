from cx_Freeze import setup, Executable
import sys
sys.setrecursionlimit(10000)

# Dependencies are automatically detected, but it might need
# fine tuning.
build_options = {'packages': ["PySide6"], 'excludes': [], 'include_files': ['settings.ini','design.ui']}

# import sys
# base = 'Win32GUI' if sys.platform=='win32' else None
#
# executables = [
#     Executable('ozvuchator.py', base=base, target_name = 'Ozvuchator')
# ]

base = 'console'

executables = [
    Executable('ozvuchator.py', base=base)
]

setup(name='ozvuchator',
      version = '1',
      description = 'A program that will read text from your screen',
      options = {'build_exe': build_options},
      executables = executables)
