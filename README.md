# What is this?
This is a program that reads text from your screen.
# System Requirements.
- OS: Windows 10/11 64-bit
- ffmpeg installed
- internet connection

# Installation.
 There are 2 ways:
 1. exe file:
    - Install ffmpeg and add it to the PATH of your system (there are a lot of guides on the Internet how to do this)
    - Download and unzip this [archive](https://drive.google.com/file/d/10EWmP2It9Iy3QwZwAw3iV4Leeu4rjXXE/view?usp=drive_link).
    - Then to run, go to `exe.win-amd64-3.9` and run `ozvuchator.exe`
 2. jupyter notebook:
    - Install ffmpeg and add it to the PATH of your system (there are a lot of guides on the Internet how to do this)
    - Install [Anaconda](https://www.anaconda.com/installation-success?source=installer)
    - Create a new virtual environment in Anaconda
    - Download this repository and unzip it, or use `git clone https://github.com/Santabot123/ozvuchator`
    - Open jupyter notebook, find the location where you unziped this repository and open `ozvuchator.ipynb`.
    - Press ⏩.  
    - Wait (the first run will be long because you need to download all the necessary libraries)

# Usage
There are 2 modes of use: manual and auto.
- Manual mode - you press the activation button (F2 by default) and select an area, after which the text will be spoken once.
- Auto mode - you select an area of the screen and when a new text appears in this area, it will be automatically spoken.



# Note
- To read what a particular parameter does, hover over it with the cursor
- If you have an Nvidia video card installed, it will be used by default, in all other cases the CPU will be used
- The longer the text is, the longer the delay before the voice is generated
- If you are going to use this program during the game, you may need to switch the game to windowed/borderless mode



