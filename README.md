# What is this?
This is a program that reads text from your screen. This program can read notes, subtitles, and more. It can also translate text before it is spoken. 
Here are some examples of what it looks and sounds like :

![Знімок екрана 2024-04-18 124413](https://github.com/Santabot123/ozvuchator/assets/56690519/1f905e47-afa7-4d1d-962b-c535d5eb15ec)

## Short demo
https://github.com/Santabot123/ozvuchator/assets/56690519/2a3650de-e093-48bf-8d13-e77204468ee6

## Demonstration of how the manual mode works
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/XBd4vCp3Rq4/0.jpg)](https://www.youtube.com/watch?v=XBd4vCp3Rq4)

## Demonstration of how the auto mode works
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/NWhn4AtTWNg/0.jpg)](https://www.youtube.com/watch?v=NWhn4AtTWNg)


# System Requirements.
- OS: Windows 10/11 64-bit
- internet connection

# Installation.
 There are 2 ways:
 1. exe file:
    - Choose a version (if you do not know what to choose, read the Versions chapter below) 
    - Download and unzip one of these archives : [Pytesseract](https://drive.google.com/file/d/1iWj3KTTYqXC0rV6dT2TqtNJLjFBBkGg_/view?usp=sharing), [EasyOCR](https://drive.google.com/file/d/1PR-r4gNP2HIV87lmcxK1qHnZHBag28hU/view?usp=sharing), [EasyOCR+Pytesseract](https://drive.google.com/file/d/1tkqsdPiK4eu4iOYi4diJva3RawpzL7EM/view?usp=sharing).
    - Then to run, go to `exe.win-amd64-3.10` and run `ozvuchator.exe`
 3. jupyter notebook:
    - Install [Anaconda](https://www.anaconda.com/installation-success?source=installer)
    - Create a new virtual environment in Anaconda
    - Download this repository and unzip it, or use `git clone https://github.com/Santabot123/ozvuchator`
    - Choose a version (if you do not know what to choose, read the Versions chapter below) 
    -  If you want to use Pyteeseract, download [Tesseract 5.3.3](https://github.com/UB-Mannheim/tesseract/wiki) (maybe when you read this there is a newer version, but I don't know if it will work as well as 5.3.3, so it is recommended to use 5.3.3)   and install it in the Tesseract-OCR folder located in the folder where you downloaded this repository.
    - Open a jupyter notebook, find the location where you unzipped this repository, and open `ozvuchator.ipynb` that corresponds to your version.
    - Press ⏩.  
    - Wait (the first run will be long because you need to download all the necessary libraries)

# Versions
There are three versions: Pytesseract, EasyOCR, and Pytesseract+EasyOCR. Here is some information about them so you can decide which one you need:
- Pytesseract - uses Tesseract, choose this version if the text you want to listen to: contrasts with the background, looks like a scanned document or screenshot. Make sure that the area of the screen you choose does not include interface elements/window borders/icons, as they can be incorrectly recognized as orthographic characters. It also takes up the least amount of disk space. Examples of text that Pytesseract handles well:
 ![Знімок екрана 2024-05-02 191519](https://github.com/Santabot123/ozvuchator/assets/56690519/9bf8da3b-6e42-4b44-974b-353b9caafa91)
![Знімок екрана 2024-05-02 192450](https://github.com/Santabot123/ozvuchator/assets/56690519/ca7f454c-750b-41e4-8d38-376f8f723980)
![Знімок екрана 2024-04-27 182902](https://github.com/Santabot123/ozvuchator/assets/56690519/6b93a6a2-fce8-4db0-b5f4-1c00ddd67ab3)

- EasyOCR - сhoose this version of your text if it doesn't contrast well with the background, is in a photo, or has some distortion due to perspective. EasyOCR can also use the Nvidia GPU instead of the CPU. Examples of text that EasyOCR handles well:
- Pytesseract+EasyOCR is essentially two previous versions combined into one, which gives you the flexibility to choose which method to use.


# Usage

There are 2 modes of use: manual and auto.
- Manual mode - you press the activation button (F2 by default) and select an area, after which the text will be spoken once.
- Auto mode - you select an area of the screen and when a new text appears in this area, it will be automatically spoken.

If you have already clicked Run and then want to change the settings, you need to close Ozvuchator and open it again.



# Note
- To read what a particular parameter does, hover over it with the cursor
- If you have an Nvidia video card installed, it will be used by default, in all other cases the CPU will be used
- The longer the sentence, the longer the delay before the sound is played.
- If you are going to use this program during the game, you may need to switch the game to windowed/borderless mode.
- Text recognition will work well **only with printed letters** .
<br>

# List of supported languages for EasyOCR method:
- afrikaans : 'af'<br>
- albanian : 'sq'<br>
- arabic : 'ar'<br>
- bengali : 'bn'<br>
- bosnian : 'bs'<br>
- bulgarian : 'bg'<br>
- croatian : 'hr'<br>
- czech : 'cs'<br>
- danish : 'da'<br>
- dutch : 'nl'<br>
- english : 'en'<br>
- estonian : 'et'<br>
- filipino : 'tl'<br>
- french : 'fr'<br>
- german : 'de'<br>
- hindi : 'hi'<br>
- hungarian : 'hu'<br>
- icelandic : 'is'<br>
- indonesian : 'id'<br>
- italian : 'it'<br>
- japanese : 'ja'<br>
- kannada : 'kn'<br>
- korean : 'ko'<br>
- latin : 'la'<br>
- latvian : 'lv'<br>
- malay : 'ms'<br>
- marathi : 'mr'<br>
- nepali : 'ne'<br>
- norwegian : 'no'<br>
- polish : 'pl'<br>
- portuguese : 'pt'<br>
- romanian : 'ro'<br>
- slovak : 'sk'<br>
- spanish : 'es'<br>
- swahili : 'sw'<br>
- swedish : 'sv'<br>
- tamil : 'ta'<br>
- telugu : 'te'<br>
- thai : 'th'<br>
- turkish : 'tr'<br>
- ukrainian : 'uk'<br>
- urdu : 'ur'<br>
- vietnamese : 'vi'<br>

# List of supported languages for Pytesseract method:
- Afrikaans : af
- Albanian : sq
- Arabic : ar
- Bengali : bn
- Bosnian : bs
- Bulgarian : bg
- Burmese : my
- Catalan : ca
- Croatian : hr
- Czech : cs
- Danish : da
- Dutch : nl
- English : en
- Estonian : et
- Finnish : fi
- French : fr
- German : de
- Gujarati : gu
- Hindi : hi
- Hungarian : hu
- Icelandic : is
- Indonesian : id
- Italian : it
- Japanese : ja 
- Kannada : kn
- Khmer : km
- Korean : ko
- Latin : la
- Latvian : lv
- Malay (macrolanguage) : ms
- Malayalam : ml
- Marathi : mr
- Modern Greek (1453-) : el
- Nepali (macrolanguage) : ne
- Norwegian : no
- Polish : pl
- Portuguese : pt
- Romanian : ro
- Serbian : sr
- Sinhala : si
- Slovak : sk
- Spanish : es
- Sundanese : su
- Swahili (macrolanguage) : sw
- Swedish : sv
- Tagalog : tl
- Tamil : ta
- Telugu : te
- Thai : th
- Turkish : tr
- Ukrainian : uk
- Urdu : ur
- Vietnamese : vi

