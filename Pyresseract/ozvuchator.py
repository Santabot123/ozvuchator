#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install dxcam
# !pip install matplotlib
# !pip install gTTS
# !pip install soundfile
# !conda install -c main ffmpeg -y
# !pip install -U deep-translator
# !pip install pydub
# !pip install PySide6
# !pip install pyaudio
# !pip install cx-Freeze
# !pip install iso639-lang
# !pip install nltk
# !conda install -c anaconda git -y
# !pip install -U git+https://github.com/madmaze/pytesseract.git


# In[2]:


#TO DO LIST
'''
- розібратись чому працює лише англійська✅
-use download tessdata to download other languages ✅
- solve problem when textt is added by partitions❌
-maybe solve global variables problem
-test every function ✅
-реалізувати укр версію
-розповіти комусь про існування такої пиздатої програми
'''


# In[3]:


print('Wait...')


# In[4]:


from gtts import gTTS
import gtts
# from tempfile import TemporaryFile
import io
from iso639 import Lang
# import soundfile as sf
# import easyocr
import pytesseract
import time
import cv2 as cv
import numpy as np
import dxcam
from IPython.display import display, clear_output,Audio
from configparser import ConfigParser
import win32api
from win32api import GetSystemMetrics
from matplotlib import pyplot as plt
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from pydub.playback import play
# import re
import string
import pathlib
from difflib import SequenceMatcher
import threading
from nltk import tokenize


# In[ ]:


from nltk import download
download('punkt',quiet=True)


# In[ ]:


class iso_converter():
    '''
    This is a simple class for converting language codes according to ISO 639
    '''
    def convert_to_pt1(item):
        try:
            return Lang(item).asdict()['pt1']
        except: return None
        

    def convert_to_pt2t(item):
        try:
            return Lang(item).asdict()['pt2t']
        except: return None
        


# In[ ]:


pytesseract_list= [iso_converter.convert_to_pt1(item) for item in [ 'afr','amh','ara','asm','aze','aze_cyrl','bel','ben','bod','bos','bre','bul','cat','ceb','ces','chi_sim','chi_sim_vert','chi_tra','chi_tra_vert','chr','cos','cym','dan','dan_frak','deu','deu_frak','deu_latf','div','dzo','ell','eng','enm','epo','equ','est','eus','fao','fas','fil','fin','fra','frm','fry','gla','gle','glg','grc','guj','hat','heb','hin','hrv','hun','hye','iku','ind','isl','ita','ita_old','jav','jpn','jpn_vert','kan','kat','kat_old','kaz','khm','kir','kmr','kor','kor_vert','lao','lat','lav','lit','ltz','mal','mar','mkd','mlt','mon','mri','msa','mya','nep','nld','nor','oci','ori','osd','pan','pol','por','pus','que','ron','san','sin','slk','slk_frak','slv','snd','spa','spa_old','sqi','srp','srp_latn','sun','swa','swe','syr','tam','tat','tel','tgk','tgl','tha','tir','ton','tur','uig','ukr','urd','uzb','uzb_cyrl','vie','yid','yor']]


# In[ ]:


translator_set=set(GoogleTranslator().get_supported_languages(as_dict=True).values())
# easyocr_set=set(['abq','ady','af','ang','ar','as','ava','az','be','bg','bh','bho','bn','bs','ch_sim','ch_tra','che','cs','cy','da','dar','de','en','es','et','fa','fr','ga','gom','hi','hr','hu','id','inh','is','it','ja','kbd','kn','ko','ku','la','lbe','lez','lt','lv','mah','mai','mi','mn','mr','ms','mt','ne','new','nl','no','oc','pi','pl','pt','ro','rs_cyrillic','rs_latin','sck','sk','sl','sq','sv','sw','ta','tab','te','th','tjk','tl','tr','ug','uk','ur','uz','vi',])
pytesseract_set=set(pytesseract_list)


speaker_set=set(gtts.lang.tts_langs().keys())
# easyocr_set.intersection(translator_set,speaker_set)
# pytesseract_set.intersection(translator_set,speaker_set)
# len(pytesseract_set.intersection(translator_set,speaker_set))


# In[ ]:


current_directory=pathlib.Path().resolve()
pytesseract.pytesseract.tesseract_cmd = fr'.\Tesseract-OCR\tesseract.exe'
import os
os.environ['TESSDATA_PREFIX'] = r'.\Tesseract-OCR\tessdata'


# In[ ]:


settings_file='settings.ini'
config = ConfigParser()
config.read(settings_file)
# list(config['settings'])


# In[ ]:


SPEAK_LANGUAGE=config['settings']['SPEAK_LANGUAGE']


TRANSLATE =eval(config['settings']['translate'])
TRANSLATE_FROM=config['settings']['TRANSLATE_FROM']

DETECTION_LANGUAGES = [SPEAK_LANGUAGE]

if TRANSLATE:
    DETECTION_LANGUAGES =[TRANSLATE_FROM]

TRANSLATE_TO=SPEAK_LANGUAGE

SPEED_UP=int(config['settings']['SPEED_UP'])

FPS=False

RESIZE=eval(config['settings']['resize'])
RESIZE_SCALE=float(config['settings']['RESIZE_SCALE'])

THRESHOLD=int(config['settings']['threshold'])
REDUCE_NOISE =eval(config['settings']['REDUCE_NOISE'])
OUTLINED= eval(config['settings']['OUTLINED'])

LEFT=int(config['settings']['LEFT']) 
TOP=int(config['settings']['TOP'])
RIGHT=int(config['settings']['RIGHT']) 
BOTTOM=int(config['settings']['BOTTOM']) 

AUTO_MODE=eval(config['settings']['auto_mode'])
PAUSE_KEY=int(config['settings']['PAUSE_KEY'], 16)
MANUAL_MODE=eval(config['settings']['manual_mode'])
MANUAL_MODE_KEY=int(config['settings']['MANUAL_MODE_KEY'], 16)

PAUSE_KEY_status=False

IGNORE_WORDS=config['settings']['ignore_words'].split(',')


# # Back end 

# In[ ]:


dw = win32api.GetSystemMetrics(0) #screen width in pixels
dh = win32api.GetSystemMetrics(1) #screen height in pixels


# In[ ]:


def select_area():
    '''
    This function allows the user to select the area of the screen where text recognition will be performed.

    Returns coordinates of the box in format: int:LEFT, int:TOP, int:RIGHT, int:BOTTOM
                                                
    '''
    def draw_rectanlge(event, x, y, flags, param):
        global ix,iy,drawing,overlay ,img
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y
        elif event == cv.EVENT_MOUSEMOVE:
            if drawing == True:
                cv.rectangle(img, (ix, iy), (x, y), (41, 215, 162), -1)
                cv.putText(img, 'PRESS ANY KEY TO SELECT THIS AREA', (ix, iy-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (55,46,252), 2)
                img=cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        elif event == cv.EVENT_LBUTTONUP:
            global LEFT,TOP,RIGHT,BOTTOM
            
            drawing = False
            if ix<x:
                LEFT=int(ix)
                RIGHT=int(x)
            else:
                LEFT=int(x)
                RIGHT=int(ix)
            if iy<y:
                TOP=int(iy)
                BOTTOM=int(y)
            else:
                TOP=int(y)
                BOTTOM=int(iy)

    global drawing,ix,iy,overlay,img
    drawing = False
    ix,iy = -1,-1

    camera = dxcam.create(output_color="BGR")

    img = camera.grab()

    
    img=cv.rectangle(img, (0, 0), (dw, dh), (0,0,255), 10)
    img=cv.putText(img, 'SELECT AN AREA', (int(dw*0.4), 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
    
    overlay = img.copy()
    alpha = 0.3
    cv.namedWindow('SELECT AREA', cv.WINDOW_NORMAL) 
    cv.setWindowProperty('SELECT AREA', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.setWindowProperty('SELECT AREA', cv.WND_PROP_TOPMOST, 1)
    cv.setMouseCallback('SELECT AREA', draw_rectanlge)
    
    while(1):
        cv.imshow('SELECT AREA',img)
        if cv.waitKey(20) >-1:
            break
        
    cv.destroyAllWindows()
    del camera

    return LEFT, TOP, RIGHT, BOTTOM

# select_area()


# In[ ]:


def show_area():
    '''
    This function will create only one screenshot and show the area where text detection will be performed
    '''
    camera = dxcam.create(output_color="RGB")
    frame = camera.grab((0,0,dw,dh))
    image_with_box=cv.rectangle(frame,[LEFT,TOP],[RIGHT,BOTTOM],[76,130,250], 8)
        
    plt.figure(figsize=(10,10))
    plt.imshow(image_with_box)
    
    print("YOUR ZONE WHERE THE TEXT WILL BE DETECTED IS: ")
    print('LEFT='+str(LEFT),'TOP='+str(TOP),'RIGHT='+str(RIGHT),'BOTTOM='+str(BOTTOM), sep=' pixels \n', end=' pixels \n')
    del camera

# show_area()


# In[ ]:


class Preprocessing():
    '''
    This class contains functions for preprocessing an image before text recognition
    '''
    def resize(screenshot):
        '''
        Resize the image to help speed up text recognition, and note that Tesseract cannot recognize too large fonts

        Args: screenshot: An image in numpy.ndarray, python list or pil formats.
        
        Returns: resized image in numpy.ndarray
        '''
        return cv.resize(screenshot, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE, interpolation = cv.INTER_AREA)
        
    def apply_threshold(screenshot):
        '''
        Turns a gray image into a black and white one
        Args: screenshot: An image in numpy.ndarray, python list or pil formats.
        Returns:  black and white image in numpy.ndarray
        '''
        global THRESHOLD
        thresh, screenshot = cv.threshold(screenshot,THRESHOLD,255,cv.THRESH_BINARY)
        return screenshot
        
    def reduce_noise(screenshot):
        '''
        Reduces the amount of noise in the image. Recommended for use only on images of poorly scanned documents
        Args: screenshot: An image in numpy.ndarray, python list or pil formats.
        Returns: image with redused noise in numpy.ndarray format
        '''
        kernel = np.ones((1, 1), np.uint8)
        screenshot = cv.dilate(screenshot, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        screenshot = cv.erode(screenshot, kernel, iterations=1)
        screenshot = cv.morphologyEx(screenshot, cv.MORPH_CLOSE, kernel)
        screenshot = cv.medianBlur(screenshot, 3)
        return screenshot
        
    def fill_background(screenshot):
        '''
        Pytesseract does not recognize outlined text by default. 
        The easiest way to deal with outlined text in this case is to fill the background of the image with the same color as the outline.
        Args: screenshot: An image in numpy.ndarray, python list or pil formats.
        Returns: image with a white background in numpy.ndarray format
        '''
        return cv.bitwise_not(cv.floodFill(screenshot, None,(1,1), (0, 0, 0), (10, 10, 10), (20, 20, 20))[1])



class Postrocessing():
    '''
    This class contains functions for postrocessing a text after text recognition
    '''
    def split_into_sentences (list_of_strings):
        '''
        Combines all text into one string and splits it into sentences
        Args: list of strings - Pytesseract will return only 1 string, while easyOSR will return a list of strings
        Return: list of sentences 
        '''
        long_string = ' '.join(list_of_strings)
        return tokenize.sent_tokenize(long_string)
        
    def remove_short_strings(list_of_strings):
        '''
        These function will remove only strings that longer than 2 characters. This function only makes sense when using easyOCR
        Args: list of strings
        Return: list of strings that longer than  2 characters.
        '''
        return [string for string in list_of_strings if len(string) > 2]
        
    def clean_result(list_of_strings):
        '''
        This function replaces unwanted text with spaces.
        This is useful when you need to read subtitles, for example, and each line starts with “Jon: line”/“Adam: replica”
        and you don't want to hear their names every time.
        Args: list of strings
        Return: list of strings where in which line, words from IGNORE_WORDS were replaced to spaces.
        ''' 
        for i, string in enumerate(list_of_strings):
            for word in IGNORE_WORDS:
                string=' '.join(string.splitlines())
                string=string.replace(word, ' ')
            list_of_strings[i]=string
        return list_of_strings

    def translator(list_of_strings):
        return [GoogleTranslator(source=TRANSLATE_FROM, target=TRANSLATE_TO).translate(string) for string in list_of_strings]

    def find_new_text(previous_string, new_string):

        las_space_index = previous_string.rfind(' ')
        if new_string.startswith(previous_string[:-2]):
            return new_string[las_space_index+1:]
        return new_string


# In[ ]:


def speak(my_text):
    '''
    This function will voice over text
    '''
    with io.BytesIO() as f:
        gTTS(text=my_text, lang=SPEAK_LANGUAGE).write_to_fp(f)
        f.seek(0)
        
        sound = AudioSegment.from_file(io.BytesIO(f.read()))
        sound = sound.speedup(1.1+(SPEED_UP/100), 150, 25)
        # audio_len=len(sound)/1000
        play(sound[0:len(sound)-200])


# In[ ]:


def download_tessdata(pt2t_lang_code):
    '''
    Tesseract requires a .traindata file for each language. So this function simply downloads these files from the official gitHub.
    Args: pt2t_lang_code - string that contains pt2t language code from iso 639
    '''
    from urllib.request import urlretrieve
    
    current_directory=pathlib.Path().resolve()
    print(f"Downloading tessdata for {Lang(pt2t_lang_code).asdict()['name']}. Wait...")
    urlretrieve(f'https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/{pt2t_lang_code}.traineddata', fr'{current_directory}\Tesseract-OCR\tessdata\{pt2t_lang_code}.traineddata' )
    print(f"{Lang(pt2t_lang_code).asdict()['name']} is downloaded")
    print("List of downloaded languages:",pytesseract.get_languages(config=''))


# download_tessdata('ukr')


# In[ ]:


def toggle(variable):
    time.sleep(0.5)
    return not variable

def compare_strings(string1, string2):
    '''
    When using auto mode, the program needs to know what text was before, otherwise it will endlessly read the same text until a new one appears.
    If the new text is more than 95% similar to the previous one, it returns True, otherwise it returns False 
    (this 95% limit exists because sometimes the model incorrectly recognizes “,” “.”  “:” “;” and other punctuation) 
    Args:
        string1: new text string
        string2: old text string
    Returns: True or False
    '''
    string1 = string1.lower()
    string2 = string2.lower()

    translator = str.maketrans('', '', string.punctuation)
    string1 = string1.translate(translator)
    string2 = string2.translate(translator)

    if SequenceMatcher(a=string1,b=string2).ratio()>0.95:
        return True
    else:
        return False


# In[ ]:


def auto_mode(LEFT, TOP, RIGHT, BOTTOM):
    '''
    This function implements an infinite loop that does this:
    Check whether the pause button was pressed, if not -> Take a screenshot -> Preprocess the image ->
    -> Recognize text -> Postprocess the text -> Voice the text -> Repeat
    Args:
        LEFT: int \ left border of the selected area
        TOP: int \ upper border of the selected area
        RIGHT: int \  right border of the selected area
        BOTTOM: int \ lower border of the selected area
    '''
    camera = dxcam.create(output_color="GRAY")
    camera.start(region=(LEFT, TOP, RIGHT, BOTTOM))
    # reader = easyocr.Reader(DETECTION_LANGUAGES)

    frames_counter=0
    all_time=0
    DETECTED_TEXT=['']
    PAUSE_KEY_status=False
    
    while True :
        if win32api.GetAsyncKeyState(PAUSE_KEY)<0:
            global stop_thread
            if stop_thread:
                break
                
            PAUSE_KEY_status=toggle(PAUSE_KEY_status)
            display('PAUSE KEY PRESSED')
        if not PAUSE_KEY_status:
             
            clear_output(wait=True)
            start= time.time()
        
            screenshot = camera.get_latest_frame()
            if RESIZE:
                screenshot = Preprocessing.resize(screenshot)
                
            screenshot=Preprocessing.apply_threshold(screenshot)

            if REDUCE_NOISE:
                screenshot=Preprocessing.reduce_noise(screenshot)

            if OUTLINED:
                screenshot=Preprocessing.fill_background(screenshot)
                      
            # plt.imshow(screenshot)
            # plt.show()
            
            result = [pytesseract.image_to_string(screenshot,lang=iso_converter.convert_to_pt2t (DETECTION_LANGUAGES[0]))]
            # result = reader.readtext(screenshot,detail=0,paragraph=True)
            result=Postrocessing.remove_short_strings(result)
            # result= [' '.join(result)]
            result=Postrocessing.clean_result(result)
    
            if TRANSLATE:
                result= Postrocessing.translator(result)
            
            if compare_strings(' '.join(result),' '.join(DETECTED_TEXT)):
                pass
            else:
                result =[Postrocessing.find_new_text(' '.join(DETECTED_TEXT), ' '.join(result))]
                result= Postrocessing.split_into_sentences(result)
                
                DETECTED_TEXT=result                
                display(result)
                
                for r in result:
                    if r=='': continue 
                    speak_thread = threading.Thread(target=speak, args=(r,), daemon=True)
                    speak_thread.start()
                    while speak_thread.is_alive():
                        if win32api.GetAsyncKeyState(PAUSE_KEY)<0:
                            PAUSE_KEY_status = toggle(PAUSE_KEY_status)
                            display('PAUSE')
                            speak_thread.join()
                            time.sleep(0.5)
                            break
                    if PAUSE_KEY_status == True:
                        break
                    speak_thread.join()
                
    
            if FPS:
                
                display('FPS: '+str(1/(time.time()-start+0.00015)) )
                frames_counter+=1
                all_time+=1/(time.time()-start+0.00015)
                display('Avg FPS: '+str(all_time/frames_counter)) 
        

    del camera


# In[ ]:


# auto_mode(*select_area())


# In[ ]:


def manual_mode():
    '''
    This function implements an infinite loop that does this:
    Check whether the MANUAL_MODE_KEY button was pressed, if yes -> Call select_area() function -> Take a screenshot of the selected area ->
    -> Preprocess the image -> Recognize text -> Postprocess the text -> Voice the text -> Repeat 
    '''
    MANUAL_MODE_KEY_status=False
    # reader = easyocr.Reader(DETECTION_LANGUAGES)
    
    while True :

        if win32api.GetAsyncKeyState(MANUAL_MODE_KEY)<0:
            global stop_thread
            if stop_thread:
                break
            
            MANUAL_MODE_KEY_status=toggle(MANUAL_MODE_KEY_status)
            display('MANUAL MODE KEY PRESSED')

 
        if MANUAL_MODE_KEY_status:
            region=select_area()
            
            camera = dxcam.create(output_color="GRAY")
            screenshot = camera.grab(region=region)

            if RESIZE:
                screenshot = Preprocessing.resize(screenshot)
                
            screenshot=Preprocessing.apply_threshold(screenshot)

            if REDUCE_NOISE:
                screenshot=Preprocessing.reduce_noise(screenshot)

            if OUTLINED:
                screenshot=Preprocessing.fill_background(screenshot)
                
            # plt.imshow(screenshot)
            # plt.show()
            
            result = [pytesseract.image_to_string(screenshot,lang=iso_converter.convert_to_pt2t (DETECTION_LANGUAGES[0]))]
            # result = reader.readtext(screenshot,detail=0,paragraph=True)
            result=Postrocessing.remove_short_strings(result)
            # result= [' '.join(result)]
            result=Postrocessing.clean_result(result)
            if TRANSLATE:
                result= Postrocessing.translator(result)
                
            result= Postrocessing.split_into_sentences (result)
                
            display(result)
            
            for r in result:
                    if r=='': continue 
                    speak(r)

    
            MANUAL_MODE_KEY_status=toggle(MANUAL_MODE_KEY_status)
            
            del camera

            clear_output(wait=True)

# manual_mode()   


# # Interface 

# In[ ]:


import sys
import os
import ctypes
from PySide6.QtWidgets import QWidget, QApplication, QButtonGroup
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import  Qt, QRunnable, Slot, QThreadPool,QProcess


# In[ ]:


stop_thread = False


class mywidget(QWidget):
    def __init__(self):
        super().__init__()
        self.ui=loader.load('design.ui',None)
        # CHECKBOXES
        self.ui.TRANSLATE_checkBox.setChecked(eval(config['settings']['translate']))
        self.ui.RESIZE_checkBox.setChecked(eval(config['settings']['RESIZE']))
        self.ui.OUTLINED_checkBox.setChecked(eval(config['settings']['OUTLINED']))
        self.ui.REDUCE_NOISE_checkBox.setChecked(eval(config['settings']['REDUCE_NOISE']))

        # RADIOBUTTONS
        self.ui.AUTO_MODE_radioButton.setChecked(eval(config['settings']['AUTO_MODE']))
        self.ui.MANUAL_MODE_radioButton.setChecked(eval(config['settings']['MANUAL_MODE']))
        self.ui.AUTO_MODE_radioButton.clicked.connect(self.switch1)
        self.ui.MANUAL_MODE_radioButton.clicked.connect(self.switch2)

        # SPINBOX
        self.ui.THRESHOLD_spinBox.setValue(int(config['settings']['threshold']))

        # LINE_EDITS
        self.ui.RESIZE_SCALE_lineEdit.setText(config['settings']['RESIZE_SCALE'])
        self.ui.SPEED_UP_lineEdit.setText(config['settings']['SPEED_UP'])
        self.ui.IGNORE_WORDS_lineEdit.setText(config['settings']['IGNORE_WORDS'])
        self.ui.LEFT_lineEdit.setText(config['settings']['LEFT'])
        self.ui.TOP_lineEdit.setText(config['settings']['TOP'])
        self.ui.RIGHT_lineEdit.setText(config['settings']['RIGHT'])
        self.ui.BOTTOM_lineEdit.setText(config['settings']['BOTTOM'])
        self.ui.PAUSE_KEY_lineEdit.setText(config['settings']['PAUSE_KEY'])
        self.ui.MANUAL_MODE_KEY_lineEdit.setText(config['settings']['MANUAL_MODE_KEY'])

        # COMBO_BOXES
        self.ui.SPEAK_LANGUAGE_comboBox.clear()
        self.ui.SPEAK_LANGUAGE_comboBox.addItems(sorted(pytesseract_set.intersection(translator_set,speaker_set)))
        self.ui.SPEAK_LANGUAGE_comboBox.setCurrentText(config['settings']['SPEAK_LANGUAGE'])
        self.ui.TRANSLATE_FROM_comboBox.clear()
        self.ui.TRANSLATE_FROM_comboBox.addItems(sorted(pytesseract_set.intersection(translator_set,speaker_set)))
        self.ui.TRANSLATE_FROM_comboBox.setCurrentText(config['settings']['TRANSLATE_FROM'])

        # BUTTONS
        self.ui.PREVIEW_pushButton.clicked.connect(self.show_preview)
        self.ui.Select_area_pushButton.clicked.connect(self.area)
        self.ui.APPLY_pushButton.clicked.connect(self.apply_settings)
        self.ui.RUN_pushButton.clicked.connect(self.run)

        app.aboutToQuit.connect(self.closeEvent)

    def show_preview(self):
        self.apply_settings()
        self.refresh_settings()
        
        THRESHOLD=int(config['settings']['threshold'])
        REDUCE_NOISE =eval(config['settings']['REDUCE_NOISE'])
        OUTLINED= eval(config['settings']['OUTLINED'])
        
        LEFT, TOP, RIGHT, BOTTOM = select_area()
        
        camera = dxcam.create(output_color="GRAY", region=(LEFT, TOP, RIGHT, BOTTOM))
        screenshot = camera.grab()
        screenshot=Preprocessing.apply_threshold(screenshot)

        if REDUCE_NOISE:
            screenshot=Preprocessing.reduce_noise(screenshot)
        
        if OUTLINED:
            screenshot=Preprocessing.fill_background(screenshot)

        cv.namedWindow('Preview', cv.WINDOW_AUTOSIZE)
        cv.moveWindow('Preview', 500,500)
        cv.imshow('Preview',screenshot)
        
        del camera
        
        
    def area(self):
        LEFT, TOP, RIGHT, BOTTOM = select_area()
        self.ui.LEFT_lineEdit.setText(str(LEFT))
        self.ui.TOP_lineEdit.setText(str(TOP))
        self.ui.RIGHT_lineEdit.setText(str(RIGHT))
        self.ui.BOTTOM_lineEdit.setText(str(BOTTOM))
    
    def switch1(self):
        self.ui.AUTO_MODE_radioButton.setChecked(True)
        self.ui.MANUAL_MODE_radioButton.setChecked(False)
        
    def switch2(self):
        self.ui.AUTO_MODE_radioButton.setChecked(False)
        self.ui.MANUAL_MODE_radioButton.setChecked(True)
        
    def apply_settings(self):
        config.set('settings','TRANSLATE',str(self.ui.TRANSLATE_checkBox.isChecked() ) )
        config.set('settings','RESIZE',str(self.ui.RESIZE_checkBox.isChecked() ) )
        config.set('settings','REDUCE_NOISE',str(self.ui.REDUCE_NOISE_checkBox.isChecked() ) )
        config.set('settings','OUTLINED',str(self.ui.OUTLINED_checkBox.isChecked() ) )

        config.set('settings','threshold',str(self.ui.THRESHOLD_spinBox.value() ) )
        
        config.set('settings','AUTO_MODE',str(self.ui.AUTO_MODE_radioButton.isChecked() ) )
        config.set('settings','MANUAL_MODE',str(self.ui.MANUAL_MODE_radioButton.isChecked() ) )

        config.set('settings','RESIZE_SCALE',self.ui.RESIZE_SCALE_lineEdit.text() )
        config.set('settings','SPEED_UP',self.ui.SPEED_UP_lineEdit.text() )
        config.set('settings','IGNORE_WORDS',self.ui.IGNORE_WORDS_lineEdit.text() )
        config.set('settings','LEFT',self.ui.LEFT_lineEdit.text() )
        config.set('settings','TOP',self.ui.TOP_lineEdit.text() )
        config.set('settings','RIGHT',self.ui.RIGHT_lineEdit.text() )
        config.set('settings','BOTTOM',self.ui.BOTTOM_lineEdit.text() )
        config.set('settings','PAUSE_KEY',self.ui.PAUSE_KEY_lineEdit.text() )
        config.set('settings','MANUAL_MODE_KEY',self.ui.MANUAL_MODE_KEY_lineEdit.text() )

        config.set('settings','SPEAK_LANGUAGE', self.ui.SPEAK_LANGUAGE_comboBox.currentText())
        config.set('settings','TRANSLATE_FROM', self.ui.TRANSLATE_FROM_comboBox.currentText())

        with open(settings_file,'w') as config_file:
            config.write(config_file)

    
    def refresh_settings(self):
        global SPEAK_LANGUAGE,DETECTION_LANGUAGES,TRANSLATE,TRANSLATE_FROM,TRANSLATE_TO,SPEED_UP,REDUCE_NOISE,OUTLINED,RESIZE,RESIZE_SCALE,PAUSE_KEY,MANUAL_MODE_KEY,IGNORE_WORDS,THRESHOLD
    
        SPEAK_LANGUAGE=config['settings']['SPEAK_LANGUAGE']
        DETECTION_LANGUAGES = [SPEAK_LANGUAGE]
        
        TRANSLATE =eval(config['settings']['translate'])
        TRANSLATE_FROM=config['settings']['TRANSLATE_FROM']
        TRANSLATE_TO=SPEAK_LANGUAGE

        if TRANSLATE:
            DETECTION_LANGUAGES =[TRANSLATE_FROM]

        
        SPEED_UP=int(config['settings']['SPEED_UP'])
        
        RESIZE=eval(config['settings']['resize'])
        RESIZE_SCALE=float(config['settings']['RESIZE_SCALE'])

        THRESHOLD=int(config['settings']['threshold'])
        REDUCE_NOISE =eval(config['settings']['REDUCE_NOISE'])
        OUTLINED= eval(config['settings']['OUTLINED'])
        
        LEFT=int(config['settings']['LEFT']) 
        TOP=int(config['settings']['TOP']) 
        RIGHT=int(config['settings']['RIGHT']) 
        BOTTOM=int(config['settings']['BOTTOM']) 
        
        AUTO_MODE=eval(config['settings']['auto_mode'])
        PAUSE_KEY=int(config['settings']['PAUSE_KEY'], 16)
        MANUAL_MODE=eval(config['settings']['manual_mode'])
        MANUAL_MODE_KEY=int(config['settings']['MANUAL_MODE_KEY'], 16)
        
        PAUSE_KEY_status=False
        
        IGNORE_WORDS=config['settings']['ignore_words'].split(',')

    
        
    def run(self):
        self.apply_settings()
        self.refresh_settings()
        

        if self.ui.RUN_pushButton.text()=='Run':
            self.ui.RUN_pushButton.setText("Wait")
            self.ui.RUN_pushButton.setStyleSheet('QPushButton{color: rgb(0,0,0);background-color: rgb(252, 215, 3);border-radius: 5px;}')

            if iso_converter.convert_to_pt2t (DETECTION_LANGUAGES[0]) not in  pytesseract.get_languages(config=''):
                download_tessdata(iso_converter.convert_to_pt2t (DETECTION_LANGUAGES[0]))
                
            if eval(config['settings']['AUTO_MODE']):
                self.t = threading.Thread(target=auto_mode, args=(LEFT, TOP, RIGHT,  BOTTOM,), daemon=True)
                self.t.start()
            elif eval(config['settings']['MANUAL_MODE']):
                
                self.t = threading.Thread(target=manual_mode,daemon=True)
                self.t.start()
                
            self.ui.RUN_pushButton.setText("Stop and exit")
            self.ui.RUN_pushButton.setStyleSheet('QPushButton{color: rgb(255, 255, 255);background-color: rgb(200, 0, 0);border-radius: 5px;}')

        else:
            global stop_thread
            stop_thread = True
            sys.exit()

  
    def show(self):
        self.ui.show()
        
    def closeEvent(self):
        sys.exit()


# In[ ]:


loader=QUiLoader()


# In[ ]:


app= QApplication(sys.argv)


# In[ ]:


widget= mywidget()

widget.show()

app.exec()


# In[ ]:




