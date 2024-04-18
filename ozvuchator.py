#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install easyocr
# !pip install dxcam
# !pip install matplotlib
# !pip install gTTS
# !pip install soundfile
# !conda install -c main ffmpeg -y
# !pip install -U deep-translator
# !pip install pydub
# !pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# !pip install PySide6
# !pip install pyaudio
# !pip install cx-Freeze


# In[2]:


print('Зачекайте...')


# In[3]:


from gtts import gTTS
# from tempfile import TemporaryFile
import io
# import soundfile as sf
import easyocr
import time
import cv2 as cv
import dxcam
from IPython.display import display, clear_output,Audio
from configparser import ConfigParser
import win32api
from matplotlib import pyplot as plt
from win32api import GetSystemMetrics
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from pydub.playback import play
# import re
import string
from difflib import SequenceMatcher
import threading


# In[4]:


# translator_set=set(GoogleTranslator().get_supported_languages(as_dict=True).values())
# ocr_set=set(['abq','ady','af','ang','ar','as','ava','az','be','bg','bh','bho','bn','bs','ch_sim','ch_tra','che','cs','cy','da','dar','de','en','es','et','fa','fr','ga','gom','hi','hr','hu','id','inh','is','it','ja','kbd','kn','ko','ku','la','lbe','lez','lt','lv','mah','mai','mi','mn','mr','ms','mt','ne','new','nl','no','oc','pi','pl','pt','ro','rs_cyrillic','rs_latin','sck','sk','sl','sq','sv','sw','ta','tab','te','th','tjk','tl','tr','ug','uk','ur','uz','vi',])
# import gtts

# speaker_set=set(gtts.lang.tts_langs().keys())
# ocr_set.intersection(translator_set,speaker_set)


# In[5]:


settings_file='settings.ini'
config = ConfigParser()
config.read(settings_file)
# list(config['settings'])


# In[6]:


SPEAK_LANGUAGE=config['settings']['SPEAK_LANGUAGE']


TRANSLATE =eval(config['settings']['translate'])
TRANSLATE_FROM=config['settings']['TRANSLATE_FROM']

DETECTION_LANGUAGES = [SPEAK_LANGUAGE,TRANSLATE_FROM]

TRANSLATE_TO=SPEAK_LANGUAGE

SPEED_UP=int(config['settings']['SPEED_UP'])

FPS=True

RESIZE=eval(config['settings']['resize'])
RESIZE_SCALE=float(config['settings']['RESIZE_SCALE'])

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

# In[7]:


dw = win32api.GetSystemMetrics(0) #screen width in pixels
dh = win32api.GetSystemMetrics(1) #screen height in pixels


# In[8]:


def select_area():
    def draw_rectanlge(event, x, y, flags, param):
        global ix,iy,drawing,overlay ,img
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y
        elif event == cv.EVENT_MOUSEMOVE:
            if drawing == True:
                cv.rectangle(img, (ix, iy), (x, y), (41, 215, 162), -1)
                cv.putText(img, 'PRESS ANY KEY TO SELECT THIS AREA', (ix, iy-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (55,46,52), 2)
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


# In[9]:


#This function will create only one screenshot and show the area where text will be performed
def show_area():
    camera = dxcam.create(output_color="RGB")
    frame = camera.grab((0,0,dw,dh))
    image_with_box=cv.rectangle(frame,[LEFT,TOP],[RIGHT,BOTTOM],[76,130,250], 8)
        
    plt.figure(figsize=(10,10))
    plt.imshow(image_with_box)
    
    print("ВАША ЗОНА, ДЕ БУДЕ ВИЯВЛЕНО ТЕКСТ: ")
    print('Ліво='+str(LEFT),'Верх='+str(TOP),'Право='+str(RIGHT),'Низ='+str(BOTTOM), sep=' pixels \n', end=' pixels \n')
    del camera

# show_area()


# In[10]:


def speak(my_text):
    with io.BytesIO() as f:
        gTTS(text=my_text, lang=SPEAK_LANGUAGE).write_to_fp(f)
        f.seek(0)
        
        sound = AudioSegment.from_file(io.BytesIO(f.read()))
        sound = sound.speedup(1.1+(SPEED_UP/100), 150, 25)
        # audio_len=len(sound)/1000
        play(sound[0:len(sound)-200])


# In[11]:


def remove_short_strings(list_of_strings):
  return [string for string in list_of_strings if len(string) > 2]

def toggle(variable):
    time.sleep(0.5)
    return not variable

def clean_result(list_of_strings):
    for i, string in enumerate(list_of_strings):
        for word in IGNORE_WORDS:
            string=string.replace(word, ' ')
        list_of_strings[i]=string
    return list_of_strings

def translator(list_of_strings):
    return [GoogleTranslator(source=TRANSLATE_FROM, target=TRANSLATE_TO).translate(string) for string in list_of_strings]

def compare_strings(string1, string2):
    string1 = string1.lower()
    string2 = string2.lower()

    translator = str.maketrans('', '', string.punctuation)
    string1 = string1.translate(translator)
    string2 = string2.translate(translator)

    if SequenceMatcher(a=string1,b=string2).ratio()>0.95:
        return True
    else:
        return False


# In[12]:


def auto_mode(LEFT, TOP, RIGHT, BOTTOM):
    camera = dxcam.create(output_color="RGB")
    camera.start(region=(LEFT, TOP, RIGHT, BOTTOM))
    reader = easyocr.Reader(DETECTION_LANGUAGES)

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
            display('НАТИСНУТО КЛАВІШУ ПАУ')
        if not PAUSE_KEY_status:
            
                
            clear_output(wait=True)
            start= time.time()
        
            screenshot = camera.get_latest_frame()
            
            screenshot = cv.resize(screenshot, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE, interpolation = cv.INTER_AREA)
            # plt.imshow(screenshot)
            # plt.show()
        
        
            result = reader.readtext(screenshot,detail=0,paragraph=True)
            result=remove_short_strings(result)
            # result= [' '.join(result)]
            result=clean_result(result)
    
            if TRANSLATE:
                result= translator(result)
                
            display(result)
            
            if compare_strings(' '.join(result),' '.join(DETECTED_TEXT)):
                pass
            else:
                DETECTED_TEXT=result                
                
                for r in result:
                    if r=='': continue 
                    speak_thread = threading.Thread(target=speak, args=(r,), daemon=True)
                    speak_thread.start()
                    # speak(r)
                    while speak_thread.is_alive():
                        if win32api.GetAsyncKeyState(PAUSE_KEY)<0:
                            PAUSE_KEY_status = toggle(PAUSE_KEY_status)
                            display('ПАУЗА')
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


# In[13]:


# auto_mode(*select_area())


# In[14]:


def manual_mode():
    MANUAL_MODE_KEY_status=False
    reader = easyocr.Reader(DETECTION_LANGUAGES)
    
    while True :

        if win32api.GetAsyncKeyState(MANUAL_MODE_KEY)<0:
            global stop_thread
            if stop_thread:
                break
            
            MANUAL_MODE_KEY_status=toggle(MANUAL_MODE_KEY_status)
            display('НАТИСНУТО КЛАВІШУ РУЧНОГО РЕЖИМУ')

 
        if MANUAL_MODE_KEY_status:
            region=select_area()
            
            camera = dxcam.create(output_color="RGB")

            screenshot = camera.grab(region=region)

            screenshot = cv.resize(screenshot, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE, interpolation = cv.INTER_AREA)
            # plt.imshow(screenshot)
            # plt.show()
            
            result = reader.readtext(screenshot,detail=0,paragraph=True)
            result=remove_short_strings(result)
            # result= [' '.join(result)]
            result=clean_result(result)
            
            if TRANSLATE:
                result= translator(result)
                
            display(result)
            
            for r in result:
                    if r=='': continue 
                    speak(r)

    
            MANUAL_MODE_KEY_status=toggle(MANUAL_MODE_KEY_status)
            
            del camera

            clear_output(wait=True)

# manual_mode()   


# # Interface 

# In[15]:


import sys
import os
import ctypes
from PySide6.QtWidgets import QWidget, QApplication, QButtonGroup

from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import  Qt, QRunnable, Slot, QThreadPool,QProcess


# In[16]:


stop_thread = False


class mywidget(QWidget):
    def __init__(self):
        super().__init__()
        self.ui=loader.load('design.ui',None)

        self.ui.TRANSLATE_checkBox.setChecked(eval(config['settings']['translate']))
        self.ui.RESIZE_checkBox.setChecked(eval(config['settings']['RESIZE']))
        


        self.ui.AUTO_MODE_radioButton.setChecked(eval(config['settings']['AUTO_MODE']))
        self.ui.MANUAL_MODE_radioButton.setChecked(eval(config['settings']['MANUAL_MODE']))
        self.ui.AUTO_MODE_radioButton.clicked.connect(self.switch1)
        self.ui.MANUAL_MODE_radioButton.clicked.connect(self.switch2)
        


        self.ui.RESIZE_SCALE_lineEdit.setText(config['settings']['RESIZE_SCALE'])
        self.ui.SPEED_UP_lineEdit.setText(config['settings']['SPEED_UP'])
        self.ui.IGNORE_WORDS_lineEdit.setText(config['settings']['IGNORE_WORDS'])
        self.ui.LEFT_lineEdit.setText(config['settings']['LEFT'])
        self.ui.TOP_lineEdit.setText(config['settings']['TOP'])
        self.ui.RIGHT_lineEdit.setText(config['settings']['RIGHT'])
        self.ui.BOTTOM_lineEdit.setText(config['settings']['BOTTOM'])
        self.ui.PAUSE_KEY_lineEdit.setText(config['settings']['PAUSE_KEY'])
        self.ui.MANUAL_MODE_KEY_lineEdit.setText(config['settings']['MANUAL_MODE_KEY'])

        self.ui.SPEAK_LANGUAGE_comboBox.setCurrentText(config['settings']['SPEAK_LANGUAGE'])
        self.ui.TRANSLATE_FROM_comboBox.setCurrentText(config['settings']['TRANSLATE_FROM'])

        self.ui.Select_area_pushButton.clicked.connect(self.area)
        self.ui.APPLY_pushButton.clicked.connect(self.apply_settings)
        self.ui.RUN_pushButton.clicked.connect(self.run)

        app.aboutToQuit.connect(self.closeEvent)

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
        
        
    def run(self):
        self.apply_settings()

        if self.ui.RUN_pushButton.text()=='Запуск':
            self.ui.RUN_pushButton.setText("Wait")
            self.ui.RUN_pushButton.setStyleSheet('QPushButton{color: rgb(0,0,0);background-color: rgb(252, 215, 3);border-radius: 5px;}')
            
            SPEAK_LANGUAGE=config['settings']['SPEAK_LANGUAGE']
            DETECTION_LANGUAGES = [SPEAK_LANGUAGE]
            
            TRANSLATE =eval(config['settings']['translate'])
            TRANSLATE_FROM=config['settings']['TRANSLATE_FROM']
            TRANSLATE_TO=SPEAK_LANGUAGE
            
            SPEED_UP=int(config['settings']['SPEED_UP'])
            
            RESIZE=eval(config['settings']['resize'])
            RESIZE_SCALE=float(config['settings']['RESIZE_SCALE'])
            
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

            if eval(config['settings']['AUTO_MODE']):
                self.t = threading.Thread(target=auto_mode, args=(LEFT, TOP, RIGHT,  BOTTOM,), daemon=True)
                self.t.start()
            elif eval(config['settings']['MANUAL_MODE']):
                
                self.t = threading.Thread(target=manual_mode,daemon=True)
                self.t.start()
                
            self.ui.RUN_pushButton.setText("Зупинити і закрити")
            self.ui.RUN_pushButton.setStyleSheet('QPushButton{color: rgb(255, 255, 255);background-color: rgb(200, 0, 0);border-radius: 5px;}')

        else:
            self.ui.RUN_pushButton.setText("Запуск")
            self.ui.RUN_pushButton.setStyleSheet('') 
            global stop_thread
            stop_thread = True
            sys.exit()
    

  
    def show(self):
        self.ui.show()
        
    def closeEvent(self):
        sys.exit()


# In[17]:


loader=QUiLoader()


# In[18]:


app= QApplication(sys.argv)


# In[ ]:


widget= mywidget()

widget.show()

app.exec()


# In[ ]:





# In[ ]:




