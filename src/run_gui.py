# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 09:21:20 2020

@author: Suhaib
"""

import PySimpleGUI as sg
import video_sentiment_PoetryMirrorFunction as vs
from PoemGenerator import *
import json
import operator

sg.theme('DarkBlack')	# Add a touch of color
# All the stuff inside your window.
#layout = [  [sg.Text('This is some poetry', font=('Traveling _Typewriter', 70))],
#            [sg.Text('Enter something on Row 2'), sg.InputText()],
#            [sg.Button('Ok'), sg.Button('Cancel'), sg.Button('Exit')] ]

json_path = "saved_poems.json"

sentiment = "neutral"
welcome_prompt = "\nHi there! I am meant to be a mirror. \n But not one of the regular mirrors \n you are used to. \n \n I believe you are a complex creature. \n And you deserve a reflection of yourself \n that goes beyond a mere visual image. \n \n So, I am going to craft some poetry \n that will offer \n a more nuanced reflection of you! \n  \n I will need your help though. \n Please click ok when you're ready."
facial_sentiment_prompt = "\nLet's start with your emotions. Think of a memorable event (can be positive, negative or neutral) you experienced last week, month or year and try to embody how that memory makes you feel. \n \n I will monitor your facial expressions for a few seconds and try to grasp a general idea of the emotions that you experience. \n \n When ready click ok and I will start looking at your emotions right away so just display whichever emotions you want me to notice. \n \n [Note that your image data will NOT be stored]"
drunkness_prompt = "I know it sounds odd, \n but I'm gonna ask anyway... \n \n How drunk should I be \n when I compose your reflection?"
shape_prompt = "What 'shape' should your poem have? Which one do you feel like choosing at this moment? \n Click on your favorite shape below and I will note it down."
working_prompt = "\n \n \n \n I am working on the poem that will be your reflection!"


font_size = 46
font_size_small = 10
font_size_smallest = 2
welcome_font_size = 30

rhyming_dict = {" ": 'AABB', "2": 'ABAB', "3": "ABBA"}

def updateJsonFile(poem_tuple):
    json_file = open(json_path, "r")
    data = json.load(json_file)
    json_file.close()
    next_id = getNextID()
    data[next_id] = poem_tuple
    jsonFile = open(json_path, "w+")
    jsonFile.write(json.dumps(data))
    jsonFile.close()


def getNextID():
    json_file = open(json_path, "r")
    data = json.load(json_file)
    keys_list = data.keys()
    keys_list = list(map(int, keys_list))
    next_id = max(keys_list) + 1
    json_file.close()
    return next_id

def make_window(using_this_layout):
    window = sg.Window('Window Title', using_this_layout, element_justification='c', 
                   no_titlebar=True).Finalize()
    window.Maximize()
    
    event, values = window.Read()
#    print(event)
#    print(values)
    window.Close()
    return (event, values)

layout = [ [sg.Text(welcome_prompt, justification='center', size=(50,15), font=('Traveling _Typewriter', welcome_font_size))],
            [sg.Button('Ok')]]

event, values = make_window(layout)

layout = [ [sg.Text(facial_sentiment_prompt, justification='center', size=(50,15), font=('Traveling _Typewriter', welcome_font_size))],
            [sg.Button('Ok')]]

event, values = make_window(layout)

sentiment = vs.facial_sentiment_runner()
chosen_sentiment_prompt = " \n \n \n \n \n Based on the expressions you embodied, I have chosen that your poem will be more on the \n " + sentiment.upper() +" side. \n \n When you're ready, let's move to the next step by clicking ok."
layout = [ [sg.Text(chosen_sentiment_prompt, justification='center', size=(50,15), font=('Traveling _Typewriter', welcome_font_size))],
            [sg.Button('Ok')]]

event, values = make_window(layout)

layout = [  [sg.Text(drunkness_prompt, justification='center', size=(50,5), font=('Traveling _Typewriter', font_size))],
            [sg.Slider(range=(0,10),orientation='h', resolution=1, default_value=0, size=(100,200), font=('Helvetica', 26), key='temperature')],
            [sg.Button('Ok')]]

event, values = make_window(layout)
chosen_temperature = values['temperature']

layout = [  [sg.Text(shape_prompt,  justification='center', size=(50,4), font=('Traveling _Typewriter', welcome_font_size))],
            [sg.Button(' ', disabled_button_color=None, image_filename="AABB.png", image_size=(374,182))],
            [sg.Text(" ", size=(50,2), font=('Traveling _Typewriter', font_size_smallest))],
            [sg.Button('2', image_filename="ABAB.png", image_size=(374,182))],
            [sg.Text(" ", size=(50,2), font=('Traveling _Typewriter', font_size_smallest))],
            [sg.Button('3', image_filename="ABBA.png", image_size=(374,182))]]

event, values = make_window(layout)
chosen_rhyming_scheme = rhyming_dict[event]


layout = [ [sg.Text("\nYour poem is being generated!\n\nPlease be patient for now\n\nThis can take up to 30 seconds", justification='center', size=(50,6), font=('Traveling _Typewriter', welcome_font_size))],]
waitwindow = sg.Window('Window Title', layout, element_justification='c',
                   no_titlebar=True).Finalize()
waitwindow.Maximize()
print("Generating!")
# generated_poem = "\n Here in the dark, \n where do we park? \n The world is on fire \n what is my desire?" # GENERATE FUNCTION --> MAKE SURE TO HAVE \n characters
generated_poem = GeneratePoem(chosen_rhyming_scheme, sentiment.lower(), chosen_temperature)
# generated_poem = "Penis am i big boi yo whadddup what is my name i steal from rich board people mad rhymes my n-word \n some\nsome\nbome"

poem_id = getNextID() # implement ID

final_choice_tuple = (sentiment, chosen_temperature, chosen_rhyming_scheme, generated_poem)


waitwindow.close()


layout = [ [sg.Text("\n\n" + generated_poem + "\n\n", justification='center', size=(80,8), font=('Traveling _Typewriter', welcome_font_size))],
           [sg.Text("Emotion: "+sentiment, justification='center', size=(20,3), font=('Traveling _Typewriter', welcome_font_size)),
            sg.Text("Drunkness: "+str(chosen_temperature), justification='center', size=(20,3), font=('Traveling _Typewriter', welcome_font_size)),
            sg.Text("Shape: ", justification='center', size=(8,3), font=('Traveling _Typewriter', welcome_font_size)),
            sg.Image(chosen_rhyming_scheme+".png", key="loading")],
            [sg.Text("Poem ID: "+str(poem_id), justification='center', size=(50,1), font=('Traveling _Typewriter', welcome_font_size))],
            [sg.Button('Ok')]]

event, values = make_window(layout)
excuse = "I couldn't generate a poem for you at this moment, please try again!"

if generated_poem != excuse:
    updateJsonFile(final_choice_tuple)

print(final_choice_tuple)
#layout = [ [sg.Text(working_prompt, justification='center', size=(50,15), font=('Traveling _Typewriter', font_size))]]

# import json
# json_path = "saved_poems.json"
# poem_tuple = ("neutral", 5, "ABAB", "\n Here in the dark, \n where do we park? \n The world is on fire \n what is my desire?")
# poem_id = 1
# data = {}
# data[poem_id] = poem_tuple
# jsonFile = open(json_path, "w+")
# jsonFile.write(json.dumps(data))
# jsonFile.close()

# Create the Window
#window = sg.Window('Window Title', layout, element_justification='c', 
#                   no_titlebar=True).Finalize()
#window.Maximize()
#
#event, values = window.Read()
#print(event)
#print(values)
#window.Close()

# Event Loop to process "events" and get the "values" of the inputs
#while True:
#    event, values = window.read()
#    if event in (None, 'Cancel', 'Exit'):	# if user closes window or clicks cancel
#        break
#    sg.popup("You entered", values[0])
#    print('You entered ', values[0])
#
#window.close()

#layout = [  [sg.Text("How drunk should I be when I compose your reflection?", size=(50,2), font=('Traveling _Typewriter', font_size))],
#            [sg.Slider(range=(0,10),orientation='h', resolution=1, default_value=0, size=(100,200), font=('Helvetica', 20), key='temperature')],
#            [sg.Text("What 'shape' should your reflection have?", size=(50,2), font=('Traveling _Typewriter', font_size))],
#            [sg.Image("1.png"), sg.Radio(' ', "RADIO1", default=True)],
#            [sg.Image("2.png"), sg.Radio(' ', "RADIO2")],
#            [sg.Image("3.png"), sg.Radio(' ', "RADIO3")],
#            [sg.Button('Ok', border_width=10)]]
