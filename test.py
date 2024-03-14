import numpy 
import speech_recognition 
import whisper
import torch
from flask import Flask
import threading

if __name__ == '__main__':

    for i in [numpy,speech_recognition, whisper,torch, Flask,threading]:
        if i.__name__:
            print(f"{i} is ok")
        else:
            print(f"{i} is wrong")

