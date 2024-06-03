# import speech
#
# word = speech.input
# print(word)
import speech_recognition as sr

recognizer = sr.Recognizer()

while True:
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        print("Say something please !")
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_whisper(audio)  # 还可以选择不同的数据源，从而用来识别不同的语言
            print("You said : {}".format(text))
        except:
            print("Sorry I can't hear you!")
