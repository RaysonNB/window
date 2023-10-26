import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    audio = r.listen(source)

try:
    text = r.recognize_google(audio, language='zh-TW')
    print("識別結果：", text)
except sr.UnknownValueError:
    print("無法識別語音")
