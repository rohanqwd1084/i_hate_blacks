import speech_recognition as sr
import webbrowser
import pyttsx3

recognizer = sr.Recognizer()
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    speak("Intializing Faron.......")

while True:
    r=sr.Recognizer()
    with sr.Microphone() as source:
        speak("Listening...")
        audio = r.listen(source,timeout=2,phrase_time_limit=1)

    speak("Recognizing...")
    try:
        with sr.Microphone() as source:
            audio = r.listen(source,timeout=2,phrase_time_limit=6)
            command=r.recognize_google(audio)
            with sr.Microphone() as source:
                print("Listening...")
                audio = r.listen(source,timeout=2,phrase_time_limit=3)
            command=r.recognize_google(audio)
            if (command.lower()=="jarvis"):
                speak("Yes Sir")
            elif (command.lower()=="open youtube"):
                speak("Opening Youtube")
                webbrowser.open("https://www.youtube.com")
            elif (command.lower()=="open google"):
                speak("Opening Google")
                webbrowser.open("https://www.google.com")
            elif (command.lower()=="open hianime and play death note"):
                speak("Opening hianime and playing Death Note")
                webbrowser.open("https://hianime.re/watch/death-note-fc8mq/ep-32")
            elif (command.lower()=="what is your name"):
                speak("My name is Faron")
            elif (command.lower()=="how are you"):
                speak("I am fine, Thank you")
            elif (command.lower()=="stop"):
                speak("Goodbye Sir")
                exit()
            elif (command.lower()=="play songs by seedhe maut"):
                speak("Playing songs by Seedhe Maut")
                webbrowser.open("https://open.spotify.com/playlist/4QwIP43a0X7nJl40Dj26yO?si=1d18d173de9f4aad")
            else:
                speak("Please say the command again.")

    except Exception as e:
        print("Could not request results; {0}".format(e))
