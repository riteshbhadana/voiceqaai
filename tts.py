from gtts import gTTS

def speak(text):
    if not text or len(text.strip()) < 20:
        return None

    tts = gTTS(text[:1000])
    tts.save("answer.mp3")
    return "answer.mp3"
