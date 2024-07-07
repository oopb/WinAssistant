from http import HTTPStatus
from dashscope import Application
import os
import whisper
import pyaudio
import wave


def call_with_stream():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    seconds = 5
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    # 完成语音输入，生成名为output.wav的MP3文件//////////////////////////////////////////

    model = whisper.load_model("base")

    # 加载音频并将其填充/修剪至30秒
    audio = whisper.load_audio("output.wav")
    audio = whisper.pad_or_trim(audio)

    # 计算log-Mel谱图并移到与模型相同的设备
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # 检测说话的语言
    _, probs = model.detect_language(mel)
    print(f"检测到的语言: {max(probs, key=probs.get)}")

    # 解码音频
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # 打印识别的文本
    print(result.text)

    # 读取MP3文件，将语音转化为文本///////////////////////////////////////////////
    # result.text作为大模型的输入////////////////////////////////////////////////

    responses = Application.call(app_id='7f11dc1e8c824f55bea1757fea356329',
                                 api_key='sk-05R1EwaLNo',
                                 prompt='你是一个熟练掌握Windows命令的程序员，接下来你会接收到一些自然语言，'
                                        '请结合你的已有知识以及知识库中文档、压缩文件、文件夹、可执行文件等相关信息将接收到的自然语言转化为windows命令，'
                                        '而且你只会输出Windows命令，不会输出其他任何多余的东西，指令为' +
                                        result.text,
                                 # input(),
                                 stream=True
                                 )

    for response in responses:
        if response.status_code != HTTPStatus.OK:
            print('request_id=%s, code=%s, message=%s\n' % (
                response.request_id, response.status_code, response.message))

    print('%s\n' % (response.output.text))
    os.popen(response.output.text)


if __name__ == '__main__':
    '''i : int = 1
    while(i <= 3):
        call_with_stream()
        i = i + 1'''

    call_with_stream()
