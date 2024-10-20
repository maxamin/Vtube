#coding=utf-8
# cython:language_level=3

import threading,time
import cv2,numpy as np;
import pyaudio,wave


class AudioRecorder:
    PlayFlag=True;
    RecordFlag=True;

    def __init__(self):
        self.PlayFlag=True;
        self.RecordFlag=True;
        self.mPyAudio=pyaudio.PyAudio()

    def play_audio(self,wave_input_path):
       
        wf = wave.open(wave_input_path, 'rb')  # 读 wav 文件
        stream = self.mPyAudio.open(format=self.mPyAudio.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(1024)  # 读数据
        while len(data) > 0:
            stream.write(data)
            data = wf.readframes(1024)

        stream.stop_stream()  # 关闭资源
        stream.close()
        self.mPyAudio.terminate()

    def _getSteroMixDeviceID(self,p):
        target = '立体声混音'
        for i in range(self.mPyAudio.get_device_count()):
            devInfo = self.mPyAudio.get_device_info_by_index(i)
            #print(devInfo)
            if devInfo['name'].find(target) >= 0 and devInfo['hostApi'] == 0:
                return i
        print('无法找到内录设备!')
        return -1

    def py_record_thread(self,wave_save_path,record_mode="mix",mic_scale=1,speaker_scale=4,nchannels=1,rate=44100,chunk=2048,format=pyaudio.paInt16):
        pya=pyaudio.PyAudio()
        dev_idx=self._getSteroMixDeviceID(pya)
        stream_speaker =self.mPyAudio.open(format=format,channels=nchannels,rate=rate,input_device_index=dev_idx,input=True)
        stream_mic=self.mPyAudio.open(format=format,channels=nchannels,rate=rate,input=True)

        wav=wave.open(wave_save_path,'wb')
        wav.setnchannels(1)
        wav.setframerate(rate)
        wav.setsampwidth(self.mPyAudio.get_sample_size(format))
        record_start_timetick=time.time()
        while(self.RecordFlag is True):
            if  "麦" in record_mode:
                data_mic=stream_mic.read(chunk)
                wav.writeframes(data_mic)
                #print(f"mic recording {i}s",end='\r')
            if "系统" in record_mode:
                data_speaker=stream_speaker.read(chunk)
                frame_speaker=np.frombuffer(data_speaker,"int16")
                frame_speaker=np.array(frame_speaker,dtype='int16')
                frame_speaker=frame_speaker*speaker_scale;
                signal =np.clip(frame_speaker, -32767, 32766)
                encodecoded = wave.struct.pack("%dh" % (len(signal)), *list(signal))
                wav.writeframes(encodecoded)
            else:
                data_mic=stream_mic.read(chunk)
                data_speaker=stream_speaker.read(chunk)
                frame_mic=np.frombuffer(data_mic,"int16")
                frame_mic=np.array(frame_mic,dtype='int16')
                frame_speaker=np.frombuffer(data_speaker,"int16")
                frame_speaker=np.array(frame_speaker,dtype='int16')
                frame_speaker=frame_speaker*speaker_scale;
                mix=(frame_mic+frame_speaker);
                signal =np.clip(mix, -32767, 32766)
                encodecoded = wave.struct.pack("%dh" % (len(signal)), *list(signal))
                wav.writeframes(encodecoded)
            record_time=round(time.time()-record_start_timetick,2)
            print(f"[P]录制{record_mode},{record_time}s",end='\r')

        wav.close()
        stream_mic.stop_stream()
        stream_mic.close()
        stream_speaker.close()

    def start_audio_record(self,file_name,record_mode="mix"):
        if ".wav" in file_name:
            wave_save_path=file_name
        else:
            wave_save_path=f"{file_name}.wav"
        self.RecordFlag=True;
        th=threading.Thread(target=self.py_record_thread,args=(wave_save_path,record_mode))
        th.start()

    def stopRecord(self):
        self.RecordFlag=False;

    def stopPlay(self):
        self.PlayFlag=False;

if __name__=="main":
    recorder=AudioRecorder()
    recorder.start_audio_record("F:","speaker_rec","speaker")
    time.sleep(5)
    recorder.stopRecord()
    print("程序执行结束")
