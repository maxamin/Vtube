# cython:language_level=3
import pickle,time,threading,cv2,os,numpy as np,sys,pickle;

from core.leras import nn
from PyQt5.QtWidgets import QFileDialog,QMessageBox,QPushButton,QTableWidgetItem;
from PyQt5.QtCore import QTimer,QThread,QObject,Qt,pyqtSignal,QObject;
from PyQt5.QtGui import QPixmap,QImage;
from PyQt5.QtCore import QStringListModel,QAbstractListModel,QModelIndex,QSize
import kit.ShareData as sd
import kit.Auth as Auth
import json
import pyaudio
import librosa  
import fairseq,pyworld

#import set_env
#set_env.set_env()
import torch

class Config:
    input_wavfile=False; input_device=True; convert_raw=True; convert_pitch=False; convert_ai=False;
    device="cuda:0"; 
    is_half=True; threhold=-40; 
    samplerate=16000; input_sr=16000;
    block_time=0.10; extra_time=1.80; crossfade_time=0.14; sola_time=0.01; delay_time=0.001;
    f0_up_key=0;   n_cpu=4; audio_volume=1.0;
    f0method="rmvpe"; play_raw=False;
    in_device_idx=0; out_device_idx=0;


class DebugInfo:
    block_frame_num=0;  hubert_in_dim=None;  hubert_out_dim=None; 
    pitch_in_dim=None;  pitch_out_dim=None;  max_f0=0;
    rvc_in_dim=None; rvc_out_dim=None; frame_infer_time=0;
    hubert_time_use=0.0; pitch_time_use=0; rvc_time_use=0.0;  total_time_use=0;
    stream_in_time=0; stream_out_time=0;
class VoiceUiHandler(QObject):
    mUpdateSignal=pyqtSignal();

    def __init__(self,ui):
        super().__init__()
        self.ui=ui;
        self.config=Config();
        self.debug=DebugInfo();
        self.p = pyaudio.PyAudio() 
        
        self.hubert_model=None;
        self.RVC_Net=None;
        self.in_stream=None; self.out_stream=None; self.wave_file=None;
        self.mUpdateSignal.connect(self.UpdateDebugInfoPreview)
        self.LoadSystemDriversList()
        
     
    def UpdateDebugInfoPreview(self):
        config=self.config
        convert_status="运行中" if self.AudioConvertRunFlag else "停止"
        if self.config.convert_raw :
            mode="原声"
        elif self.config.convert_pitch:
            mode="仅变调"
        else:
            mode="AI变声"
        self.ui.tableWidgetVoiceConvert.setItem(0,0,QTableWidgetItem(f"{convert_status}"));
        self.ui.tableWidgetVoiceConvert.setItem(0,1,QTableWidgetItem(f"{self.debug.stream_out_time}ms"));
        self.ui.tableWidgetVoiceConvert.setItem(0,1,QTableWidgetItem(f"{self.debug.stream_in_time}ms"));
        self.ui.tableWidgetVoiceConvert.setItem(0,2,QTableWidgetItem(f"{self.debug.frame_infer_time}ms"));
        self.ui.tableWidgetVoiceConvert.setItem(0,3,QTableWidgetItem(f"{self.debug.stream_out_time}ms"));
        self.ui.tableWidgetVoiceConvert.setItem(0,4,QTableWidgetItem(f"{self.debug.total_time_use}ms")); 
        self.ui.tableWidgetVoiceConvert.setItem(0,5,QTableWidgetItem(f"{mode}")); 
        self.ui.tableWidgetVoiceConvert.setItem(0,6,QTableWidgetItem(f"{round(self.debug.max_f0)}hz")); 
    


    def LoadSystemDriversList(self): 

        #--- 加载显卡列表
        self.ui.comboAudioCalcDevice.clear()
        self.ui.comboInferDevices.clear()
        for i in range(torch.cuda.device_count()):
            gpu_name=torch.cuda.get_device_name(i)
            self.ui.comboAudioCalcDevice.addItem(gpu_name)
            self.ui.comboInferDevices.addItem(gpu_name)
        
            #---- 输出host_api的结构信息
        #for i in range(4):
        #    api_info=self.p.get_host_api_info_by_index(i)
        #    print(api_info)
        
        #---- 加载声音输入设备列表
        device_count= self.p.get_device_count() 
        self.ui.comboInputAudioDevices.clear()
        self.ui.comboOutputAudioDevices.clear()
        for i in range(device_count):
            dev_info=self.p.get_device_info_by_index(i)
            device_name=dev_info.get("name","未知设备")
            device_idx=dev_info.get("index",0)
            hostApi=dev_info.get("hostApi",0)
            hostApiStr={0:"MME",1:"DirectSound",2:"WASAPI",3:"WDM-KS",4:"-"}[hostApi]
            maxInputChannels=dev_info.get("maxInputChannels",0)
            maxOutputChannels=dev_info.get("maxOutputChannels",0)
            device_info_str=f"{device_idx}|{device_name}({hostApiStr})"
            if hostApi in [0]:
                #print(dev_info)
                if maxInputChannels>0:
                    self.ui.comboInputAudioDevices.addItem(device_info_str)
                if maxOutputChannels>0:
                    self.ui.comboOutputAudioDevices.addItem(device_info_str)
                #print(f"-------------------{i}---------")
        
        self.ChooseDefaultDevices()

    #---------------------------------
    #------------ 加载模型-------------
    #---------------------------------
    def LoadVoiceModels(self): 
        if self.config.convert_raw or self.config.convert_pitch:
            return

        #----加载Hubert模型
        if self.hubert_model is None:
            models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([r"../weights/hubert_base.pt"],suffix="",) 
            self.hubert_model = models[0]
            if self.config.is_half:
                self.hubert_model = self.hubert_model.half()
            else:
                self.hubert_model = self.hubert_model.float()
            self.hubert_model.eval()
            self.hubert_model = self.hubert_model.to(self.config.device)
            print("完成加载HuBert模型") 

        from vvc.rvc_models import (SynthesizerTrnMs256NSFsid,SynthesizerTrnMs256NSFsid_nono,
                                           SynthesizerTrnMs768NSFsid,SynthesizerTrnMs768NSFsid_nono)
        
        #----判断模型是否存在
        rvc_model_path=self.ui.lineEditRvcModelFile.text()
        if os.path.exists(rvc_model_path) is False:
            QMessageBox.information(None,"错误","变声模型文件不存在，请检查文件\n Voice Convert Model Not Exists");
            return False

        #----加载RVC的变声模型
        try:
            cpt = torch.load(rvc_model_path, map_location="cpu")
            self.rvc_model_sr = cpt["config"][-1]
            self.config.samplerate= cpt["config"][-1]
            cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
            self.if_f0 = cpt.get("f0", 1)
            self.version = cpt.get("version", "v1")
            if self.version == "v1":
                if self.if_f0 == 1:
                    self.RVC_Net = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=self.config.is_half)
                else:
                    self.RVC_Net = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif self.version == "v2":
                if self.if_f0 == 1:
                    self.RVC_Net = SynthesizerTrnMs768NSFsid(  *cpt["config"], is_half=self.config.is_half)
                else:
                    self.RVC_Net = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del self.RVC_Net.enc_q
            self.RVC_Net.load_state_dict(cpt["weight"], strict=False)            
        except :
            #print("！！加载声音模型发生错误")
            QMessageBox.information(None,"错误","加载声音模型发生错误")
            return False



        self.RVC_Net.eval().to(self.config.device)
        if self.config.is_half:
            self.RVC_Net = self.RVC_Net.half()
        else:
            self.RVC_Net = self.RVC_Net.float()
        print(f"完成加载声音模型,版本{self.version},采样率:{self.rvc_model_sr}")
        return True
    
         

    def initParameters(self):
        #self.config.samplerate = self.rvc_model_sr
        self.raw_input_frame=int(self.config.block_time*self.config.input_sr)
        self.config.crossfade_time = min(self.config.crossfade_time, self.config.block_time)
        self.block_frame = int(self.config.block_time * self.config.samplerate)
        self.crossfade_frame = int(self.config.crossfade_time * self.config.samplerate)
        self.sola_search_frame = int(0.01 * self.config.samplerate)
        self.extra_frame = int(self.config.extra_time * self.config.samplerate)
        self.zc = self.config.samplerate // 100
        self.total_frame=self.extra_frame+ self.crossfade_frame+ self.sola_search_frame+ self.block_frame;

        self.input_wav: np.ndarray = np.zeros(self.total_frame,dtype="float32",)     
        self.cache_pitch: np.ndarray = np.zeros(self.input_wav.shape[0] // self.zc,dtype="int32",)
        self.cache_pitchf: np.ndarray = np.zeros(self.input_wav.shape[0] // self.zc,dtype="float64",)

        self.output_wav_cache: torch.Tensor = torch.zeros(self.total_frame,device=self.config.device,dtype=torch.float32,)
        self.output_wav: torch.Tensor = torch.zeros(self.block_frame, device=self.config.device, dtype=torch.float32)
        self.sola_buffer: torch.Tensor = torch.zeros(self.crossfade_frame, device=self.config.device, dtype=torch.float32)
        self.fade_in_window: torch.Tensor = torch.linspace(0.0, 1.0, steps=self.crossfade_frame, device=self.config.device, dtype=torch.float32)
        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
         

    #---- 开始声音捕获--------
  
    def btnStartVoiceConvertClick(self):
        self.AudioConvertRunFlag=True;

        #---选择GPU设备
        gpu_idx=self.ui.comboAudioCalcDevice.currentIndex()
        self.config.device=f"cuda:{gpu_idx}"
        #--- Ai换声提前加载模型
        if self.config.convert_ai:
            succeed=self.LoadVoiceModels()
            if succeed==False:
                return

        #-----禁用不能调节的界面
        self.ui.sliderBlockTime.setEnabled(False)
        self.ui.spinBlockTime.setEnabled(False)
        self.ui.comboAudioCalcDevice.setEnabled(False)
        self.ui.comboInputAudioDevices.setEnabled(False)
        self.ui.comboOutputAudioDevices.setEnabled(False)
        self.ui.btnBrowseRvcModel.setEnabled(False)
        self.ui.lineEditRvcModelFile.setEnabled(False)
        
        #----预先计算各个帧长度
        self.initParameters()

        #--- 设置输入麦克风的音频流
        if self.config.input_device:
            if self.in_stream is not None:
                self.in_stream.close() 
            self.in_stream=self.p.open(format=pyaudio.paFloat32,channels=1,rate=self.config.input_sr,input=True,frames_per_buffer=self.raw_input_frame,
                                  input_device_index=self.config.in_device_idx);
            self.out_stream=self.p.open(format=pyaudio.paFloat32,channels=1,rate=self.config.samplerate,output=True,frames_per_buffer=self.block_frame,
                               input_device_index=self.config.out_device_idx);

        #---- 读取wav声音文件
        if self.config.input_wavfile:
            import wave
            wav_path=self.ui.lineEditAudioFile.text()
            self.wave_file=wave.open(wav_path,"rb");
            nchannels, sampwidth, framerate, nframes=self.wave_file.getparams()[0:4]
            print(f"加载文件({wav_path}),声道{nchannels},位宽{sampwidth},采样率{framerate},帧数{nframes}")
            self.out_stream=self.p.open(format=pyaudio.paFloat32,channels=1,rate=framerate,output=True,frames_per_buffer=self.block_frame,
                               input_device_index=self.config.out_device_idx);
           
        #--- 设置输出音频流
        

        th=threading.Thread(target=self.CaptureAudioThreadRun)
        th.start()
         

    def CaptureAudioThreadRun(self): 
        #print(f"开始启动换声") 
        while(self.AudioConvertRunFlag): 

            #--- 每帧处理计时
            self.frame_start_time=time.time()

            #---- 人工延迟时间
            sleep_time=float(self.config.delay_time)
            time.sleep(sleep_time)

            #---- 从wav文件读取数据
            if self.config.input_wavfile:
                indata=self.wave_file.readframes(self.block_frame) 
                nchannels, sampwidth, framerate, nframes=self.wave_file.getparams()[0:4]
                block_frame_data=np.fromstring(indata,dtype=np.int16).astype(np.float32)/32677.0
                #print(f"block_frame_data:{block_frame_data.shape}",end='\r')
                if nchannels>1:
                    block_frame_data=np.reshape(block_frame_data,(-1,nchannels));
                    block_frame_data=np.mean(block_frame_data,axis=1)
                

            #---从麦克风输入读取数据
            if self.config.input_device: 
                indata=self.in_stream.read(self.raw_input_frame)
                block_frame_data=np.fromstring(indata,dtype=np.float32).astype(np.float32)
                block_frame_data=librosa.resample(y=block_frame_data,orig_sr=self.config.input_sr,target_sr=self.config.samplerate)
                
            #--计算输入流读取时间
            self.stream_in_endtime=time.time()
            self.debug.stream_in_time=round((self.stream_in_endtime-self.frame_start_time)*1000)

            block_frame_data=block_frame_data*self.config.audio_volume;

            #----流数据进行变声(Infer)
            infer_start_time=time.time()
            outdata,status=self.audio_process_convert(block_frame_data,self.block_frame,None,None)
            self.infer_end_time=time.time()
            self.debug.frame_infer_time=round((self.infer_end_time-infer_start_time)*1000)
             
            #----输出到输出设备
            self.out_stream.write(outdata.tobytes())
            self.stream_out_end_time=time.time()
            self.debug.stream_out_time=round((self.stream_out_end_time-self.infer_end_time)*1000)

            #---- 计算整体延迟
            frame_end_time=time.time()
            self.debug.total_time_use=round((frame_end_time-self.frame_start_time)*1000)
            self.mUpdateSignal.emit() 

        #---- 结束进程
        if self.in_stream is not None: 
           self.p.close(self.in_stream)
           self.in_stream=None;
        if self.out_stream is not None: 
           self.p.close(self.out_stream)
           self.out_stream=None;
        #print("结束换声线程(End  Audio Convert )")


        #------------------------------------------
        #------------- 每捕获声音片段处理-----------
        #-------------------------------------------
    def audio_process_convert(self, block_frame_data, frame_count, time_info,status):
        infer_start_time=time.time()  

        #---- 输入前数据计算 
        #channels=len(block_frame_data.shape)
        #if channels>0:
        # print(f"block_frame_data.shape：{block_frame_data.shape}",end='\r')
        
        #-----  直接播放原声（调试）
        if self.config.convert_raw: 
            return (block_frame_data,pyaudio.paContinue)

        if self.config.convert_pitch:
            block_frame_data=librosa.effects.pitch_shift(y=block_frame_data, sr=self.config.samplerate, n_steps=self.config.f0_up_key)
            return block_frame_data,pyaudio.paContinue

        if self.config.convert_ai:
            if self.hubert_model is None:
                print("！错误，语音识别模型未加载，即将结束换声")
                self.AudioConvertRunFlag=False;
                return block_frame_data,pyaudio.paContinue
            if self.RVC_Net is None:
                print("！错误，换声的模型未加载")
                self.AudioConvertRunFlag=False;
                return block_frame_data,pyaudio.paContinue

        #---响应阈值,过滤低于阈值的声音
        frame_length=2048;hop_length=1024;
        rms = librosa.feature.rms(y=block_frame_data, frame_length=frame_length, hop_length=hop_length) 
        if self.config.threhold > -60:
            db_threhold = (librosa.amplitude_to_db(rms, ref=1.0)[0] < self.config.threhold)
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    block_frame_data[i * hop_length : (i + 1) * hop_length] = 0
                      
        #---- 和前次的input_wav的非block部分拼接  
        self.input_wav[:] = np.append(self.input_wav[self.block_frame :] , block_frame_data)  
        input_resample_16000=librosa.resample(y=self.input_wav,orig_sr=self.config.samplerate,target_sr=16000)
        feats = torch.from_numpy(input_resample_16000).to(self.config.device)
        block_frame_data_16000= feats[-self.block_frame :].cpu().numpy()

        block_ratio = self.block_frame / self.total_frame
        nextra_ratio = (self.crossfade_frame + self.sola_search_frame + self.block_frame) / self.total_frame
        

        #---- Hubert特征提取的输入数据准备
        infer_start_time=time.time()
        feats = feats.view(1, -1)
        if self.config.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        feats = feats.to(self.config.device) 
        self.debug.hubert_in_dim=feats.cpu().numpy().shape 

        #----------- Hubert特征提取 [获得结果维度 1,n,256]  n和输入的序列总长度应该相关
        with torch.no_grad():
            padding_mask = torch.BoolTensor(feats.shape).to(self.config.device).fill_(False)
            inputs = {"source": feats,"padding_mask": padding_mask, "output_layer": 9 if self.version == "v1" else 12, }
            logits = self.hubert_model.extract_features(**inputs)
            if self.version == "v1":
                 self.hubert_feats = self.hubert_model.final_proj(logits[0])
            else:
               self.hubert_feats=logits[0]
        self.hubert_feats = torch.nn.functional.interpolate(self.hubert_feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        self.debug.hubert_out_dim=self.hubert_feats.cpu().numpy().shape 
        hubert_end_time=time.time()
        self.debug.hubert_time_use=hubert_end_time-infer_start_time 

        #----- Pitch音高提取(block_frame_data) 
        if self.if_f0 == 1:
            pitch, pitchf = self.get_f0(block_frame_data_16000, self.config.f0_up_key, self.config.n_cpu, self.config.f0method)
            self.cache_pitch[:] = np.append(self.cache_pitch[pitch[:-1].shape[0] :], pitch[:-1])
            self.cache_pitchf[:] = np.append(self.cache_pitchf[pitchf[:-1].shape[0] :], pitchf[:-1])
            self.p_len = min(self.hubert_feats.shape[1], 13000, self.cache_pitch.shape[0])
        else:
            self.cache_pitch, self.cache_pitchf = None, None
            self.p_len = min(self.hubert_feats.shape[1], 13000)
        pitch_end_time=time.time()
        self.debug.pitch_time_use=pitch_end_time-hubert_end_time
        self.debug.pitch_out_dim=pitch.shape 
        self.debug.pitch_in_dim=block_frame_data_16000.shape 

        #------ RVC_Net 网络推理最终结果
        block_rate=self.block_frame/self.total_frame
        non_extra_rate=1-(self.extra_frame/self.total_frame)
        hubert_pitch_feats = self.hubert_feats[:, :self.p_len, :]
        self.debug.rvc_in_dim=hubert_pitch_feats.shape
        if self.if_f0 == 1:
            cache_pitch = self.cache_pitch[:self.p_len]
            cache_pitchf = self.cache_pitchf[:self.p_len]
            cache_pitch = torch.LongTensor(cache_pitch).unsqueeze(0).to(self.config.device)
            cache_pitchf = torch.FloatTensor(cache_pitchf).unsqueeze(0).to(self.config.device)
        self.p_len = torch.LongTensor([self.p_len]).to(self.config.device)
        sid=torch.LongTensor([0]).to(self.config.device)
        with torch.no_grad():
            if self.if_f0 == 1: 
                infered_audio=self.RVC_Net.infer(hubert_pitch_feats, self.p_len,cache_pitch, cache_pitchf, sid, nextra_ratio)[0][0, 0].data.cpu().float()
            else:
                infered_audio=self.RVC_Net.infer(hubert_pitch_feats, self.p_len, sid, nextra_ratio)[0][0, 0].data.cpu().float()
        infer_time_end=time.time();
        self.debug.rvc_time_use=infer_time_end-pitch_end_time
        self.debug.rvc_out_dim=infered_audio.shape

        #---- SOLA 音调变换
        self.output_wav_cache[-infered_audio.shape[0] :] = infered_audio  
        infer_wav = self.output_wav_cache[-self.crossfade_frame - self.sola_search_frame - self.block_frame :]
        cor_nom = torch.nn.functional.conv1d(input=infer_wav[None, None, : self.crossfade_frame + self.sola_search_frame],
                weight=self.sola_buffer[None, None, :],)
        cor_den = torch.sqrt(
            torch.nn.functional.conv1d(infer_wav[None, None, : self.crossfade_frame + self.sola_search_frame]** 2,
                torch.ones(1, 1, self.crossfade_frame, device=self.config.device),)+ 1e-8            )
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        
        self.output_wav[:] = infer_wav[sola_offset : sola_offset + self.block_frame]
        self.output_wav[: self.crossfade_frame] *= self.fade_in_window
        self.output_wav[: self.crossfade_frame] += self.sola_buffer[:]

        # crossfade
        if sola_offset < self.sola_search_frame:
                self.sola_buffer[:] = (
                    infer_wav[-self.sola_search_frame- self.crossfade_frame+ sola_offset : 
                              -self.sola_search_frame+ sola_offset]
                    * self.fade_out_window
                )
        else:
            self.sola_buffer[:] = (
                infer_wav[-self.crossfade_frame :] * self.fade_out_window
            )
                
        #---- 向扬声器输出声音 
        outdata= self.output_wav[:].cpu().numpy().astype(np.float32)          
        return (outdata,pyaudio.paContinue)
        
 

 

    def get_f0(self, x, f0_up_key, n_cpu, method="harvest"):
        
        if method == "crepe":
            return self.get_f0_crepe(x, f0_up_key)
        if method == "rmvpe":
            return self.get_f0_rmvpe(x, f0_up_key)
        if method == "pm":
            return self.get_f0_pm(x, f0_up_key);
        if method == "harvest":
            return self.get_f0_pm(x, f0_up_key);
        

    def get_f0_mel_hz(self, f0): 
        f0_min = self.f0_min=50;
        f0_max = self.f0_max=1100;
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak
     
  
    def get_f0_rmvpe(self, x, f0_up_key):
        if hasattr(self, "model_rmvpe") == False:
            from vvc.rmvpe import RMVPE
            self.model_rmvpe = RMVPE("../weights/rmvpe.pt", is_half=self.config.is_half, device=self.config.device)            
            print("完成加载基频提取模型")
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        frame_max_f0_value=max(f0)
        ratio=pow(2, f0_up_key / 12)
        max_f0_limit=1000; 
        if self.debug.max_f0 *ratio>=max_f0_limit:
            ratio=frame_max_f0_value/max_f0_limit;
        f0 = f0* ratio;
        self.debug.max_f0=frame_max_f0_value;
        #print(f"[P]{f0}")
        return self.get_f0_mel_hz(f0)
    #-----------------------
    #---- 停止声音捕获--------
    #----------------------
    def btnStopVoiceConvertClick(self):
        self.AudioConvertRunFlag=False 
        self.ui.sliderBlockTime.setEnabled(True)
        self.ui.spinBlockTime.setEnabled(True)
        self.ui.comboAudioCalcDevice.setEnabled(True)
        self.ui.comboInputAudioDevices.setEnabled(True)
        self.ui.comboOutputAudioDevices.setEnabled(True)
        self.ui.btnBrowseRvcModel.setEnabled(True)
        self.ui.lineEditRvcModelFile.setEnabled(True)
    
    def btnBrowseRvcModelClick(self):
        model_path,ok=QFileDialog.getOpenFileName(None,caption="Model File",directory=self.ui.lineEditRvcModelFile.text(),filter="pth(*.pth)");
        if ok:
            self.ui.lineEditRvcModelFile.setText(model_path);

    def btnBrowseAudioFileClick(self):
        pass


            
    def btnBrowseIndexModelClick(self):
        pass

    def OnSetVoiceConfigData(self): 
        self.config.convert_raw=self.ui.radioAudioOutputRaw.isChecked()
        self.config.convert_pitch=self.ui.radioAudioOutputPitch.isChecked()
        self.config.convert_ai=self.ui.radioAudioOutputAI.isChecked()
        self.config.threhold=self.ui.spinVoiceThreshold.value();
        self.config.f0_up_key=self.ui.spinPitchUp.value(); 
        self.config.block_time=self.ui.spinBlockTime.value()/1000.0;
        self.config.delay_time=self.ui.spinDelay.value()/1000.0;
        self.config.audio_volume=self.ui.spinAudioVolume.value()/10.0;

        self.ui.sliderVoiceThreshold.setValue(self.config.threhold)
        self.ui.sliderPitchUp.setValue(self.config.f0_up_key)
        self.ui.sliderBlockTime.setValue(self.ui.spinBlockTime.value())
        self.ui.sliderDelay.setValue(self.ui.spinDelay.value())
        self.ui.sliderAudioVolume.setValue(self.ui.spinAudioVolume.value())

        
    def OnSlideVoiceConfig(self):
        self.ui.spinVoiceThreshold.setValue(self.ui.sliderVoiceThreshold.value())
        self.ui.spinPitchUp.setValue(self.ui.sliderPitchUp.value())
        self.ui.spinBlockTime.setValue(self.ui.sliderBlockTime.value())
        self.ui.spinDelay.setValue(self.ui.sliderDelay.value())
        self.ui.spinAudioVolume.setValue(self.ui.sliderAudioVolume.value())

    def OnChangeAudioDevice(self):
        in_device_name=self.ui.comboInputAudioDevices.currentText()
        out_device_name=self.ui.comboOutputAudioDevices.currentText()
        in_arr=in_device_name.split("|")
        out_arr=out_device_name.split("|")
        self.config.in_device_idx=int(in_arr[0]);
        self.config.out_device_idx=int(out_arr[0])
        #print(f"in_device_idx:{self.config.in_device_idx},out_device_idx:{self.config.out_device_idx}")

    def ChooseDefaultDevices(self):
        #print("默认HostAPI:",self.p.get_default_host_api_info());
        if self.p.get_default_input_device_info() is None:
            return
        if self.p.get_default_output_device_info() is None:
            return;

        default_in_idx=self.p.get_default_input_device_info().get("index",0)
        default_out_idx=self.p.get_default_output_device_info().get("index",0)
        for i in range(self.ui.comboInputAudioDevices.count()):
            text=self.ui.comboInputAudioDevices.itemText(i)
            idx=int(text.split("|")[0])
            if default_in_idx==idx:
                self.ui.comboInputAudioDevices.setCurrentIndex(i)

        for i in range(self.ui.comboOutputAudioDevices.count()):
            text=self.ui.comboOutputAudioDevices.itemText(i)
            idx=int(text.split("|")[0])
            if default_out_idx==idx:
                self.ui.comboOutputAudioDevices.setCurrentIndex(i)
                 