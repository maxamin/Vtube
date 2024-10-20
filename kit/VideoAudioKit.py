#coding=utf-8
# cython:language_level=3
import threading,os,ffmpeg
from kit import ShareData as sd

def mux_video_audio(video_path="",audio_path="",out_path="",wait_time=0.5,mode=0):
    if os.path.exists(video_path) is False:
        print("[P]video path not exists",end='\r')
        return
    if audio_path=="":
        audio_path=os.path.splitext(video_path)[0]+".wav"
        print(f"[P]auto generate audio path :{audio_path}",end='\r')
    if  os.path.exists(audio_path) is False:
        print("audio  path not exists")
        return

    out_path=out_path+"_merge.mp4"
    print(f"[P]output merge:{out_path}",end='\r')
    if os.path.exists(sd.ffmpeg_path) is False:
        print(f"ffmpeg file {sd.ffmpeg_path} not exists")
        return

    if mode==0:
        cmd=f"{sd.ffmpeg_path} -i {video_path} -i {audio_path} {out_path} -loglevel quiet"
        os.system(cmd)
    if mode==1:
        audio = ffmpeg.input(filename=audio_path)
        video = ffmpeg.input(filename=video_path)
        out = ffmpeg.output(video, audio, out_path)
        process=out.overwrite_output()
        process.run_async(cmd=sd.ffmpeg_path, pipe_stdin=True, pipe_stdout=False,
                                                pipe_stderr=False)
    print("[P]完成音视频组合，输出{out_path}",end='\r')

#VideoAudioKit.mux_video_audio(r"F:\SwapRecord\20230326_144350.avi")