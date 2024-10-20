import cv2
import time
from tqdm import tqdm
import numpy as np
import onnxruntime as rt
 
 
class Matting:
    def __init__(self, model_path='onnx_model\modnet.onnx', input_size=(512, 512)):
        self.model_path = model_path
        ep_flags = {'device_id':0}
        self.sess = rt.InferenceSession(self.model_path,providers=[ ("CUDAExecutionProvider", ep_flags) ])
        self.input_name = self.sess.get_inputs()[0].name
        self.label_name = self.sess.get_outputs()[0].name
        self.input_size = input_size
        self.txt_font = cv2.FONT_HERSHEY_PLAIN
 
    def normalize(self, im, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        im = im.astype(np.float32, copy=False) / 255.0
        im -= mean
        im /= std
        return im
 
    def resize(self, im, target_size=608, interp=cv2.INTER_LINEAR):
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            w = target_size[0]
            h = target_size[1]
        else:
            w = target_size
            h = target_size
        im = cv2.resize(im, (w, h), interpolation=interp)
        return im
 
    def preprocess(self, image, target_size=(512, 512), interp=cv2.INTER_LINEAR):
        image = self.normalize(image)
        image = self.resize(image, target_size=target_size, interp=interp)
        image = np.transpose(image, [2, 0, 1])
        image = image[None, :, :, :]
        return image
 
    def predict_frame(self, bgr_image):
        assert len(bgr_image.shape) == 3, "Please input RGB image."
        raw_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w, c = raw_image.shape
        image = self.preprocess(raw_image, target_size=self.input_size)
 
        pred = self.sess.run(
            [self.label_name],
            {self.input_name: image.astype(np.float32)}
        )[0]
        pred = pred[0, 0]
        matte_np = self.resize(pred, target_size=(w, h), interp=cv2.INTER_NEAREST)
        matte_np = np.expand_dims(matte_np, axis=-1)
        return matte_np
 
    def predict_image(self, source_image_path, save_image_path):
        bgr_image = cv2.imread(source_image_path)
        assert len(bgr_image.shape) == 3, "Please input RGB image."
        matte_np = self.predict_frame(bgr_image)
        matting_frame = matte_np * bgr_image + (1 - matte_np) * np.full(bgr_image.shape, 255.0)
        matting_frame = matting_frame.astype('uint8')
        cv2.imwrite(save_image_path, matting_frame)
 
    def predict_camera(self):
        cap_video = cv2.VideoCapture(0)
        if not cap_video.isOpened():
            raise IOError("Error opening video stream or file.")
        
        count = 0
        while cap_video.isOpened():
            ret, raw_frame = cap_video.read()
            if ret:
                beg = time.time()
                count += 1
                matte_np = self.predict_frame(raw_frame)
                matting_frame = matte_np * raw_frame + (1 - matte_np) * np.full(raw_frame.shape, 255.0)
                matting_frame = matting_frame.astype('uint8')
 
                end = time.time()
                fps = round(count / (end - beg), 2)
                if count >= 50:
                    count = 0
                    beg = end
 
                cv2.putText(matting_frame, "fps: " + str(fps), (20, 20), self.txt_font, 2, (0, 0, 255), 1)
 
                cv2.imshow('Matting', matting_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap_video.release()
        cv2.destroyWindow()
 
    def check_video(self, src_path, dst_path):
        cap1 = cv2.VideoCapture(src_path)
        fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
        number_frames1 = cap1.get(cv2.CAP_PROP_FRAME_COUNT)
        cap2 = cv2.VideoCapture(dst_path)
        fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
        number_frames2 = cap2.get(cv2.CAP_PROP_FRAME_COUNT)
        assert fps1 == fps2 and number_frames1 == number_frames2, "fps or number of frames not equal."
 
    def predict_video(self, video_path, save_path, threshold=2e-7):
        # 使用odf策略
        time_beg = time.time()
        pre_t2 = None  # 前2步matte
        pre_t1 = None  # 前1步matte
 
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        number_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("source video fps: {}, video resolution: {}, video frames: {}".format(fps, size, number_frames))
        videoWriter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
 
        ret, frame = cap.read()
        with tqdm(range(int(number_frames))) as t:
            for c in t:
                matte_np = self.predict_frame(frame)
                if pre_t2 is None:
                    pre_t2 = matte_np
                elif pre_t1 is None:
                    pre_t1 = matte_np
                    # 第一帧写入
                    matting_frame = pre_t2 * frame + (1 - pre_t2) * np.full(frame.shape, 255.0)
                    videoWriter.write(matting_frame.astype('uint8'))
                else:
                    # odf
                    error_interval = np.mean(np.abs(pre_t2 - matte_np))
                    error_neigh = np.mean(np.abs(pre_t1 - pre_t2))
                    if error_interval < threshold < error_neigh:
                        pre_t1 = pre_t2
 
                    matting_frame = pre_t1 * frame + (1 - pre_t1) * np.full(frame.shape, 255.0)
                    videoWriter.write(matting_frame.astype('uint8'))
                    pre_t2 = pre_t1
                    pre_t1 = matte_np
 
                ret, frame = cap.read()
            # 最后一帧写入
            matting_frame = pre_t1 * frame + (1 - pre_t1) * np.full(frame.shape, 255.0)
            videoWriter.write(matting_frame.astype('uint8'))
            cap.release()
        print("video matting over, time consume: {}, fps: {}".format(time.time() - time_beg, number_frames / (time.time() - time_beg)))
 
 
if __name__ == '__main__':
    import set_env
    set_env.set_env()
    model = Matting(model_path=r'..\weights\modnet_photographic_portrait_matting.onnx', input_size=(512, 512))
    model.predict_camera()
    #model.predict_image('images\\1.jpeg', 'output\\1.png')
    #model.predict_image('images\\2.jpeg', 'output\\2.png')
    #model.predict_image('images\\3.jpeg', 'output\\3.png')
    #model.predict_image('images\\4.jpeg', 'output\\4.png')
    # model.predict_video("video\dance.avi", "output\dance_matting.avi")
 
