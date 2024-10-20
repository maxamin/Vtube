import os,sys,traceback,queue,threading,time,itertools,numpy as np,cv2
from pathlib import Path
from core import pathex,imagelib
from core import imagelib
import core.swapmodels as models
from core.swapmodels.Model_SAEHD.SAEHDModel import SAEHDModel
from core.interact import interact as io

train_model=None; train_pause=False;
s2c = queue.Queue()
c2s = queue.Queue()
e=threading.Event()

def trainerThread (s2c, c2s, e,
                    model_class_name = None,saved_models_path = None,training_data_src_path = None,
                    training_data_dst_path = None,pretraining_data_path = None,pretrained_model_path = None,
                    no_preview=False,force_model_name=None,force_gpu_idxs=None,
                    cpu_only=None,silent_start=True,execute_programs = None,debug=False,**kwargs):

    global train_model,train_pause;
    try:
        start_time = time.time()

        save_interval_min = 30

        if not training_data_src_path.exists():
            training_data_src_path.mkdir(exist_ok=True, parents=True)

        if not training_data_dst_path.exists():
            training_data_dst_path.mkdir(exist_ok=True, parents=True)

        if not saved_models_path.exists():
            saved_models_path.mkdir(exist_ok=True, parents=True)
            
        model=SAEHDModel(
                    is_training=True,saved_models_path=saved_models_path,
                    training_data_src_path=training_data_src_path,training_data_dst_path=training_data_dst_path,
                    pretraining_data_path=pretraining_data_path,pretrained_model_path=pretrained_model_path,
                    no_preview=no_preview,force_model_name=force_model_name,force_gpu_idxs=force_gpu_idxs,
                    cpu_only=cpu_only,silent_start=silent_start,debug=debug,force_ui_options=kwargs)
        train_model=model;
        is_reached_goal = train_model.is_reached_iter_goal()

        shared_state = { 'after_save' : False }
        loss_string = ""
        save_iter =  model.get_iter()

        def model_save():
            if not debug and not is_reached_goal:
                io.log_info ("[P]Saving Model....", end='\r')
                model.save()
                shared_state['after_save'] = True
                    
        def model_backup():
            if not debug and not is_reached_goal:
                model.create_backup()             

        def send_preview():
            previews = model.get_previews()
            c2s.put ( {'op':'show', 'previews': previews, 'iter':model.get_iter(), 'loss_history': model.get_loss_history().copy() } )
            e.set()  

        def test_preview():
            #io.log_info('receive test preview op.')
            ok,test_previews=model.get_test_preview()
            if ok is False: return
            c2s.put ( {'op':'show', 'previews': test_previews, 'iter':model.get_iter(), 'loss_history': model.get_loss_history().copy() } )
            e.set()
            

        if model.get_target_iter() != 0:
            if is_reached_goal:
                io.log_info('Model already trained to target iteration. You can use preview.')
            else:
                io.log_info('Starting. Target iteration: %d. Press "Enter" to stop training and save model.' % ( model.get_target_iter()  ) )
        else:
            io.log_info('开始执行模型训练')

        last_save_time = time.time()

        #--- 第一次运行的提示输出
        if model.get_iter() == 0:
            io.log_info("")
            io.log_info("Trying to do the first iteration. If an error occurs, reduce the model parameters.")
            io.log_info("")

        for i in itertools.count(0,1):

            if  is_reached_goal is True:
                io.log_info("reach target iteration num")
                #break;

            if train_pause is True:
                time.sleep(0.5)
                continue;

            cur_time = time.time()
                    
            iter, iter_time = model.train_one_iter()

            loss_history = model.get_loss_history()
            time_str = time.strftime("[%H:%M:%S]")
            if iter_time >= 10:
                loss_string = "{0}[#{1:06d}][{2:.5s}s]".format ( time_str, iter, '{:0.4f}'.format(iter_time) )
            else:
                loss_string = "{0}[#{1:06d}][{2:04d}ms]".format ( time_str, iter, int(iter_time*1000) )

            #--- 输出 Loss history
            if shared_state['after_save']:
                shared_state['after_save'] = False
                loss_history_count=len(loss_history)
                loss_history_span=iter-save_iter;
                mean_loss=0.0;
                if loss_history_span<loss_history_count:
                    mean_loss = np.mean ( loss_history[-loss_history_span:-1], axis=0)
                else:
                    mean_loss = np.mean ( loss_history[-100:-1], axis=0)
                for loss_value in mean_loss:
                    loss_string += "[%.4f]" % (loss_value)

                io.log_info (loss_string)

                save_iter = iter
            else:
                for loss_value in loss_history[-1]:
                    loss_string += "[%.4f]" % (loss_value)
            io.log_info ("[T]"+loss_string, end='\r')


            #--- 判断是否到达设定迭代数
            if model.get_target_iter() != 0 and model.is_reached_iter_goal():
                io.log_info ('Reached target iteration.')
                model_save()
                is_reached_goal = True
                io.log_info ('You can use preview now.')
            
            #--- 判断是否保存模型
            need_save = False
            while time.time() - last_save_time >= save_interval_min*60:
                last_save_time += save_interval_min*60
                need_save = True
                
            if not is_reached_goal and need_save:
                model_save()
                send_preview()

            if i==0:
                if is_reached_goal:
                    model.pass_one_iter()
                send_preview()

            if debug:
                time.sleep(0.005)

            while not s2c.empty():
                input = s2c.get()
                op = input['op']
                if op == 'save':
                    model_save()
                elif op == 'backup':
                    model_backup()
                elif op == 'preview':
                    if is_reached_goal:
                        model.pass_one_iter()
                    send_preview()
                elif op == 'test':
                    test_preview()
                elif op == 'close':
                    model_save()
                    i = -1
                    break

            if i == -1:
                break



        model.finalize()

    except Exception as e:
        print ('Error: %s' % (str(e)))
        traceback.print_exc()
        
    c2s.put ( {'op':'close'} )
    print("结束模型训练任务线程")
    train_model=None;



def StartTrainThread(**kwargs):
    io.log_info ("Running trainer.\r\n")

    no_preview = kwargs.get('no_preview', False)
    global s2c,c2s,e
    
    thread = threading.Thread(target=trainerThread, args=(s2c, c2s, e), kwargs=kwargs,name="TrainModel" )
    thread.start()
    print("训练线程已经启动")


    print("TrainMain主线程开始等待训练线程加载模型")
    e.wait() #Wait for inital load to occur.
    #print("主线程等待结束，开始执行")

    if no_preview:
        while True:
            if not c2s.empty():
                input = c2s.get()
                op = input.get('op','')
                if op == 'close':
                    break
            try:
                io.process_messages(0.1)
            except KeyboardInterrupt:
                s2c.put ( {'op': 'close'} )
    else:
        wnd_name = "Training preview"
        io.named_window(wnd_name)
        io.capture_keys(wnd_name)

        previews = None
        loss_history = None
        selected_preview = 0
        update_preview = False
        is_showing = False
        is_waiting_preview = False
        show_last_history_iters_count = 0
        iter = 0
        while True:
            if not c2s.empty():
                input = c2s.get()
                op = input['op']
                if op == 'show':
                    is_waiting_preview = False
                    loss_history = input['loss_history'] if 'loss_history' in input.keys() else None
                    previews = input['previews'] if 'previews' in input.keys() else None
                    iter = input['iter'] if 'iter' in input.keys() else 0
                    if previews is not None:
                        max_w = 0
                        max_h = 0
                        for (preview_name, preview_rgb) in previews:
                            (h, w, c) = preview_rgb.shape
                            max_h = max (max_h, h)
                            max_w = max (max_w, w)

                        max_size = 800
                        if max_h > max_size:
                            max_w = int( max_w / (max_h / max_size) )
                            max_h = max_size

                        #make all previews size equal
                        for preview in previews[:]:
                            (preview_name, preview_rgb) = preview
                            (h, w, c) = preview_rgb.shape
                            if h != max_h or w != max_w:
                                previews.remove(preview)
                                previews.append ( (preview_name, cv2.resize(preview_rgb, (max_w, max_h))) )
                        selected_preview = selected_preview % len(previews)
                        update_preview = True
                elif op == 'close':
                    break

            if update_preview:
                update_preview = False

                selected_preview_name = previews[selected_preview][0]
                selected_preview_rgb = previews[selected_preview][1]
                (h,w,c) = selected_preview_rgb.shape

                # HEAD
                head_lines = [
                    '[s]:save [b]:backup [enter]:exit',
                    '[p]:update [space]:next preview [l]:change history range',
                    'Preview: "%s" [%d/%d]' % (selected_preview_name,selected_preview+1, len(previews) )
                    ]
                head_line_height = 15
                head_height = len(head_lines) * head_line_height
                head = np.ones ( (head_height,w,c) ) * 0.1

                for i in range(0, len(head_lines)):
                    t = i*head_line_height
                    b = (i+1)*head_line_height
                    head[t:b, 0:w] += imagelib.get_text_image (  (head_line_height,w,c) , head_lines[i], color=[0.8]*c )

                final = head

                if loss_history is not None:
                    if show_last_history_iters_count == 0:
                        loss_history_to_show = loss_history
                    else:
                        loss_history_to_show = loss_history[-show_last_history_iters_count:]

                    lh_img = models.ModelBase.get_loss_history_preview(loss_history_to_show, iter, w, c)
                    final = np.concatenate ( [final, lh_img], axis=0 )

                final = np.concatenate ( [final, selected_preview_rgb], axis=0 )
                final = np.clip(final, 0, 1)

                io.show_image( wnd_name, (final*255).astype(np.uint8) )
                is_showing = True

            key_events = io.get_key_events(wnd_name)
            key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0,0,False,False,False)

            if key == ord('\n') or key == ord('\r'):
                s2c.put ( {'op': 'close'} )
            elif key == ord('s'):
                s2c.put ( {'op': 'save'} )
            elif key == ord('b'):
                s2c.put ( {'op': 'backup'} )
            elif key == ord('p'):
                if not is_waiting_preview:
                    is_waiting_preview = True
                    s2c.put ( {'op': 'preview'} )
            elif key == ord('l'):
                if show_last_history_iters_count == 0:
                    show_last_history_iters_count = 5000
                elif show_last_history_iters_count == 5000:
                    show_last_history_iters_count = 10000
                elif show_last_history_iters_count == 10000:
                    show_last_history_iters_count = 50000
                elif show_last_history_iters_count == 50000:
                    show_last_history_iters_count = 100000
                elif show_last_history_iters_count == 100000:
                    show_last_history_iters_count = 0
                update_preview = True
            elif key == ord(' '):
                selected_preview = (selected_preview + 1) % len(previews)
                update_preview = True

            try:
                io.process_messages(0.1)
            except KeyboardInterrupt:
                s2c.put ( {'op': 'close'} )

        io.destroy_all_windows()
        print("Train Main Preview Thread End \n")

def ExprtDFM(saved_models_path):
    model=SAEHDModel(is_exporting=True,saved_models_path=saved_models_path,cpu_only=True)
    model.export_dfm () 