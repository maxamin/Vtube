import numpy as np,cv2,time
import kit.Auth as auth 
import wmi


w = wmi.WMI()
time_start_token=time.time()
cpu_id=w.Win32_Processor()[0].ProcessorId
bios_info=w.Win32_BIOS()[0].Version
os_sn=w.Win32_OperatingSystem()[0].SerialNumber
merge=cpu_id+bios_info+os_sn
machineSN=auth.get_md5_hash_str(merge).upper()
print("merge",merge)
print("machineSN",machineSN)

time_end=time.time()
print(time_end-time_start_token)