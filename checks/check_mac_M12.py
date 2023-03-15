
import platform
import warnings

use_linalg = False
mac_arch = platform.machine()
if mac_arch == 'arm64':

    with warnings.catch_warnings():
        Warningskill.filterwarnings("ignore",category=DeprecationWarning)
        import numpy.distutils.system_info as sysinfo
        info = sysinfo.get_info('accelerate')
        if info is not None and len(info)>0:
            for x in info['extra_link_args']:
                if 'Accelerate' in x:
                    use_linalg = True
         
print(f"Accelerate found? , use linalg={use_linalg}")




    
    
