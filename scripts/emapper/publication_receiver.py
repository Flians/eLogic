"""Linux commit notifications with loss-aware scan fallback; no netlist authority."""
import ctypes,json,os,select,struct,time
from pathlib import Path

class TimingTrace:
 def __init__(self,path,deadline):self.path=Path(path);self.deadline=deadline
 def emit(self,event,monotonic=None,**fields):
  record=dict(event=event,clock='CLOCK_MONOTONIC',monotonic=time.monotonic()if monotonic is None else monotonic,deadline_monotonic=self.deadline,**fields)
  with self.path.open('a')as out:out.write(json.dumps(record,separators=(',',':'))+'\n')

class Receiver:
 def __init__(self,root,mode):
  if mode not in ('scan','events'):raise ValueError('unknown publication receiver')
  self.root=Path(root);self.mode=mode;self.fd=None;self.ready={};self.rescan=True;self.rescans=0;self.events=0;self.fallbacks=[];self.service_seconds=0.;self.wait_seconds=0.
  if mode=='events':
   try:
    lib=ctypes.CDLL(None,use_errno=True)
    lib.inotify_init1.argtypes=[ctypes.c_int];lib.inotify_init1.restype=ctypes.c_int
    lib.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32];lib.inotify_add_watch.restype=ctypes.c_int
    self.fd=lib.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
    if self.fd<0:self.fd=None;raise OSError(ctypes.get_errno(),'inotify_init1')
    # IN_MOVED_TO: native commits by rename; DELETE_SELF/MOVE_SELF/IGNORED force fallback.
    if lib.inotify_add_watch(self.fd,os.fsencode(self.root),0x80|0x400|0x800)<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
   except (OSError,AttributeError)as exc:
    self.close();self.fallbacks.append(repr(exc))
 def __enter__(self):return self
 def __exit__(self,*unused):self.close()
 def close(self):
  if self.fd is not None:os.close(self.fd);self.fd=None
 def invalidate(self,reason):self.rescan=True;self.fallbacks.append(reason)
 def drain(self):
  if self.fd is None:return
  while True:
   try:data=os.read(self.fd,65536)
   except BlockingIOError:break
   except OSError as exc:self.invalidate(repr(exc));self.close();break
   if not data:self.invalidate('empty-inotify-read');self.close();break
   pos=0
   while pos<len(data):
    if len(data)-pos<16:self.invalidate('truncated-event');break
    wd,mask,cookie,size=struct.unpack_from('iIII',data,pos);pos+=16
    if pos+size>len(data):self.invalidate('truncated-name');break
    name=os.fsdecode(data[pos:pos+size].split(b'\0',1)[0]);pos+=size
    if mask&0x4000:self.invalidate('queue-overflow')
    if mask&(0x400|0x800|0x8000):self.invalidate('watch-invalidated');self.close()
    if mask&0x80 and len(name)==8 and name.isascii()and name.isdecimal():self.ready[name]=self.root/name;self.events+=1
   if self.fd is None:break
 def folders(self):
  started=time.monotonic()
  self.drain()
  if self.mode=='scan'or self.fd is None or self.rescan:
   self.ready.update((p.name,p)for p in self.root.glob('[0-9]'*8));self.rescans+=1;self.rescan=False
  folders=[self.ready[k]for k in sorted(self.ready,reverse=True)];self.ready.clear();self.service_seconds+=time.monotonic()-started;return folders
 def retry(self,folders,processed):
  self.ready.update((p.name,p)for p in folders if int(p.name)not in processed)
 def wait(self,seconds):
  started=time.monotonic()
  if self.fd is None:time.sleep(seconds)
  else:select.select([self.fd],[],[],seconds)
  self.wait_seconds+=time.monotonic()-started
 def report(self):return dict(requested=self.mode,effective='events'if self.fd is not None else'scan',rescans=self.rescans,events=self.events,fallbacks=self.fallbacks,service_seconds=self.service_seconds,wait_seconds=self.wait_seconds)
