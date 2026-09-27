import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import struct, mmap, time, threading, queue
import numpy as np

EDGE_DTYPE=np.dtype([('src','<u4'),('dst','<u4'),('weight','<f4')])
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    GPU_AVAILABLE=True
except Exception:
    cp=None; cpsp=None; GPU_AVAILABLE=False
try:
    import scipy.sparse as sps
except Exception:
    sps=None

class ConnectivityLoader:
    def __init__(self,edge_path,points_path): self.edge_path=Path(edge_path); self.points_path=Path(points_path)
    def load_points(self):
        a=np.load(self.points_path,mmap_mode='r')
        if a.ndim!=2 or a.shape[1]<2: raise ValueError('points must be N x 2 or N x 3')
        a=np.asarray(a[:,:3],dtype=np.float32)
        if a.shape[1]==2: a=np.column_stack((a,np.zeros(len(a),np.float32)))
        return np.ascontiguousarray(a)
    def load_edges(self,n):
        size=self.edge_path.stat().st_size
        rec=EDGE_DTYPE.itemsize
        usable=size-(size%rec)
        if usable<=0:return np.empty(0,np.int32),np.empty(0,np.int32),np.empty(0,np.float32)
        with self.edge_path.open('rb') as f, mmap.mmap(f.fileno(),usable,access=mmap.ACCESS_READ) as mm:
            a=np.frombuffer(mm,dtype=EDGE_DTYPE,count=usable//rec).copy()  # .copy() breaks the mmap buffer export so mm.close() below doesn't raise BufferError
            src=np.asarray(a['src'],np.uint32); dst=np.asarray(a['dst'],np.uint32); w=np.asarray(a['weight'],np.float32)
            mask=(src<n)&(dst<n)&np.isfinite(w)&(w!=0)
            return src[mask].astype(np.int32,copy=True),dst[mask].astype(np.int32,copy=True),w[mask].astype(np.float32,copy=True)

class Network:
    def __init__(self,p,e,gpu=True):
        self.points=p; self.src,self.dst,self.weight=e; self.n=len(p); self.device='CPU'; self.xp=np
        if gpu and GPU_AVAILABLE:
            try:
                rows=cp.asarray(self.dst); cols=cp.asarray(self.src); vals=cp.asarray(self.weight)
                self.M=cpsp.csr_matrix((vals,(rows,cols)),shape=(self.n,self.n),dtype=cp.float32)
                self.xp=cp; self.device='CUDA'; cp.cuda.Stream.null.synchronize()
            except Exception as ex: print('CUDA:',ex); self._cpu()
        else:self._cpu()
        self.reset()
    def _cpu(self):
        if sps is None: raise RuntimeError('Install scipy or CuPy')
        self.M=sps.csr_matrix((self.weight,(self.dst,self.src)),shape=(self.n,self.n),dtype=np.float32)
        self.xp=np; self.device='CPU'
    def reset(self):
        x=self.xp; self.t=0; self.v=x.full(self.n,-65,dtype=x.float32); self.ref=x.zeros(self.n,dtype=x.int16); self.fired=np.empty(0,np.int32); self.step_ms=0
        if self.device=='CUDA': cp.cuda.Stream.null.synchronize()
    def step(self,current=18,source=0):
        x=self.xp; t0=time.perf_counter(); active=self.ref<=0
        inj=x.zeros(self.n,dtype=x.float32); inj[source]=current if self.n else 0
        incoming=self.M@inj
        self.v=x.where(active,self.v-(self.v+65)/20+incoming,self.v)
        fg=active&(self.v>=-50)
        fired=cp.asnumpy(cp.flatnonzero(fg)).astype(np.int32) if self.device=='CUDA' else np.flatnonzero(fg).astype(np.int32)
        if fired.size: self.v[fg]=-70; self.ref[fg]=5
        self.ref=x.maximum(self.ref-1,0); self.fired=fired; self.t+=1
        if self.device=='CUDA': cp.cuda.Stream.null.synchronize()
        self.step_ms=(time.perf_counter()-t0)*1000
    def snapshot(self,indices):
        if self.device=='CUDA': return cp.asnumpy(self.v[indices]),cp.asnumpy(self.ref[indices]),self.fired.copy(),self.t,self.step_ms
        return self.v[indices].copy(),self.ref[indices].copy(),self.fired.copy(),self.t,self.step_ms
    def info(self):
        if self.device!='CUDA':return 'CPU'
        try:
            p=cp.cuda.runtime.getDeviceProperties(0); n=p['name'].decode() if isinstance(p['name'],bytes) else str(p['name']); return n
        except:return 'CUDA GPU'

def demo():
    rng=np.random.default_rng(4); n=12000; p=rng.normal(size=(n,3)).astype(np.float32); p/=np.linalg.norm(p,axis=1,keepdims=True)+1e-8; p*=180
    s=np.arange(n,dtype=np.int32); parts=[]
    for d in range(1,4): parts.extend([(s,(s+d)%n),( (s+d)%n,s)])
    src=np.concatenate([a for a,b in parts]); dst=np.concatenate([b for a,b in parts]); w=rng.uniform(.1,.5,len(src)).astype(np.float32)
    return p,(src,dst,w)

class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title('GPU Neural Circuit Simulation'); self.geometry('1100x760'); self.running=False; self.net=None; self.q=queue.Queue(); self.worker=None; self.load_worker=None
        self.current=tk.DoubleVar(value=18); self.speed=tk.DoubleVar(value=1); self.render_n=600; self.indices=np.empty(0,np.int32); self.photo=None; self.last_render=0
        top=ttk.Frame(self); top.pack(fill='x',padx=8,pady=8)
        ttk.Button(top,text='Load connectivity',command=self.choose).pack(side='left'); self.btn=ttk.Button(top,text='Start',command=self.toggle); self.btn.pack(side='left',padx=4); ttk.Button(top,text='Reset',command=self.reset).pack(side='left'); ttk.Button(top,text='Pulse',command=self.pulse).pack(side='left',padx=4)
        ttk.Label(top,text='Current').pack(side='left',padx=(15,3)); ttk.Scale(top,from_=1,to=50,variable=self.current,orient='horizontal',length=120).pack(side='left'); ttk.Label(top,text='Speed').pack(side='left',padx=(15,3)); ttk.Scale(top,from_=.25,to=4,variable=self.speed,orient='horizontal',length=100).pack(side='left')
        self.info_label=ttk.Label(top,text='initializing'); self.info_label.pack(side='left',padx=15)
        self.canvas=tk.Canvas(self,bg='#070a10',highlightthickness=0); self.canvas.pack(fill='both',expand=True); self.canvas.bind('<Configure>',self.on_canvas_resize); self.status=ttk.Label(self,text='Loading demo...',anchor='w'); self.status.pack(fill='x')
        self.protocol('WM_DELETE_WINDOW',self.close); self.after(50,self.commands); self.after(80,self.render_loop); threading.Thread(target=self.make_demo,daemon=True).start()
    def on_canvas_resize(self,event=None):
        # Canvas starts at Tk's placeholder 1x1 size until the window is
        # actually mapped. The demo network often finishes loading before
        # that happens, so draw_static() run at load time can compute
        # positions against a 1x1 canvas and collapse every point into a
        # sub-pixel clump. Redraw whenever the canvas gets its real size
        # (or is resized later) so the points always use current geometry.
        if self.net is not None: self.draw_static()
    def make_demo(self):
        try:self.q.put(('net',Network(*demo(),gpu=True)))
        except Exception as e:self.q.put(('err',str(e)))
    def choose(self):
        ep=filedialog.askopenfilename(title='connectivity_edges.bin',filetypes=[('Binary','*.bin')]);
        if not ep:return
        pp=filedialog.askopenfilename(title='connectivity_points.npy',filetypes=[('NumPy','*.npy')]);
        if not pp:return
        self.running=False; self.btn.config(text='Start'); self.info_label.config(text='loading...'); self.status.config(text='Reading connectivity...')
        threading.Thread(target=self.load,args=(ep,pp),daemon=True).start()
    def load(self,ep,pp):
        try:
            l=ConnectivityLoader(ep,pp); p=l.load_points(); e=l.load_edges(len(p)); self.q.put(('net',Network(p,e,True)))
        except Exception as e:self.q.put(('err',repr(e)))
    def commands(self):
        try:
            while 1:
                typ,obj=self.q.get_nowait()
                if typ=='net':
                    self.running=False; self.net=obj; self.indices=np.linspace(0,self.net.n-1,min(self.render_n,self.net.n),dtype=np.int32); self.canvas.delete('all'); self.info_label.config(text=f'{obj.device}: {obj.info()}'); self.status.config(text=f'{obj.n:,} nodes / {len(obj.src):,} edges'); self.draw_static()
                else: messagebox.showerror('Error',obj)
        except queue.Empty:pass
        self.after(50,self.commands)
    def draw_static(self):
        self.canvas.delete('all'); w=max(1,self.canvas.winfo_width()); h=max(1,self.canvas.winfo_height()); p=self.net.points[self.indices]; span=max(float(np.ptp(p[:,:2],axis=0).max()),1); sc=min(w,h)/(2.4*span); x=w/2+p[:,0]*sc; y=h/2-p[:,1]*sc
        self.xy=np.column_stack((x,y)); self.canvas.create_text(10,10,anchor='nw',text='GPU simulation — display sample only',fill='#8aa0b5');
        for a,b in self.xy.astype(int): self.canvas.create_oval(a-2,b-2,a+2,b+2,fill='#34485d',outline='')
    def toggle(self):
        if self.net is None:return
        self.running=not self.running; self.btn.config(text='Pause' if self.running else 'Start')
        if self.running and (self.worker is None or not self.worker.is_alive()): self.worker=threading.Thread(target=self.sim_loop,daemon=True); self.worker.start()
    def sim_loop(self):
        # No sleep here previously: this tight loop held the GIL almost
        # continuously, starving the main thread's Tk after()/mainloop
        # callbacks -- the window stopped repainting/responding even
        # though net.step() kept advancing in the background. A short
        # sleep each outer iteration yields the GIL regularly so the
        # GUI keeps rendering, without meaningfully slowing simulation.
        while self.running and self.net:
            for _ in range(max(1,int(float(self.speed.get())*3))):
                if not self.running:break
                self.net.step(float(self.current.get()))
            time.sleep(0.001)
    def reset(self):
        self.running=False; self.btn.config(text='Start')
        if self.net:self.net.reset()
    def pulse(self):
        if self.net:self.net.step(float(self.current.get()))
    def render_loop(self):
        if self.net:self.render()
        self.after(80,self.render_loop)
    def render(self):
        # Intentionally do NOT update hundreds of Tk canvas objects every frame.
        if not hasattr(self,'xy') or self.net is None:return
        v,r,f,t,ms=self.net.snapshot(self.indices); fired=set(f.tolist());
        # Only redraw fired points and a tiny activity sample; this keeps Tk responsive.
        self.canvas.delete('active');
        for i,idx in enumerate(self.indices):
            if int(idx) in fired:
                x,y=self.xy[i]; self.canvas.create_oval(x-5,y-5,x+5,y+5,fill='#ff2050',outline='',tags='active')
        self.status.config(text=f'{self.net.device} | nodes {self.net.n:,} | edges {len(self.net.src):,} | t {t:,} | firing {len(f):,} | step {ms:.3f} ms')
    def close(self):self.running=False; self.destroy()

if __name__=='__main__':App().mainloop()
