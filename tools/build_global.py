import io, base64, json, math, os, urllib.request, concurrent.futures as cf
import numpy as np
from PIL import Image
os.environ.setdefault("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
C="/home/claude/build/cache"
def get(z,x,y):
    p=f"{C}/terr_{z}_{x}_{y}.png"
    if not os.path.exists(p):
        open(p,"wb").write(urllib.request.urlopen(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",timeout=30).read())
    a=np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
    return a[...,0]*256+a[...,1]+a[...,2]/256-32768
z=4; n=2**z
with cf.ThreadPoolExecutor(16) as ex:
    tiles=list(ex.map(lambda j:get(z,j%n,j//n), range(n*n)))
mos=np.zeros((n*256,n*256),np.float32)
for j,t in enumerate(tiles):
    y,x=divmod(j,n); mos[y*256:(y+1)*256,x*256:(x+1)*256]=t
W,H=720,360
lat=90-(np.arange(H)+.5)*180/H; lon=-180+(np.arange(W)+.5)*360/W
latc=np.clip(lat,-85.05,85.05)
my=(1-np.log(np.tan(np.radians(latc))+1/np.cos(np.radians(latc)))/math.pi)/2*n*256
mx=(lon+180)/360*n*256
E=mos[np.clip(my.astype(int),0,n*256-1)[:,None], np.clip(mx.astype(int),0,n*256-1)[None,:]]
E=np.clip(E,-11000,9000).astype("<i2")
b64=base64.b64encode(E.tobytes()).decode()
# blue marble 2048 with ocean darkening matching place packs
bm=Image.open("/home/claude/dl/package/example/img/earth-blue-marble.jpg").convert("RGB").resize((2048,1024),Image.LANCZOS)
a=np.clip(np.asarray(bm).astype(np.float32)*1.15,0,255)
Eb=np.asarray(Image.fromarray(E.astype(np.float32)).resize((2048,1024),Image.BILINEAR))
t=np.clip((-Eb-4)/22,0,1)[...,None]
a=a*(1-t)+np.array([20,36,50],np.float32)*t
buf=io.BytesIO(); Image.fromarray(a.astype(np.uint8)).save(buf,"JPEG",quality=78,optimize=True,progressive=True)
bmuri="data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()
open("glob/global.js","w").write("const GLOBAL_BM="+json.dumps(bmuri)+";\nconst GLOBAL_ELEV={w:%d,h:%d,b64:%s};\n"%(W,H,json.dumps(b64)))
print(len(bmuri)//1024,"KB bm", len(b64)//1024,"KB elev")
