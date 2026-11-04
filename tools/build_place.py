"""Build imagery + elevation layers for one drop zone.

Layers k = 0..10, each 1024 px square in a local azimuthal-equidistant projection
centred on the target. Metres per pixel = 10 * 2**k  (k0 = 10.24 km wide, k10 = 10,486 km).
Imagery: Sentinel-2 L2A true colour (k0..6), NASA Blue Marble fill (all k).
Elevation: AWS Terrarium tiles, 129x129 grid per layer, int16 metres.
"""
import io, json, math, os, sys, urllib.request, concurrent.futures as cf
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling, transform as tx
from rasterio.transform import from_origin
import mgrs

os.environ.setdefault("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["GDAL_HTTP_MULTIRANGE"] = "YES"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["VSI_CACHE"] = "TRUE"
for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
    os.environ.pop(k, None)

N = 1024
KMAX = 10
SKMAX = 6
BUCKET = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/"
BM = "/home/claude/dl/package/example/img/earth-blue-marble.jpg"
CACHE = "/home/claude/build/cache"
os.makedirs(CACHE, exist_ok=True)


def get(url, binary=True):
    p = os.path.join(CACHE, url.replace("https://", "").replace("/", "_").replace("?", "_").replace("&", "_"))
    if os.path.exists(p):
        return open(p, "rb").read()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = r.read()
            open(p, "wb").write(d)
            return d
        except Exception as e:
            if getattr(e, "code", None) == 404:
                return None
            err = e
    print("FAIL", url, err)
    return None


def list_prefixes(prefix):
    d = get(BUCKET + "?list-type=2&delimiter=/&prefix=" + prefix)
    if not d:
        return []
    s = d.decode()
    out = []
    for part in s.split("<Prefix>")[2:]:
        out.append(part.split("</Prefix>")[0])
    return out


def scene_meta(prefix):
    name = prefix.rstrip("/").split("/")[-1]
    d = get(BUCKET + prefix + name + ".json")
    if not d:
        return None
    p = json.loads(d)["properties"]
    return dict(prefix=prefix, cloud=p.get("eo:cloud_cover", 100), nodata=p.get("s2:nodata_pixel_percentage", 100),
                date=name.split("_")[2])


SCENES = {}
MONTHS = None
FOCUS = [None]
GAIN = [None]


def pick_scenes(tile, north, pool):
    if tile in SCENES:
        return SCENES[tile]
    zone, band, sq = tile[:-3], tile[-3], tile[-2:]
    months = MONTHS or ([6, 7, 8, 9, 5, 4, 10] if north else [1, 2, 12, 3, 11, 10])
    cands = []
    for yr in (2024, 2023):
        for m in months[:4]:
            cands += list_prefixes(f"sentinel-s2-l2a-cogs/{zone}/{band}/{sq}/{yr}/{m}/")
        if len(cands) > 20:
            break
    metas = [x for x in pool.map(scene_meta, cands) if x]
    metas.sort(key=lambda x: (x["cloud"] + x["nodata"] * 0.15))
    if FOCUS[0] is not None and not SCENES:
        # tile under the drop zone: rank by cloud cover right around the target (scene classification band)
        lat, lon = FOCUS[0]
        def local(x):
            try:
                with rasterio.open("/vsicurl/" + BUCKET + x["prefix"] + "SCL.tif", overview_level=1) as src:
                    (cx,), (cy,) = tx("EPSG:4326", src.crs, [lon], [lat])
                    win = from_bounds(cx - 12000, cy - 12000, cx + 12000, cy + 12000, transform=src.transform).round_offsets().round_lengths()
                    a = src.read(1, window=win, boundless=True, fill_value=0)
                    bad = np.isin(a, [0, 3, 8, 9, 10]).mean()
                    return bad
            except Exception as e:
                return 1.0
        top = [x for x in metas if x["nodata"] < 40][:14]
        scores = list(pool.map(local, top))
        for x, sc in zip(top, scores):
            x["local"] = sc
        top.sort(key=lambda x: x["local"])
        print("focus scenes", [(x["prefix"].split("/")[-2], round(x["local"], 3)) for x in top[:4]], flush=True)
        top[0]["focus"] = True
        SCENES[tile] = [top[0]] + [x for x in metas if x["cloud"] < 5 and x is not top[0]][:2]
        return SCENES[tile]
    full = [x for x in metas if x["nodata"] < 3 and x["cloud"] < 15]
    best = (full[:1] or []) + [x for x in metas if x["cloud"] < 5][:3]
    seen, out = set(), []
    for x in best:
        if x["prefix"] not in seen:
            seen.add(x["prefix"]); out.append(x)
    SCENES[tile] = out[:3]
    return SCENES[tile]


def tiles_for_bbox(crs, W):
    m = mgrs.MGRS()
    xs = np.linspace(-W / 2, W / 2, 13)
    X, Y = np.meshgrid(xs, xs)
    lon, lat = tx(crs, "EPSG:4326", X.ravel().tolist(), Y.ravel().tolist())
    tiles = set()
    for la, lo in zip(lat, lon):
        if -80 < la < 84:
            tiles.add(m.toMGRS(la, lo, MGRSPrecision=0))
    return tiles


def fill_from(dst, src_arr, src_tf, src_crs, dst_tf, dst_crs, nodata=0):
    tmp = np.zeros_like(dst)
    reproject(src_arr, tmp, src_transform=src_tf, src_crs=src_crs, dst_transform=dst_tf, dst_crs=dst_crs,
              resampling=Resampling.bilinear, src_nodata=nodata, dst_nodata=0)
    hole = (dst.sum(axis=0) == 0)
    ok = (tmp.min(axis=0) > 0)  # avoid edge-darkened pixels
    m = hole & ok
    dst[:, m] = tmp[:, m]


def read_scene_into(dst, prefix, mpp, dst_tf, dst_crs, W):
    name = prefix.rstrip("/").split("/")[-1]
    url = "/vsicurl/" + BUCKET + prefix + "TCI.tif"
    lvl = int(round(math.log2(mpp / 10))) - 1  # overview index; -1 = full res
    try:
        kw = {} if lvl < 0 else {"overview_level": min(lvl, 3)}
        with rasterio.open(url, **kw) as src:
            # dst bbox in src crs
            xs = np.linspace(-W / 2, W / 2, 9)
            X, Y = np.meshgrid(xs, xs)
            sx, sy = tx(dst_crs, src.crs, X.ravel().tolist(), Y.ravel().tolist())
            b = (max(min(sx), src.bounds.left), max(min(sy), src.bounds.bottom),
                 min(max(sx), src.bounds.right), min(max(sy), src.bounds.top))
            if b[0] >= b[2] or b[1] >= b[3]:
                return
            win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
            if win.width < 2 or win.height < 2:
                return
            arr = src.read(window=win)
            wtf = src.window_transform(win)
            fill_from(dst, arr, wtf, src.crs, dst_tf, dst_crs)
    except Exception as e:
        print("  scene fail", name, e)


def terrarium(z, x, y):
    d = get(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png")
    if not d:
        return np.zeros((256, 256), np.float32)
    a = np.asarray(Image.open(io.BytesIO(d)).convert("RGB")).astype(np.float32)
    return a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768


def elevation_grid(crs, W, z, pool, n=129):
    step = W / (n - 1)
    dtf = from_origin(-W / 2 - step / 2, W / 2 + step / 2, step, step)
    xs = np.linspace(-W / 2, W / 2, 9)
    X, Y = np.meshgrid(xs, xs)
    lon, lat = tx(crs, "EPSG:4326", X.ravel().tolist(), Y.ravel().tolist())
    lat = np.clip(lat, -85, 85)
    def t(lo, la):
        nn = 2 ** z
        return (lo + 180) / 360 * nn, (1 - math.log(math.tan(math.radians(la)) + 1 / math.cos(math.radians(la))) / math.pi) / 2 * nn
    txs, tys = zip(*[t(a, b) for a, b in zip(lon, lat)])
    x0, x1 = int(min(txs)), int(max(txs)); y0, y1 = max(int(min(tys)), 0), min(int(max(tys)), 2 ** z - 1)
    if x1 - x0 > 2 ** z - 1:
        x0, x1 = 0, 2 ** z - 1
    jobs = [(z, xx % (2 ** z), yy) for yy in range(y0, y1 + 1) for xx in range(x0, x1 + 1)]
    tiles = list(pool.map(lambda j: terrarium(*j), jobs))
    cols = x1 - x0 + 1
    mos = np.zeros(((y1 - y0 + 1) * 256, cols * 256), np.float32)
    for i, a in enumerate(tiles):
        r, c = divmod(i, cols)
        mos[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256] = a
    E = 20037508.342789244
    res = 2 * E / (256 * 2 ** z)
    stf = from_origin(-E + x0 * 256 * res, E - y0 * 256 * res, res, res)
    out = np.zeros((n, n), np.float32)
    reproject(mos, out, src_transform=stf, src_crs="EPSG:3857", dst_transform=dtf, dst_crs=crs,
              resampling=Resampling.bilinear)
    return np.clip(out, -11000, 9000).astype(np.int16)


def main(pid, lat, lon):
    out = f"/home/claude/fall/places/{pid}"
    os.makedirs(out, exist_ok=True)
    crs = rasterio.crs.CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
    north = lat >= 0
    FOCUS[0] = (lat, lon)
    bm = np.asarray(Image.open(BM).convert("RGB")).transpose(2, 0, 1).copy()
    bm = np.clip(bm.astype(np.float32) * 1.15, 1, 255).astype(np.uint8)
    bmtf = from_origin(-180, 90, 360 / bm.shape[2], 180 / bm.shape[1])
    pool = cf.ThreadPoolExecutor(24)
    elev = []
    for k in range(KMAX + 1):
        mpp = 10 * 2 ** k
        W = N * mpp
        dtf = from_origin(-W / 2, W / 2, mpp, mpp)
        dst = np.zeros((3, N, N), np.uint8)
        if k <= SKMAX:
            tiles = tiles_for_bbox(crs, W)
            print(f"k{k}: {len(tiles)} MGRS tiles", flush=True)
            for t in sorted(tiles):
                pick_scenes(t, north, pool)
            jobs = [(t, s) for t in sorted(tiles) for s in SCENES[t]]
            # order: lowest cloud first across tiles
            jobs.sort(key=lambda j: (0 if j[1].get("focus") else 1, j[1]["cloud"]))
            # read in parallel into per-job buffers, then composite in order
            def rd(j):
                buf = np.zeros_like(dst)
                read_scene_into(buf, j[1]["prefix"], mpp, dtf, crs, W)
                return buf
            for buf in cf.ThreadPoolExecutor(8).map(rd, jobs):
                hole = dst.sum(axis=0) == 0
                dst[:, hole] = buf[:, hole]
            # colour tune sentinel TCI (it is dark)
            has = dst.sum(axis=0) > 0
            f = dst.astype(np.float32) / 255
            if GAIN[0] is None:
                p90 = np.percentile(dst.max(axis=0)[has], 90)
                GAIN[0] = float(np.clip(195 / max(p90, 1), 0.9, 1.3))
                print("gain", GAIN[0], flush=True)
            f = np.clip(f * GAIN[0], 0, 1) ** 0.85
            dst = np.where(has[None], (f * 255).astype(np.uint8), 0)
        cover = (dst.sum(axis=0) > 0).mean()
        fill_from(dst, bm, bmtf, "EPSG:4326", dtf, crs)
        z = max(1, min(13, 11 - k + 1))
        # deep water -> uniform dark ocean so sentinel sea scenes and blue marble match
        em = elevation_grid(crs, W, max(1, z - 1), pool, n=256).astype(np.float32)
        em = np.asarray(Image.fromarray(em).resize((N, N), Image.BILINEAR))
        t = np.clip((-em - 4) / 22, 0, 1)[None]
        ocean = np.array([20, 36, 50], np.float32)[:, None, None]
        dst = (dst.astype(np.float32) * (1 - t) + ocean * t).astype(np.uint8)
        Image.fromarray(dst.transpose(1, 2, 0)).save(f"{out}/k{k}.jpg", quality=82, optimize=True, progressive=True)
        e = elevation_grid(crs, W, z, pool)
        elev.append(e)
        print(f"k{k} done, sentinel cover {cover:.2f}, elev centre {e[64,64]} max {e.max()}", flush=True)
    np.stack(elev).astype("<i2").tofile(f"{out}/elev.bin")
    json.dump({k: v for k, v in SCENES.items()}, open(f"{out}/scenes.json", "w"))


if __name__ == "__main__":
    if len(sys.argv) > 4:
        MONTHS = [int(x) for x in sys.argv[4].split(",")]
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
