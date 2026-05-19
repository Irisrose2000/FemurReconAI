"""
generate_dataset.py  —  Standalone dataset generation script.
Run from the femur_dataset/ folder:
    python generate_dataset.py --n_samples 20 --output_dir data/processed
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import ball
from tqdm import tqdm

# ── inline femur geometry ────────────────────────────────────────────
def _ellipsoid(grid, cx, cy, cz, rx, ry, rz):
    zz,yy,xx = grid
    return ((xx-cx)/rx)**2 + ((yy-cy)/ry)**2 + ((zz-cz)/rz)**2 <= 1.0

def _cylinder(grid, z0, z1, cx, cy, rx, ry, taper=0.0):
    zz,yy,xx = grid
    t  = np.clip((zz-z0)/max(z1-z0,1),0,1)
    rx_t = rx*(1+taper*t); ry_t = ry*(1+taper*t)
    return (zz>=z0)&(zz<=z1)&(((xx-cx)**2/rx_t**2+(yy-cy)**2/ry_t**2)<=1.0)

def generate_femur(D,H,W,seed):
    rng   = np.random.default_rng(seed)
    grid  = np.mgrid[0:D,0:H,0:W].astype(np.float32)
    cx,cy = W/2+rng.uniform(-2,2), H/2+rng.uniform(-2,2)

    head_r   = rng.uniform(0.11,0.14)*H
    neck_len = rng.uniform(0.14,0.20)*D
    shaft_r  = rng.uniform(0.09,0.13)*H
    canal_r  = rng.uniform(0.045,0.07)*H

    z_head      = rng.uniform(0.07,0.12)*D
    z_neck_end  = z_head+neck_len
    z_shaft_top = z_neck_end
    z_shaft_bot = z_shaft_top + rng.uniform(0.55,0.65)*D
    z_distal    = min(D-1, z_shaft_bot + rng.uniform(0.06,0.12)*D)

    hx = cx + rng.uniform(-0.14,-0.10)*W
    head  = _ellipsoid(grid, hx, cy, z_head, head_r, head_r, head_r*0.9)
    neck_r= head_r*rng.uniform(0.55,0.70)
    neck  = _cylinder(grid, z_head, z_neck_end, hx, cy, neck_r, neck_r, taper=rng.uniform(-0.25,-0.10))
    gt_r  = rng.uniform(0.08,0.11)*H
    gt    = _ellipsoid(grid, cx+rng.uniform(0.10,0.16)*W, cy, z_neck_end, gt_r*0.7, gt_r*0.9, gt_r)
    shaft_outer = _cylinder(grid, z_shaft_top, z_shaft_bot, cx, cy, shaft_r, shaft_r, taper=0.10)
    canal_mask  = _cylinder(grid, z_shaft_top, z_shaft_bot, cx, cy, canal_r, canal_r)
    shaft = shaft_outer & ~canal_mask
    distal_r = rng.uniform(0.14,0.18)*H
    distal = _cylinder(grid, z_shaft_bot, z_distal, cx, cy, shaft_r*1.4, shaft_r*1.4)
    mc = _ellipsoid(grid, cx-0.07*W, cy-0.05*H, z_distal, distal_r*0.9, distal_r*0.85, distal_r)
    lc = _ellipsoid(grid, cx+0.07*W, cy+0.05*H, z_distal, distal_r*0.9, distal_r*0.85, distal_r)

    femur = head|neck|gt|shaft|distal|mc|lc
    femur = ndi.gaussian_filter(femur.astype(np.float32), sigma=1.0) > 0.35
    canal = ndi.gaussian_filter(canal_mask.astype(np.float32), sigma=0.8) > 0.3

    z0 = int(z_shaft_top); z1 = int(min(z_shaft_bot, D-1))
    params = dict(
        canal_radius_mm=round(float(canal_r),2),
        shaft_radius_mm=round(float(shaft_r),2),
        femur_length_mm=round(float(z_distal-z_head),1),
        z_shaft_top=z0, z_shaft_bot=z1,
    )
    return femur, canal, params, (z0, z1)

# ── inline fracture application ──────────────────────────────────────
FRACTURE_TYPES = ['A1','A2','A3','B1','B2','C1','C2']
WEIGHTS        = [0.25,0.20,0.10,0.15,0.12,0.10,0.08]

def _slab(shape, z_c, half, tilt_xz=0, tilt_yz=0):
    D,H,W = shape
    zz,yy,xx = np.mgrid[0:D,0:H,0:W].astype(np.float32)
    dz=zz-z_c; dx=xx-W/2; dy=yy-H/2
    tx=np.deg2rad(tilt_xz); ty=np.deg2rad(tilt_yz)
    pv = dz*np.cos(tx)*np.cos(ty) - dx*np.sin(tx) - dy*np.sin(ty)
    return (pv>=-half)&(pv<=half)

def _add_roughness(gap, sigma=1.5, thr=0.35):
    noise  = ndi.gaussian_filter(np.random.randn(*gap.shape).astype(np.float32), sigma)
    noise  = (noise-noise.min())/(noise.max()-noise.min()+1e-8)
    extra  = ndi.binary_dilation(gap, iterations=1) & (noise > 1-thr)
    return gap | extra

def apply_fracture(mask, z0, z1, D, H, W, ao_code, seed):
    rng = np.random.default_rng(seed)
    gap = np.zeros_like(mask, dtype=bool)
    z_c = int(rng.integers(z0+3, max(z0+4, z1-3)))

    if ao_code == 'A1':
        half = float(rng.uniform(1.5,5.0))
        gap  = _slab((D,H,W), z_c, half, float(rng.uniform(-12,12)))
    elif ao_code == 'A2':
        half = float(rng.uniform(2.0,7.0))
        gap  = _slab((D,H,W), z_c, half, float(rng.uniform(30,60))*rng.choice([-1,1]), float(rng.uniform(-15,15)))
    elif ao_code == 'A3':
        half = float(rng.uniform(2.0,6.0))
        zz,yy,xx = np.mgrid[0:D,0:H,0:W].astype(np.float32)
        dz=zz-z_c; dx=xx-W/2; theta=0.15*dz
        pv = dz*np.cos(np.deg2rad(20)) - (dx*np.cos(theta))*np.sin(np.deg2rad(20))
        gap = (pv>=-half)&(pv<=half)
    elif ao_code in ('B1','B2'):
        half = float(rng.uniform(3.0,8.0))
        g1 = _slab((D,H,W), z_c-half*0.5, half*0.4, float(rng.uniform(15,35)))
        g2 = _slab((D,H,W), z_c+half*0.5, half*0.4, float(rng.uniform(-35,-15)))
        gap = g1|g2
        if ao_code=='B2':
            for _ in range(int(rng.integers(1,3))):
                fz=z_c+int(rng.integers(-6,6)); fr=float(rng.uniform(2,5))
                fy=H/2+float(rng.uniform(-0.10,0.10))*H; fx=W/2+float(rng.uniform(-0.10,0.10))*W
                zz,yy,xx=np.mgrid[0:D,0:H,0:W].astype(np.float32)
                gap |= ((zz-fz)**2+(yy-fy)**2+(xx-fx)**2)<=fr**2
    elif ao_code=='C1':
        span=z1-z0
        zu=z0+int(span*float(rng.uniform(0.20,0.38))); zl=z0+int(span*float(rng.uniform(0.62,0.80)))
        g1=_slab((D,H,W),zu,float(rng.uniform(1.5,4.0)),float(rng.uniform(-20,20)))
        g2=_slab((D,H,W),zl,float(rng.uniform(1.5,4.0)),float(rng.uniform(-20,20)))
        gap=g1|g2; z_c=(zu+zl)//2
    else:  # C2
        half=float(rng.uniform(8.0,16.0))
        gap=_slab((D,H,W),z_c,half,float(rng.uniform(-25,25)))
        for _ in range(int(rng.integers(3,6))):
            fz=z_c+float(rng.uniform(-half*0.6,half*0.6))
            fr=float(rng.uniform(2,5))
            fy=H/2+float(rng.uniform(-0.12,0.12))*H; fx=W/2+float(rng.uniform(-0.12,0.12))*W
            zz,yy,xx=np.mgrid[0:D,0:H,0:W].astype(np.float32)
            gap|=((zz-fz)**2+(yy-fy)**2+(xx-fx)**2)<=fr**2

    gap  = _add_roughness(gap.astype(bool)&mask)
    gap  = gap & mask
    gap_size_mm = float(gap.sum()**(1/3))   # rough estimate
    n_frags = {'A1':2,'A2':2,'A3':2,'B1':3,'B2':4,'C1':3,'C2':5}.get(ao_code,2)
    return mask & ~gap, gap, gap_size_mm, n_frags

# ── inline CT simulator ──────────────────────────────────────────────
def simulate_ct(femur_mask, canal_mask, seed):
    rng = np.random.default_rng(seed)
    D,H,W = femur_mask.shape
    vol   = np.full((D,H,W), -1000.0, dtype=np.float32)

    # soft tissue
    dilated = ndi.binary_dilation(femur_mask, iterations=8)
    vol[dilated & ~femur_mask] = float(rng.uniform(40,70))

    # cortex
    cortex = femur_mask & ~ndi.binary_erosion(femur_mask, iterations=3)
    vol[cortex] = rng.uniform(800,1500, size=(D,H,W)).astype(np.float32)[cortex]

    # cancellous
    canc = femur_mask & ~cortex & ~canal_mask
    base = rng.uniform(200,500, size=(D,H,W)).astype(np.float32)
    base += ndi.gaussian_filter(rng.standard_normal((D,H,W)).astype(np.float32), sigma=3)*40
    vol[canc] = base[canc]

    # marrow
    vol[canal_mask & femur_mask] = rng.uniform(-80,80, size=(D,H,W)).astype(np.float32)[canal_mask & femur_mask]

    # noise
    sigma = float(rng.uniform(18,38))
    noise = rng.standard_normal((D,H,W)).astype(np.float32)*sigma
    noise[femur_mask] *= 0.4
    vol += noise

    # bias field
    gs = 4
    small = rng.standard_normal((gs,gs,gs)).astype(np.float32)
    bias  = ndi.zoom(small, [D/gs,H/gs,W/gs], order=1)[:D,:H,:W]
    vol  += bias * float(rng.uniform(25,60))

    return vol

# ── inline window + crop ─────────────────────────────────────────────
def window_and_crop(vol, mask_intact, mask_frac, gap, shape, hu_min=200, hu_max=1800):
    def _cp(arr, tgt, pad_val=0.0):
        out = np.full(tgt, pad_val, dtype=arr.dtype)
        slcs_s=[]; slcs_d=[]
        for s,t in zip(arr.shape, tgt):
            if s>=t: st=(s-t)//2; slcs_s.append(slice(st,st+t)); slcs_d.append(slice(0,t))
            else:    st=(t-s)//2; slcs_s.append(slice(0,s));     slcs_d.append(slice(st,st+s))
        out[tuple(slcs_d)] = arr[tuple(slcs_s)]
        return out
    win = np.clip(vol, hu_min, hu_max)
    win = (win-hu_min)/(hu_max-hu_min)
    return (_cp(win.astype(np.float32), shape),
            _cp(mask_intact.astype(np.float32), shape),
            _cp(mask_frac.astype(np.float32),   shape),
            _cp(gap.astype(np.float32),          shape))

# ── main ─────────────────────────────────────────────────────────────
def generate_dataset(n_samples, output_dir, volume_shape=(128,128,64), seed_base=0):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    D,H,W = volume_shape
    gen_D = D*2   # generate taller then crop

    ao_types = random.choices(FRACTURE_TYPES, weights=WEIGHTS, k=n_samples)
    metadata = []

    for i in tqdm(range(n_samples), desc='Generating samples'):
        seed = seed_base + i*9973
        random.seed(seed); np.random.seed(seed)
        ao = ao_types[i]

        try:
            # 1. Geometry
            intact, canal, params, (z0,z1) = generate_femur(gen_D,H,W, seed)
            # 2. Fracture
            fractured, gap, gap_mm, n_frags = apply_fracture(intact, z0, z1, gen_D, H, W, ao, seed+1)
            # 3. CT simulation
            ct = simulate_ct(fractured, canal & fractured, seed+2)
            # 4. Window + crop
            win, mask_c, frac_c, gap_c = window_and_crop(ct, intact, fractured, gap, (D,H,W))
            # 5. Save
            out_path = out / f"sample_{i:05d}.npz"
            np.savez_compressed(str(out_path),
                windowed    = win,
                mask        = mask_c,
                fractured   = frac_c,
                completed   = mask_c,
                gap_mask    = gap_c,
                spacing     = np.array([1.5,1.0,1.0], dtype=np.float32),
                ao_code     = np.bytes_(ao),
                fracture_type=np.bytes_(ao),
                gap_size_mm = np.float32(gap_mm),
                n_fragments = np.int32(n_frags),
            )
            metadata.append(dict(idx=i,status='ok',ao_code=ao,fracture_type=ao,
                                 gap_size_mm=round(gap_mm,2), n_fragments=n_frags,
                                 femur_len_mm=params['femur_length_mm'],
                                 canal_mm=params['canal_radius_mm']*2,
                                 path=str(out_path)))
        except Exception as e:
            metadata.append(dict(idx=i,status='error',error=str(e),ao_code=ao,fracture_type=ao,
                                 gap_size_mm=0, n_fragments=0, femur_len_mm=0, canal_mm=0, path=''))

    # Save manifest
    from collections import Counter
    ok = [m for m in metadata if m['status']=='ok']
    summary = dict(
        total_samples=len(ok), total_errors=len(metadata)-len(ok),
        per_ao_code=dict(Counter(m['ao_code'] for m in ok)),
        mean_gap_mm=round(float(np.mean([m['gap_size_mm'] for m in ok])),2) if ok else 0,
    )
    with open(out/'manifest.json','w') as f:
        json.dump({'summary':summary,'samples':metadata}, f, indent=2)

    print(f"\n✅  Generated {len(ok)} samples  →  {out}")
    print(f"   AO distribution: {summary['per_ao_code']}")
    print(f"   Mean gap: {summary['mean_gap_mm']} mm")
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples',  type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='data/processed')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.n_samples, args.output_dir, seed_base=args.seed)