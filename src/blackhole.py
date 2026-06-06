"""Black-hole gravitational lensing — real Schwarzschild null geodesics, ray-traced per pixel.

Physics (geometrized G=c=1, mass M=1, horizon r_s=2M): each photon path is planar through the BH
centre; in that plane u=1/r obeys  d2u/dphi2 = -u + 3M u^2 . We integrate that ODE for every pixel
ray (vectorised over all pixels), tracking the 3D position so we can (a) capture rays that fall past
the horizon (black), (b) light up an accretion disk where a ray crosses the equatorial plane, and
(c) sample a starfield for rays that escape — bent, so the far side of the disk is lensed into view.

Heavy (per-pixel ODE x many frames) -> meant for CI, not the VM. Self-implemented; physics is PD.

  python blackhole.py --output out.mp4 --frames 120 --w 1280 --h 720 [--still]
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio

M = 1.0
RS = 2.0 * M
DISK_IN, DISK_OUT = 6.0, 22.0      # ISCO (3 r_s) .. outer
CAM_D = 20.0


def _starfield(dirs):
    """dirs: (...,3) unit. Procedural starfield + faint nebula by direction hash."""
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    # quantise direction to a grid and hash -> sparse stars
    gx = np.floor((np.arctan2(z, x) + math.pi) / (2*math.pi) * 1400).astype(np.int64)
    gy = np.floor((np.arcsin(np.clip(y, -1, 1)) + math.pi/2) / math.pi * 800).astype(np.int64)
    h = (gx * 73856093) ^ (gy * 19349663)
    h = (h ^ (h >> 13)) & 2047
    star = (h < 2).astype(np.float32)                       # sparser, dimmer -> stable backdrop not flicker
    bright = (0.35 + 0.45*((h % 7) / 6.0)) * star
    col = np.zeros(dirs.shape, np.float32)
    for c in range(3):
        col[..., c] = bright * (150 + 40*((h>>(c*2)) & 1))
    neb = (0.5 + 0.5*np.sin(x*3+1)) * (0.5+0.5*np.cos(y*2))  # very faint nebula
    col[..., 0] += neb*4; col[..., 1] += neb*6; col[..., 2] += neb*14
    return col


def _disk_color(r, az, spin):
    """emission by radius (hot inner->cool outer) x swirling spiral arms (rotate w/ spin) x Doppler."""
    t = np.clip((r - DISK_IN) / (DISK_OUT - DISK_IN), 0, 1)
    bri = (1.0 - t) ** 1.7 * 1.0 + 0.12
    swirl = 0.55 + 0.45*np.sin(2.0*az - spin - 6.0*t)       # spiral arms winding inward, rotating
    dopp = 0.55 + 0.65*np.cos(az - 1.1)                     # one side brighter (relativistic beaming look)
    b = bri * swirl * dopp
    col = np.stack([255*np.ones_like(t), 205 - 120*t, 95 - 70*t], -1) * b[..., None]
    return np.clip(col, 0, 255)


def render_frame(cam_ang, W, H, fov=0.42, steps=700, dphi=0.022, inc=1.40, spin=0.0):
    # camera position (orbit azimuthally at fixed inclination)
    C = CAM_D * np.array([math.sin(cam_ang)*math.sin(inc), math.cos(inc), math.cos(cam_ang)*math.sin(inc)])
    e_r = C / np.linalg.norm(C)                              # BH->camera (same for all pixels)
    fwd = -e_r                                               # look at BH
    up0 = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up0); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    # pixel ray directions
    aspect = W / H
    xs = (np.linspace(-1, 1, W) * math.tan(fov) * aspect)
    ys = (np.linspace(1, -1, H) * math.tan(fov))
    gx, gy = np.meshgrid(xs, ys)
    dirs = fwd[None, None, :] + gx[..., None]*right[None, None, :] + gy[..., None]*up[None, None, :]
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    P = W*H
    d = dirs.reshape(P, 3)
    vr = d @ e_r                                             # radial comp
    vt_vec = d - vr[:, None]*e_r[None, :]
    vt = np.linalg.norm(vt_vec, axis=1) + 1e-9
    e_t = vt_vec / vt[:, None]                               # per-pixel tangential basis
    # initial geodesic state
    u = np.full(P, 1.0/CAM_D, np.float32)
    w = (-(vr/vt) * u).astype(np.float32)                    # du/dphi
    phi = np.zeros(P, np.float32)
    out = np.zeros((P, 3), np.float32)
    done = np.zeros(P, bool)
    prev_y = np.full(P, CAM_D*e_r[1], np.float32)            # world-y at phi=0 (= C_y)
    disk_acc = np.zeros((P, 3), np.float32)
    for _ in range(steps):
        a = ~done
        # RK2 step on (u,w): u'=w, w'=-u+3M u^2
        u1 = u + w*dphi*0.5
        w1 = w + (-u + 3*M*u*u)*dphi*0.5
        u_n = np.clip(u + w1*dphi, 1e-6, 12.0)
        w_n = w + (-u1 + 3*M*u1*u1)*dphi
        phi_n = phi + dphi
        r_n = 1.0/u_n
        # world position at phi_n
        cphi, sphi = np.cos(phi_n), np.sin(phi_n)
        py = r_n*(cphi*e_r[1] + sphi*e_t[:, 1])
        # disk crossing: world-y changed sign, at radius in disk band, near equatorial plane
        cross = a & (prev_y*py < 0) & (r_n > DISK_IN) & (r_n < DISK_OUT)
        if cross.any():
            px = r_n[cross]*(cphi[cross]*e_r[0] + sphi[cross]*e_t[cross, 0])
            pz = r_n[cross]*(cphi[cross]*e_r[2] + sphi[cross]*e_t[cross, 2])
            az = np.arctan2(pz, px)                           # disk-plane azimuth of crossing point
            disk_acc[cross] += _disk_color(r_n[cross], az, spin)
            done[cross] = True                                # opaque disk
        # horizon capture
        cap = a & (r_n <= RS*1.02)
        done[cap] = True                                      # stays black (out=0)
        # escape: far and moving outward (u decreasing)
        esc = a & (r_n > 55) & (w_n < 0)
        if esc.any():
            # final direction = d(pos)/dphi
            drdphi = -w_n[esc]/(u_n[esc]**2)
            pos_dir = (drdphi[:, None]*(cphi[esc, None]*e_r[None, :] + sphi[esc, None]*e_t[esc])
                       + r_n[esc, None]*(-sphi[esc, None]*e_r[None, :] + cphi[esc, None]*e_t[esc]))
            pos_dir /= (np.linalg.norm(pos_dir, axis=1, keepdims=True)+1e-9)
            out[esc] = _starfield(pos_dir)
            done[esc] = True
        u, w, phi, prev_y = u_n, w_n, phi_n, py
        if done.all():
            break
    out += disk_acc
    img = np.clip(out.reshape(H, W, 3), 0, 255).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="/tmp/bh.mp4")
    ap.add_argument("--frames", type=int, default=120)        # total frames in the full orbit
    ap.add_argument("--start", type=int, default=0)           # this chunk's first frame index
    ap.add_argument("--count", type=int, default=0)           # frames this chunk renders (0 = all)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--still", action="store_true")
    a = ap.parse_args()
    def cam_for(i):
        f = i / a.frames
        inc = 1.32 - 0.60*math.sin(math.pi*f)        # tilt edge-on -> toward face-on -> back (halo morphs)
        spin = 2*math.pi*f*3.0                        # disk makes 3 rotations over the clip
        cam_ang = 2*math.pi*f*0.5                     # slow half azimuth orbit
        return cam_ang, inc, spin
    if a.still:
        ca, inc, spin = cam_for(a.frames//6)
        Image.fromarray(render_frame(ca, a.w, a.h, inc=inc, spin=spin)).save(a.output)
        print(f"[OK] still -> {a.output}"); return
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    cnt = a.count or a.frames
    wri = imageio.get_writer(str(out), fps=a.fps, codec="libx264", quality=8, macro_block_size=8)
    for k in range(cnt):
        i = a.start + k
        ca, inc, spin = cam_for(i)
        wri.append_data(render_frame(ca, a.w, a.h, inc=inc, spin=spin))
        print(f"  frame {i+1}/{a.frames}", flush=True)
    wri.close()
    print(f"[OK] black hole frames [{a.start},{a.start+cnt}) -> {out}")


if __name__ == "__main__":
    main()
