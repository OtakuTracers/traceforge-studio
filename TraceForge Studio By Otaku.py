#!/usr/bin/env python3
"""
✦ Anime Line Art Studio  v6
────────────────────────────────────────────────────────────────
TABS
  0 · Original       — exact source image, no modifications
  1 · Line Art       — pencil sketch
  2 · Colour Guide   — original image + optional edge overlay + palette strip
  3 · Colour Map     — SEPARATE tab: flat colour zones with hex labels per region
  4 · Trace Steps    — professional artist construction steps (head circle → jaw
                       → neck → eye guides → nose/mouth → hair → details)

Colour Guide preview = original image only (zero saturation shift).
Trace Steps = anatomy-driven, cumulative layers, each step has text instructions.
Paint tab REMOVED.
All heavy processing is threaded.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading, os, sys, math

# ── Auto-install ──────────────────────────────────────────────────────────────
def _pip(pkg):
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for _m, _p in [("PIL","pillow"),("numpy","numpy"),("cv2","opencv-python")]:
    try: __import__(_m)
    except ImportError: _pip(_p)

from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2

# ══════════════════════════════════════════════════════════════════════════════
#  Theme
# ══════════════════════════════════════════════════════════════════════════════
BG     = "#0d0d18"
PANEL  = "#12121f"
CARD   = "#191928"
BORDER = "#22223a"
ACCENT = "#ff4d6d"
BLUE   = "#4cc9f0"
GREEN  = "#06d6a0"
AMBER  = "#ffd166"
PURPLE = "#b388ff"
TEXT   = "#e2e2f0"
SUB    = "#8888b0"
MUTED  = "#44445e"
HOVER  = "#1e1e32"

_FB = None

def _f(size, weight="normal"):
    return (_FB, size, weight)


# ══════════════════════════════════════════════════════════════════════════════
#  Image loading
# ══════════════════════════════════════════════════════════════════════════════

def load_image_pil(path: str, max_dim=1400):
    """Pillow loader — handles WEBP, PNG, JPG reliably. Returns (PIL RGB, cv2 BGR)."""
    pil = Image.open(path).convert("RGB")
    w, h = pil.size
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        pil = pil.resize((int(w*s), int(h*s)), Image.LANCZOS)
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return pil, bgr


# ══════════════════════════════════════════════════════════════════════════════
#  Line art
# ══════════════════════════════════════════════════════════════════════════════

def make_sketch(img_cv, blur_k=21) -> Image.Image:
    if blur_k % 2 == 0: blur_k += 1
    gray   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    inv    = 255 - gray
    blur   = cv2.GaussianBlur(inv, (blur_k, blur_k), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    k      = np.array([[0,-0.4,0],[-0.4,2.6,-0.4],[0,-0.4,0]])
    return Image.fromarray(
        np.clip(cv2.filter2D(sketch,-1,k), 0, 255).astype(np.uint8))


# ══════════════════════════════════════════════════════════════════════════════
#  Colour extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_colours(img_cv, n=10):
    """K-means on original BGR pixels. Zero saturation change."""
    h, w = img_cv.shape[:2]
    data = img_cv.reshape(-1,3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, labels, centers = cv2.kmeans(data, n, None, crit, 12, cv2.KMEANS_PP_CENTERS)
    centers = np.round(centers).astype(np.uint8)
    labels  = labels.flatten()
    counts  = np.bincount(labels, minlength=n)
    order   = np.argsort(-counts)
    centers, counts = centers[order], counts[order]
    remap = np.empty(n, dtype=np.int32)
    for ni,oi in enumerate(order): remap[oi]=ni
    label_map   = remap[labels].reshape(h,w)
    palette_rgb = [(int(c[2]),int(c[1]),int(c[0])) for c in centers]
    return palette_rgb, counts, label_map


def build_edge_overlay(label_map) -> Image.Image:
    """Transparent RGBA — only dark boundary lines, no zone fill."""
    label_u8 = (label_map % 32 * 8).astype(np.uint8)
    edges    = cv2.Canny(label_u8, 8, 25)
    kern     = np.ones((2,2), np.uint8)
    edges    = cv2.dilate(edges, kern)
    h, w     = edges.shape
    rgba     = np.zeros((h, w, 4), np.uint8)
    m        = edges > 0
    rgba[m]  = [15, 15, 15, 210]
    return Image.fromarray(rgba, "RGBA")


def composite_over(orig: Image.Image, overlay: Image.Image, alpha=0.7) -> Image.Image:
    base = orig.convert("RGBA")
    ov   = overlay.copy()
    r,g,b,a = ov.split()
    a = a.point(lambda x: int(x*alpha))
    ov = Image.merge("RGBA",(r,g,b,a))
    base.alpha_composite(ov)
    return base.convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
#  Colour Map — flat zones with labelled hex colours
# ══════════════════════════════════════════════════════════════════════════════

def build_colour_map(orig_pil: Image.Image, palette_rgb, label_map) -> Image.Image:
    """
    Flat colour zone image on white canvas.
    Each zone is filled with its palette colour.
    A small hex label is drawn at the centroid of each zone.
    """
    h, w   = label_map.shape
    canvas = Image.new("RGB", (w, h), (250,250,250))
    draw   = ImageDraw.Draw(canvas)

    palette_arr = np.array(palette_rgb, dtype=np.uint8)

    for idx, rgb in enumerate(palette_rgb):
        mask = (label_map == idx)
        if not mask.any(): continue

        # Fill zone pixels
        arr = np.array(canvas)
        arr[mask] = rgb
        canvas = Image.fromarray(arr)
        draw   = ImageDraw.Draw(canvas)

        # Find centroid for label placement
        ys, xs = np.where(mask)
        cx, cy = int(xs.mean()), int(ys.mean())

        # Hex label
        hex_c = "#{:02x}{:02x}{:02x}".format(*rgb)

        # Choose contrasting text colour
        lum = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
        txt_col = (0,0,0) if lum > 128 else (255,255,255)

        # Small filled pill behind text
        tw, th = 62, 14
        x0,y0 = cx-tw//2, cy-th//2
        draw.rectangle([x0,y0,x0+tw,y0+th], fill=rgb, outline=txt_col)
        draw.text((cx, cy), hex_c.upper(), fill=txt_col, anchor="mm")

    # Draw zone borders
    for idx in range(len(palette_rgb)):
        mask_u8 = ((label_map == idx) * 255).astype(np.uint8)
        contours,_ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        arr = np.array(canvas)
        cv2.drawContours(arr, contours, -1, (30,30,30), 1)
        canvas = Image.fromarray(arr)
        draw   = ImageDraw.Draw(canvas)

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
#  Trace Steps — professional artist construction
# ══════════════════════════════════════════════════════════════════════════════

STEP_META = [
    {
        "title": "Step 1 — Head Construction Circle",
        "colour": (255, 100, 100),
        "instruction": (
            "Start with the cranium circle.\n"
            "Professional artists block the skull as a sphere first.\n"
            "Find the largest rounded shape in the upper portion —\n"
            "this is your head boundary. Keep strokes loose and light."
        ),
        "blur": 80, "lo": 2, "hi": 8, "dilate": 6,
    },
    {
        "title": "Step 2 — Jaw, Chin & Face Silhouette",
        "colour": (255, 180, 50),
        "instruction": (
            "Add the jaw line and chin below the skull circle.\n"
            "Anime faces taper sharply from cheekbones to a pointed chin.\n"
            "Connect the circle to the chin with two curved lines.\n"
            "This completes the overall face shape."
        ),
        "blur": 50, "lo": 5, "hi": 15, "dilate": 5,
    },
    {
        "title": "Step 3 — Neck & Shoulders",
        "colour": (100, 220, 100),
        "instruction": (
            "Draw the neck cylinder beneath the chin.\n"
            "Then block in the shoulders as a wide curve or trapezoid.\n"
            "Keep proportions: neck width ≈ 1/3 of face width.\n"
            "These anchors help you place everything else."
        ),
        "blur": 40, "lo": 8, "hi": 20, "dilate": 4,
    },
    {
        "title": "Step 4 — Eye Line & Brow Guides",
        "colour": (80, 180, 255),
        "instruction": (
            "Draw a horizontal guideline across the middle of the face.\n"
            "Anime eyes sit on or just above this midline.\n"
            "Sketch large oval placeholders for each eye.\n"
            "Space them roughly one eye-width apart."
        ),
        "blur": 25, "lo": 12, "hi": 30, "dilate": 3,
    },
    {
        "title": "Step 5 — Nose & Mouth Placement",
        "colour": (180, 100, 255),
        "instruction": (
            "Place the nose halfway between the eye line and chin.\n"
            "In anime style the nose is often minimal — a small dot or line.\n"
            "The mouth sits halfway between nose and chin.\n"
            "Keep both features small relative to the eyes."
        ),
        "blur": 15, "lo": 18, "hi": 45, "dilate": 2,
    },
    {
        "title": "Step 6 — Hair Outline & Volume",
        "colour": (255, 220, 50),
        "instruction": (
            "Draw the overall hair silhouette over and around the head circle.\n"
            "Anime hair is drawn in large flowing clumps, not individual strands.\n"
            "The hairline starts slightly above the head circle's top.\n"
            "Block in bangs, side sections, and back volume separately."
        ),
        "blur": 8, "lo": 25, "hi": 60, "dilate": 2,
    },
    {
        "title": "Step 7 — Fine Details & Clean Lines",
        "colour": (50, 220, 200),
        "instruction": (
            "Refine all edges: pupils, irises, eyelashes, lip details.\n"
            "Add ear shapes, hair strands within each clump.\n"
            "Clean up the jaw and face outline.\n"
            "This final pass is where you go dark with confident strokes."
        ),
        "blur": 3, "lo": 35, "hi": 80, "dilate": 1,
    },
]


def build_drawing_steps(img_cv: np.ndarray) -> list:
    """
    Returns list of dicts, each with:
      'title', 'instruction', 'image' (PIL RGB — white background, cumulative edges)
    Steps are cumulative — each step adds its layer on top of previous ones.
    """
    gray    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w    = gray.shape
    results = []

    # Cumulative canvas (white)
    cumulative = np.ones((h, w), np.uint8) * 255

    for meta in STEP_META:
        bk = meta["blur"]
        if bk % 2 == 0: bk += 1

        blurred = cv2.GaussianBlur(gray, (bk, bk), 0)
        edges   = cv2.Canny(blurred, meta["lo"], meta["hi"])

        # Dilate to make strokes visible
        d = meta["dilate"]
        if d > 0:
            kern  = np.ones((d,d), np.uint8)
            edges = cv2.dilate(edges, kern)

        # Draw this step's edges onto cumulative canvas in step colour
        # We'll composite as RGB
        if not results:
            canvas_rgb = np.stack([cumulative]*3, axis=-1)
        else:
            canvas_rgb = np.array(results[-1]["image"])

        col = meta["colour"]
        mask = edges > 0

        # Soften previous steps slightly (they appear darker/grey)
        canvas_rgb = canvas_rgb.copy()
        # New edges drawn in step colour
        canvas_rgb[mask] = col

        step_img = Image.fromarray(canvas_rgb.astype(np.uint8), "RGB")

        results.append({
            "title":       meta["title"],
            "instruction": meta["instruction"],
            "colour":      col,
            "image":       step_img,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Smart canvas widget
# ══════════════════════════════════════════════════════════════════════════════

class ImageCanvas(tk.Canvas):
    def __init__(self, parent, bg=BG, **kw):
        super().__init__(parent, bg=bg, highlightthickness=0, **kw)
        self._pil  = None
        self._tkim = None
        self.bind("<Configure>", self._on_cfg)

    def set_image(self, pil_img):
        self._pil = pil_img.convert("RGB") if pil_img else None
        self._redraw()

    def clear(self):
        self._pil = None; self.delete("all")

    def _on_cfg(self, e):
        if self._pil: self._redraw(e.width, e.height)

    def _redraw(self, w=None, h=None):
        if not self._pil: return
        self.update_idletasks()
        w = w or self.winfo_width(); h = h or self.winfo_height()
        if w < 2 or h < 2: return
        iw,ih = self._pil.size
        scale  = min(w/iw, h/ih)
        nw,nh  = max(1,int(iw*scale)), max(1,int(ih*scale))
        disp   = self._pil.resize((nw,nh), Image.LANCZOS)
        self._tkim = ImageTk.PhotoImage(disp)
        self.delete("all")
        self.create_image(w//2, h//2, anchor="center", image=self._tkim)


# ══════════════════════════════════════════════════════════════════════════════
#  Palette strip
# ══════════════════════════════════════════════════════════════════════════════

class PaletteStrip(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, highlightthickness=0, **kw)
        self._pal=[]; self._cnt=[]
        self.bind("<Configure>", lambda e: self._draw(e.width, e.height))

    def set_palette(self, pal, cnt):
        self._pal=pal; self._cnt=cnt
        self.update_idletasks()
        self._draw(self.winfo_width(), self.winfo_height())

    def _draw(self, w, h):
        self.delete("all")
        if not self._pal or w<10: return
        self.create_rectangle(0,0,w,h,fill=BG,outline="")
        n=len(self._pal); tot=max(sum(self._cnt),1); sw=w/n
        pad=max(3,int(sw*0.05))
        for i,(rgb,cnt) in enumerate(zip(self._pal,self._cnt)):
            x0=int(i*sw)+pad; x1=int((i+1)*sw)-pad
            hx="#{:02x}{:02x}{:02x}".format(*rgb)
            self.create_rectangle(x0,8,x1,h-28,fill=hx,outline="#555566")
            cx=(x0+x1)//2
            self.create_text(cx,h-17,text=hx.upper(),fill="#c8c8e0",
                             font=("TkFixedFont",7),anchor="center")
            self.create_text(cx,h-6,text=f"{100*cnt//tot}%",fill=MUTED,
                             font=("TkFixedFont",6),anchor="center")


# ══════════════════════════════════════════════════════════════════════════════
#  Projector
# ══════════════════════════════════════════════════════════════════════════════

class Projector(tk.Toplevel):
    def __init__(self, master, pil_img):
        super().__init__(master)
        self._img=pil_img
        self.overrideredirect(True)
        self.wm_attributes("-topmost",True)
        self.wm_attributes("-alpha",0.50)
        w,h=520,520
        sw,sh=self.winfo_screenwidth(),self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self._dx=self._dy=self._rw=self._rh=0
        self._build()

    def _build(self):
        bar=tk.Frame(self,bg="#0b0b14",height=36)
        bar.pack(fill="x"); bar.pack_propagate(False)
        tk.Label(bar,text="✦  COLOUR PROJECTOR",bg="#0b0b14",fg=ACCENT,
                 font=(_FB,9,"bold")).pack(side="left",padx=12)
        tk.Button(bar,text="✕",command=self.destroy,
                  bg=ACCENT,fg="white",relief="flat",cursor="hand2",
                  font=(_FB,9,"bold"),padx=8,
                  activebackground="#ff7088").pack(side="right",padx=8,pady=6)
        tk.Label(bar,text="opacity",bg="#0b0b14",fg=MUTED,
                 font=(_FB,8)).pack(side="right",padx=4)
        av=tk.DoubleVar(value=0.50)
        tk.Scale(bar,from_=0.08,to=0.95,resolution=0.04,
                 orient="horizontal",variable=av,bg="#0b0b14",fg=SUB,
                 troughcolor=CARD,highlightthickness=0,showvalue=False,length=90,
                 command=lambda v:self.wm_attributes("-alpha",float(v))
                 ).pack(side="right")
        bar.bind("<ButtonPress-1>",self._ds)
        bar.bind("<B1-Motion>",    self._dm)
        self.ic=ImageCanvas(self,bg="#000")
        self.ic.pack(fill="both",expand=True)
        self.ic.set_image(self._img)
        rh=tk.Label(self.ic,text="◢",bg="#0b0b14",fg=ACCENT,
                    font=(_FB,14),cursor="sizing")
        rh.place(relx=1,rely=1,anchor="se")
        rh.bind("<ButtonPress-1>",self._rs)
        rh.bind("<B1-Motion>",    self._rm)

    def _ds(self,e): self._dx=e.x_root-self.winfo_x();self._dy=e.y_root-self.winfo_y()
    def _dm(self,e): self.geometry(f"+{e.x_root-self._dx}+{e.y_root-self._dy}")
    def _rs(self,e): self._rx=e.x_root;self._ry=e.y_root;self._rw=self.winfo_width();self._rh=self.winfo_height()
    def _rm(self,e):
        nw=max(200,self._rw+(e.x_root-self._rx)); nh=max(150,self._rh+(e.y_root-self._ry))
        self.geometry(f"{nw}x{nh}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main App
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        global _FB
        _FB = ("Segoe UI"   if sys.platform=="win32"
               else "SF Pro Text" if sys.platform=="darwin"
               else "DejaVu Sans")

        self.title("TraceForge Studio by Otaku")
        self.configure(bg=BG)
        self.geometry("1380x920")
        self.minsize(1050,700)

        # State
        self.orig_pil     = None
        self.orig_cv      = None
        self.lineart_pil  = None
        self.overlay_rgba = None
        self.pal_rgb      = None
        self.pal_cnt      = None
        self.colour_map   = None
        self.draw_steps   = []     # list of step dicts
        self._proj        = None
        self._cur_step    = 0

        self._build_ui()

    # ═════════════════════════════════════════════════════════════════════════
    #  UI
    # ═════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self._topbar()
        body=tk.Frame(self,bg=BG)
        body.pack(fill="both",expand=True)
        self._sidebar(body)
        tk.Frame(body,bg=BORDER,width=1).pack(side="left",fill="y")
        self._tabs(body)

    # ── Top bar ───────────────────────────────────────────────────────────────

    def _topbar(self):
        bar=tk.Frame(self,bg=PANEL,height=58)
        bar.pack(fill="x"); bar.pack_propagate(False)
        lg=tk.Frame(bar,bg=PANEL)
        lg.pack(side="left",padx=20,pady=10)
        tk.Label(lg,text="✦",bg=PANEL,fg=ACCENT,
                 font=(_FB,20,"bold")).pack(side="left")
        tf=tk.Frame(lg,bg=PANEL)
        tf.pack(side="left",padx=8)
        tk.Label(tf,text="TraceForge Studio by Otaku",
                 bg=PANEL,fg=TEXT,font=_f(14,"bold")).pack(anchor="w")
        tk.Label(tf,text="line art  ·  colour guide  ·  colour map  ·  trace steps  ·  projector",
                 bg=PANEL,fg=MUTED,font=_f(8)).pack(anchor="w")
        self.status_var=tk.StringVar(value="Ready — open an image to begin")
        chip=tk.Frame(bar,bg=CARD,padx=16,pady=6)
        chip.pack(side="right",padx=20,pady=14)
        tk.Label(chip,textvariable=self.status_var,
                 bg=CARD,fg=BLUE,font=_f(8)).pack()

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _sidebar(self, parent):
        sb=tk.Frame(parent,bg=PANEL,width=268)
        sb.pack(side="left",fill="y"); sb.pack_propagate(False)

        vc=tk.Canvas(sb,bg=PANEL,highlightthickness=0)
        vs=ttk.Scrollbar(sb,orient="vertical",command=vc.yview)
        vc.configure(yscrollcommand=vs.set)
        vs.pack(side="right",fill="y")
        vc.pack(side="left",fill="both",expand=True)
        inner=tk.Frame(vc,bg=PANEL)
        win=vc.create_window((0,0),window=inner,anchor="nw")
        inner.bind("<Configure>",lambda e:vc.configure(scrollregion=vc.bbox("all")))
        vc.bind("<Configure>",   lambda e:vc.itemconfig(win,width=e.width))

        def sec(label, col=MUTED):
            f=tk.Frame(inner,bg=PANEL)
            f.pack(fill="x",padx=18,pady=(16,4))
            tk.Frame(f,bg=BORDER,height=1).pack(fill="x",pady=(0,5))
            tk.Label(f,text=label,bg=PANEL,fg=col,font=_f(8,"bold")).pack(anchor="w")

        def hint(txt):
            tk.Label(inner,text=txt,bg=PANEL,fg=MUTED,font=_f(8),
                     anchor="w",justify="left",wraplength=218
                     ).pack(anchor="w",padx=18,pady=(0,4))

        def sep():
            tk.Frame(inner,bg=BORDER,height=1).pack(fill="x",padx=18,pady=5)

        # ① Load
        sec("①  LOAD IMAGE")
        self._btn(inner,"Open Image",self._load_image,icon="📂",style="primary")
        hint("PNG · JPG · WEBP — all supported via Pillow")
        sep()

        # ② Settings
        sec("②  SETTINGS")
        tk.Label(inner,text="Line Softness",bg=PANEL,fg=SUB,font=_f(9)
                 ).pack(anchor="w",padx=18)
        self.line_var=tk.IntVar(value=21)
        self._scale(inner,self.line_var,5,51,2)
        tk.Label(inner,text="Colour Zones",bg=PANEL,fg=SUB,font=_f(9)
                 ).pack(anchor="w",padx=18,pady=(10,0))
        self.pal_var=tk.IntVar(value=10)
        self._scale(inner,self.pal_var,4,20,1)
        sep()

        # ③ Generate
        sec("③  GENERATE")
        self._btn(inner,"Generate All",self._generate,icon="⚡",style="primary")
        self.prog_var=tk.DoubleVar(value=0)
        self.prog_lbl=tk.StringVar(value="")
        sty=ttk.Style(); sty.theme_use("default")
        sty.configure("Slim.Horizontal.TProgressbar",
                      troughcolor=CARD,background=ACCENT,
                      thickness=5,borderwidth=0,lightcolor=ACCENT,darkcolor=ACCENT)
        ttk.Progressbar(inner,variable=self.prog_var,maximum=100,
                        style="Slim.Horizontal.TProgressbar"
                        ).pack(fill="x",padx=18,pady=(8,2))
        tk.Label(inner,textvariable=self.prog_lbl,bg=PANEL,fg=MUTED,
                 font=_f(7)).pack(anchor="w",padx=18)
        sep()

        # ④ Colour Guide overlay controls
        sec("④  COLOUR GUIDE OVERLAY")
        self.overlay_var=tk.BooleanVar(value=False)
        tk.Checkbutton(inner,text="Show Edge Overlay",variable=self.overlay_var,
                       bg=PANEL,fg=TEXT,selectcolor=CARD,
                       activebackground=PANEL,font=_f(9),
                       command=self._refresh_guide).pack(anchor="w",padx=18)
        tk.Label(inner,text="Overlay Opacity",bg=PANEL,fg=SUB,font=_f(9)
                 ).pack(anchor="w",padx=18,pady=(8,0))
        self.ov_alpha=tk.IntVar(value=60)
        tk.Scale(inner,from_=10,to=100,orient="horizontal",variable=self.ov_alpha,
                 bg=PANEL,fg=TEXT,troughcolor=CARD,highlightthickness=0,
                 activebackground=ACCENT,relief="flat",sliderlength=16,width=8,
                 command=lambda _:self._refresh_guide()
                 ).pack(fill="x",padx=18)
        sep()

        # ⑤ Export
        sec("⑤  EXPORT")
        self._btn(inner,"Save Line Art",          self._save_line,    icon="📄")
        self._btn(inner,"Save Overlay (transparent)",self._save_overlay_t,icon="💾")
        self._btn(inner,"Save Composite PNG",     self._save_overlay_c,icon="🖼️")
        self._btn(inner,"Export Trace Steps",     self._export_steps, icon="📋")
        sep()

        # ⑥ Projector
        sec("⑥  PROJECTOR")
        self._btn(inner,"Launch Projector",self._launch_proj,icon="📽️",style="blue")
        hint("Float colour guide over Krita.\nDrag to reposition, slider = opacity.")

        tk.Frame(inner,bg=PANEL,height=20).pack()

    # ── Tabs ──────────────────────────────────────────────────────────────────

    def _tabs(self, parent):
        wrap=tk.Frame(parent,bg=BG)
        wrap.pack(fill="both",expand=True)

        sty=ttk.Style()
        sty.configure("A.TNotebook",background=BG,borderwidth=0,tabmargins=0)
        sty.configure("A.TNotebook.Tab",background=CARD,foreground=SUB,
                      font=_f(9),padding=[18,9])
        sty.map("A.TNotebook.Tab",
                background=[("selected",ACCENT)],
                foreground=[("selected","white")])

        self.nb=ttk.Notebook(wrap,style="A.TNotebook")
        self.nb.pack(fill="both",expand=True,padx=10,pady=10)

        # Tab 0 — Original
        t0=tk.Frame(self.nb,bg=BG)
        self.nb.add(t0,text="  Original  ")
        self.ic_orig=ImageCanvas(t0)
        self.ic_orig.pack(fill="both",expand=True)
        self._welcome(t0)

        # Tab 1 — Line Art
        t1=tk.Frame(self.nb,bg=BG)
        self.nb.add(t1,text="  Line Art  ")
        self.ic_line=ImageCanvas(t1)
        self.ic_line.pack(fill="both",expand=True)

        # Tab 2 — Colour Guide (original + optional edge overlay + palette strip)
        t2=tk.Frame(self.nb,bg=BG)
        self.nb.add(t2,text="  Colour Guide  ")
        self.ic_guide=ImageCanvas(t2)
        self.ic_guide.pack(fill="both",expand=True)
        pw=tk.Frame(t2,bg=BG,height=100); pw.pack(fill="x",side="bottom")
        pw.pack_propagate(False)
        self.pal_strip=PaletteStrip(pw,bg=BG)
        self.pal_strip.pack(fill="both",expand=True,padx=12,pady=6)

        # Tab 3 — Colour Map (flat zones + hex labels)
        t3=tk.Frame(self.nb,bg=BG)
        self.nb.add(t3,text="  Colour Map  ")
        self._build_colourmap_tab(t3)

        # Tab 4 — Trace Steps
        t4=tk.Frame(self.nb,bg=BG)
        self.nb.add(t4,text="  Trace Steps  ")
        self._build_trace_tab(t4)

    # ── Colour Map tab ────────────────────────────────────────────────────────

    def _build_colourmap_tab(self, parent):
        info=tk.Frame(parent,bg=PANEL,height=46)
        info.pack(fill="x"); info.pack_propagate(False)
        tk.Label(info,
                 text="Each colour zone shows its exact hex value — use this as your colouring reference.",
                 bg=PANEL,fg=SUB,font=_f(9),pady=14).pack(side="left",padx=16)

        self.ic_cmap=ImageCanvas(parent)
        self.ic_cmap.pack(fill="both",expand=True)

    # ── Trace Steps tab ───────────────────────────────────────────────────────

    def _build_trace_tab(self, parent):
        # Top navigation bar
        nav=tk.Frame(parent,bg=PANEL,height=52)
        nav.pack(fill="x"); nav.pack_propagate(False)

        self.step_lbl=tk.Label(nav,text="Step — / —",bg=PANEL,fg=ACCENT,
                               font=_f(11,"bold"))
        self.step_lbl.pack(side="left",padx=20,pady=14)

        btn_frame=tk.Frame(nav,bg=PANEL)
        btn_frame.pack(side="right",padx=16,pady=10)
        self._nav_btn(btn_frame,"◀  Prev",self._step_prev,side="left")
        self._nav_btn(btn_frame,"Next  ▶",self._step_next,side="left",primary=True)

        # Step indicator dots
        self.dot_frame=tk.Frame(nav,bg=PANEL)
        self.dot_frame.pack(side="left",padx=12,pady=18)
        self._dot_labels=[]

        # Main area: image left, instructions right
        body=tk.Frame(parent,bg=BG)
        body.pack(fill="both",expand=True)

        # Image canvas (left, 70%)
        img_frame=tk.Frame(body,bg=BG)
        img_frame.pack(side="left",fill="both",expand=True)
        self.ic_step=ImageCanvas(img_frame)
        self.ic_step.pack(fill="both",expand=True,padx=(10,4),pady=10)

        # Instruction panel (right, fixed 320px)
        self.instr_frame=tk.Frame(body,bg=PANEL,width=320)
        self.instr_frame.pack(side="right",fill="y")
        self.instr_frame.pack_propagate(False)
        tk.Frame(self.instr_frame,bg=BORDER,width=1).pack(side="left",fill="y")

        instr_inner=tk.Frame(self.instr_frame,bg=PANEL)
        instr_inner.pack(fill="both",expand=True,padx=20,pady=20)

        tk.Label(instr_inner,text="HOW TO DRAW THIS STEP",
                 bg=PANEL,fg=MUTED,font=_f(8,"bold")).pack(anchor="w",pady=(0,12))

        self.step_title_lbl=tk.Label(instr_inner,text="",bg=PANEL,fg=ACCENT,
                                     font=_f(11,"bold"),anchor="w",wraplength=270,
                                     justify="left")
        self.step_title_lbl.pack(anchor="w",pady=(0,10))

        tk.Frame(instr_inner,bg=BORDER,height=1).pack(fill="x",pady=(0,14))

        self.instr_text=tk.Label(instr_inner,text="",bg=PANEL,fg=TEXT,
                                 font=_f(10),anchor="nw",wraplength=270,
                                 justify="left")
        self.instr_text.pack(anchor="nw",fill="x")

        # Colour indicator
        tk.Frame(instr_inner,bg=BORDER,height=1).pack(fill="x",pady=(20,10))
        cf=tk.Frame(instr_inner,bg=PANEL)
        cf.pack(anchor="w")
        tk.Label(cf,text="Layer colour: ",bg=PANEL,fg=MUTED,font=_f(8)
                 ).pack(side="left")
        self.step_colour_swatch=tk.Label(cf,text="      ",bg="#888888",
                                         relief="solid",bd=1)
        self.step_colour_swatch.pack(side="left")

        # Progress bar for steps
        tk.Frame(instr_inner,bg=BORDER,height=1).pack(fill="x",pady=(14,8))
        self.step_prog=ttk.Progressbar(instr_inner,maximum=len(STEP_META),
                                       style="Slim.Horizontal.TProgressbar")
        self.step_prog.pack(fill="x")
        self.step_prog_lbl=tk.Label(instr_inner,text="",bg=PANEL,fg=MUTED,font=_f(8))
        self.step_prog_lbl.pack(anchor="w",pady=(4,0))

    # ── Step navigation ───────────────────────────────────────────────────────

    def _show_step(self, idx):
        if not self.draw_steps: return
        n=len(self.draw_steps)
        idx=max(0,min(idx,n-1))
        self._cur_step=idx
        step=self.draw_steps[idx]

        self.ic_step.set_image(step["image"])
        self.step_lbl.config(text=f"Step {idx+1} of {n}")
        self.step_title_lbl.config(text=step["title"])
        self.instr_text.config(text=step["instruction"])
        col="#{:02x}{:02x}{:02x}".format(*step["colour"])
        self.step_colour_swatch.config(bg=col)
        self.step_prog["value"]=idx+1
        self.step_prog_lbl.config(text=f"{idx+1} / {n} steps complete")

        # Update dots
        for i,dot in enumerate(self._dot_labels):
            if i<idx:   dot.config(bg=GREEN,   fg=GREEN)
            elif i==idx: dot.config(bg=ACCENT,  fg=ACCENT)
            else:        dot.config(bg=BORDER,  fg=BORDER)

    def _step_next(self):
        self._show_step(self._cur_step+1)

    def _step_prev(self):
        self._show_step(self._cur_step-1)

    def _build_step_dots(self, n):
        """Rebuild dot indicators after generation."""
        for w in self.dot_frame.winfo_children():
            w.destroy()
        self._dot_labels=[]
        for i in range(n):
            d=tk.Label(self.dot_frame,text="●",bg=BORDER,fg=BORDER,
                       font=(_FB,8))
            d.pack(side="left",padx=2)
            self._dot_labels.append(d)

    # ── Welcome ───────────────────────────────────────────────────────────────

    def _welcome(self, parent):
        self._hint=tk.Label(parent,
            text=("📂   Open an anime image to begin\n\n"
                  "──  Tabs  ──\n\n"
                  "Original     — exact source image\n"
                  "Line Art     — pencil sketch to trace\n"
                  "Colour Guide — original + zone boundaries\n"
                  "Colour Map   — flat zones with hex labels\n"
                  "Trace Steps  — step-by-step drawing guide"),
            bg=BG,fg=MUTED,font=_f(11),justify="center")
        self._hint.place(relx=0.5,rely=0.5,anchor="center")

    # ═════════════════════════════════════════════════════════════════════════
    #  Widget factories
    # ═════════════════════════════════════════════════════════════════════════

    def _btn(self, parent, text, cmd, icon="", style="normal"):
        S={"primary":(ACCENT,"white","#ff7088"),
           "blue":   (BLUE,  BG,    "#80dfff"),
           "normal": (CARD,  TEXT,  HOVER)}
        bg,fg,hov=S.get(style,S["normal"])
        lbl=f"{icon}  {text}" if icon else text
        b=tk.Button(parent,text=lbl,command=cmd,
                    bg=bg,fg=fg,font=_f(9),relief="flat",cursor="hand2",
                    activebackground=hov,activeforeground="white",
                    padx=14,pady=10,anchor="w",borderwidth=0,highlightthickness=0)
        b.pack(fill="x",padx=18,pady=3)
        b.bind("<Enter>",lambda e,b=b,h=hov:b.config(bg=h))
        b.bind("<Leave>",lambda e,b=b,c=bg: b.config(bg=c))
        return b

    def _nav_btn(self, parent, text, cmd, side="left", primary=False):
        bg=ACCENT if primary else CARD
        fg="white" if primary else TEXT
        hov="#ff7088" if primary else HOVER
        b=tk.Button(parent,text=text,command=cmd,
                    bg=bg,fg=fg,font=_f(9),relief="flat",cursor="hand2",
                    activebackground=hov,activeforeground="white",
                    padx=14,pady=6,borderwidth=0,highlightthickness=0)
        b.pack(side=side,padx=4)
        return b

    def _scale(self, parent, var, lo, hi, res):
        tk.Scale(parent,from_=lo,to=hi,resolution=res,orient="horizontal",
                 variable=var,bg=PANEL,fg=TEXT,troughcolor=CARD,
                 highlightthickness=0,activebackground=ACCENT,
                 relief="flat",sliderlength=16,width=8
                 ).pack(fill="x",padx=18)

    # ═════════════════════════════════════════════════════════════════════════
    #  Core actions
    # ═════════════════════════════════════════════════════════════════════════

    def _load_image(self):
        p=filedialog.askopenfilename(
            title="Select anime image",
            filetypes=[("Images","*.png *.jpg *.jpeg *.webp *.bmp"),("All","*.*")])
        if not p: return
        try:
            pil,bgr=load_image_pil(p)
        except Exception as ex:
            messagebox.showerror("Load Error",str(ex)); return
        self.orig_pil=pil; self.orig_cv=bgr
        try: self._hint.place_forget()
        except: pass
        self.ic_orig.set_image(pil)
        self.nb.select(0)
        self._status(f"Loaded  ·  {os.path.basename(p)}  ({pil.width}×{pil.height})")

    def _generate(self):
        if self.orig_cv is None:
            messagebox.showwarning("No image","Load an image first."); return
        self._prog(0,"Starting…")
        threading.Thread(target=self._process,daemon=True).start()

    def _process(self):
        try:
            img=self.orig_cv; pil=self.orig_pil

            self.after(0,self._prog,8,"Sketching line art…")
            self.lineart_pil=make_sketch(img,self.line_var.get())

            self.after(0,self._prog,25,"Extracting palette…")
            pal,cnt,lmap=extract_colours(img,n=self.pal_var.get())
            self.pal_rgb=pal; self.pal_cnt=cnt; self.label_map=lmap

            self.after(0,self._prog,42,"Building edge overlay…")
            self.overlay_rgba=build_edge_overlay(lmap)

            self.after(0,self._prog,55,"Building colour map…")
            self.colour_map=build_colour_map(pil,pal,lmap)

            self.after(0,self._prog,68,"Generating trace steps…")
            self.draw_steps=build_drawing_steps(img)

            self.after(0,self._prog,95,"Rendering…")
            self.after(0,self._show_results)

        except Exception as ex:
            import traceback; traceback.print_exc()
            self.after(0,lambda:messagebox.showerror("Error",str(ex)))
            self.after(0,self._prog,0,"Error")

    def _show_results(self):
        self._prog(100,"Done ✓")

        # Line Art
        self.ic_line.set_image(self.lineart_pil)

        # Colour Guide — original image (overlay applied only if checkbox ON)
        self._refresh_guide()
        self.pal_strip.set_palette(self.pal_rgb,self.pal_cnt)

        # Colour Map
        self.ic_cmap.set_image(self.colour_map)

        # Trace Steps
        self._build_step_dots(len(self.draw_steps))
        self._cur_step=0
        self._show_step(0)

        self.nb.select(1)
        self._status("Done ✓  Use Trace Steps tab to start drawing!")

    # ── Colour Guide refresh ──────────────────────────────────────────────────

    def _refresh_guide(self):
        if self.orig_pil is None: return
        if self.overlay_var.get() and self.overlay_rgba is not None:
            preview=composite_over(self.orig_pil,self.overlay_rgba,
                                   self.ov_alpha.get()/100)
        else:
            preview=self.orig_pil.copy()
        self.ic_guide.set_image(preview)

    # ── Projector ─────────────────────────────────────────────────────────────

    def _launch_proj(self):
        if self.orig_pil is None:
            messagebox.showwarning("Not ready","Generate first."); return
        img=(composite_over(self.orig_pil,self.overlay_rgba,
                             self.ov_alpha.get()/100)
             if self.overlay_var.get() and self.overlay_rgba
             else self.orig_pil.copy())
        if self._proj and self._proj.winfo_exists():
            self._proj.ic.set_image(img); self._proj.lift(); return
        self._proj=Projector(self,img)

    # ── Save / export ─────────────────────────────────────────────────────────

    def _save_line(self):
        if not self.lineart_pil:
            messagebox.showwarning("Nothing to save","Generate first."); return
        p=filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG","*.png")],title="Save Line Art")
        if p: self.lineart_pil.save(p); messagebox.showinfo("Saved ✓","Line art saved!")

    def _save_overlay_t(self):
        if not self.overlay_rgba:
            messagebox.showwarning("Nothing to save","Generate first."); return
        p=filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG","*.png")],title="Save Transparent Overlay")
        if p:
            self.overlay_rgba.save(p)
            messagebox.showinfo("Saved ✓","Transparent overlay saved!\n"
                                "Import as a layer above your image in Krita.")

    def _save_overlay_c(self):
        if not self.orig_pil or not self.overlay_rgba:
            messagebox.showwarning("Nothing to save","Generate first."); return
        p=filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG","*.png")],title="Save Composite PNG")
        if p:
            composite_over(self.orig_pil,self.overlay_rgba,
                           self.ov_alpha.get()/100).save(p)
            messagebox.showinfo("Saved ✓","Composite saved!")

    def _export_steps(self):
        if not self.draw_steps:
            messagebox.showwarning("Nothing to export","Generate first."); return
        folder=filedialog.askdirectory(title="Choose export folder")
        if not folder: return
        for i,step in enumerate(self.draw_steps):
            fname=f"step_{i+1:02d}_{step['title'].split('—')[0].strip().replace(' ','_').lower()}.png"
            step["image"].save(os.path.join(folder,fname))
        messagebox.showinfo("Exported ✓",
            f"{len(self.draw_steps)} step images saved to:\n{folder}")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _status(self,t): self.status_var.set(t)

    def _prog(self,v,label=""):
        self.prog_var.set(v); self.prog_lbl.set(label)
        self.update_idletasks()


# ══════════════════════════════════════════════════════════════════════════════
if __name__=="__main__":
    app=App()
    app.mainloop()
