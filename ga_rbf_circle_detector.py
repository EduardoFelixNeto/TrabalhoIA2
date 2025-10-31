
# ga_rbf_circle_detector_v2.py
# ----------------------------------------------------
# GA + RBFNN detector (improved)
# - Adaptive threshold (Otsu) on RBF output (or fixed fallback)
# - Fitness = IoU + α*Recall − β*FP − γ*parsimony − δ*diversity
# - Diversity penalty repels RBF centers to avoid mode collapse (units empilhadas no mesmo círculo)
# - Slightly relaxed sparsity to not perder bolas pequenas
# ----------------------------------------------------
import os
import math
import random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageOps

# ---------------- Config ----------------
IMG_DIR = "./images_2"
OUT_DIR = "./output"
DOWNSCALE_TO = 256
K_UNITS = 10             # allow a couple of extra units
POP_SIZE = 50
GENERATIONS = 80
TOURNAMENT_K = 3
CROSSOVER_PROB = 0.8
MUTATION_RATE = 0.12
MUTATION_SIGMA_POS = 5.0
MUTATION_SIGMA_SIG = 0.08
MUTATION_SIGMA_A   = 0.12
RANDOM_SEED = 123

# Fitness weights
LAMBDA_FP = 0.35         # false positive weight
LAMBDA_SP = 0.008        # sparsity (slightly weaker than v1)
ALPHA_REC = 0.18         # recall reward
DELTA_DIV = 0.06         # diversity penalty strength
DIV_RHO_FRAC = 0.12      # diversity length scale as fraction of min(w,h)
THRESH_OUT_FALLBACK = 0.42
BIAS_RANGE = (-0.1, 1.0)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

@dataclass
class RBFGenome:
    genes: np.ndarray  # [cx, cy, log_sigma, amplitude]*K + [bias]
    w: int
    h: int

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)

def load_first_image(path: str) -> Image.Image:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    for fname in os.listdir(path):
        if fname.lower().endswith(exts):
            return Image.open(os.path.join(path, fname)).convert("L")
    raise FileNotFoundError(f"No images found in {path}.")

def downscale(im: Image.Image, max_dim: int) -> Image.Image:
    w, h = im.size
    scale = max(w, h) / max_dim if max(w, h) > max_dim else 1.0
    return im.resize((int(round(w/scale)), int(round(h/scale))), Image.LANCZOS) if scale > 1 else im

def binarize_target(im: Image.Image) -> np.ndarray:
    # invert so black -> 1, white -> 0
    inv = ImageOps.invert(im)
    arr = np.asarray(inv, dtype=np.float32) / 255.0
    thr = max(0.15, min(0.85, arr.mean()*0.8))
    return (arr > thr).astype(np.float32)

def init_genome(w: int, h: int, k_units: int) -> RBFGenome:
    genes = []
    rmin, rmax = 0.03*min(w,h), 0.35*min(w,h)
    for _ in range(k_units):
        cx = np.random.uniform(0, w)
        cy = np.random.uniform(0, h)
        log_sigma = np.random.uniform(math.log(rmin), math.log(rmax))
        a = np.clip(np.random.normal(0.8, 0.25), 0.0, 1.5)
        genes.extend([cx, cy, log_sigma, a])
    bias = np.random.uniform(*BIAS_RANGE)
    genes.append(bias)
    return RBFGenome(np.array(genes, dtype=np.float32), w, h)

def decode_params(genome: RBFGenome):
    g = genome.genes
    params = []
    for i in range(0, 4*K_UNITS, 4):
        cx, cy, log_sigma, a = g[i:i+4]
        params.append((cx, cy, math.exp(log_sigma), a))
    bias = g[4*K_UNITS]
    return params, bias

def rbfn_forward(genome: RBFGenome) -> np.ndarray:
    params, bias = decode_params(genome)
    h, w = genome.h, genome.w
    yy, xx = np.mgrid[0:h, 0:w]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    out = np.full((h, w), bias, dtype=np.float32)
    for cx, cy, sigma, a in params:
        dx = xx - cx
        dy = yy - cy
        g = np.exp(-(dx*dx + dy*dy) / (2.0 * (sigma*sigma) + 1e-9))
        out += a * g
    return 1.0 / (1.0 + np.exp(-out))

def iou(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    inter = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    return (inter / union) if union else (1.0 if inter == 0 else 0.0)

def recall(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    tp = np.logical_and(pred_mask, target_mask).sum()
    pos = target_mask.sum()
    return (tp / pos) if pos else 1.0

def otsu_threshold(values: np.ndarray) -> float:
    # values in [0,1]; classic Otsu on histogram
    hist, bin_edges = np.histogram(values.ravel(), bins=256, range=(0,1))
    hist = hist.astype(np.float64)
    prob = hist / (hist.sum() + 1e-12)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t*omega - mu)**2 / (omega*(1.0 - omega) + 1e-12)
    idx = np.nanargmax(sigma_b2)
    thr = (idx + 0.5) / 256.0
    return float(thr)

def diversity_penalty(genome: RBFGenome) -> float:
    params, _ = decode_params(genome)
    centers = np.array([[cx, cy] for (cx, cy, _, _) in params], dtype=np.float32)
    # pairwise gaussian attraction (we penalize closeness)
    if centers.shape[0] < 2:
        return 0.0
    rho = DIV_RHO_FRAC * min(genome.w, genome.h)
    inv_two_rho2 = 1.0 / (2.0 * rho * rho + 1e-9)
    pen = 0.0
    for i in range(centers.shape[0]):
        for j in range(i+1, centers.shape[0]):
            d2 = ((centers[i] - centers[j])**2).sum()
            pen += math.exp(-d2 * inv_two_rho2)
    # normalize roughly by number of pairs
    n_pairs = centers.shape[0]*(centers.shape[0]-1)/2.0
    return pen / (n_pairs + 1e-9)

def fitness(genome: RBFGenome, target_mask: np.ndarray) -> float:
    out = rbfn_forward(genome)
    # adaptive threshold (Otsu); fallback if degenerate
    thr = otsu_threshold(out)
    if not (0.05 <= thr <= 0.95):
        thr = THRESH_OUT_FALLBACK
    pred_mask = (out >= thr)

    iou_val = iou(pred_mask, target_mask)
    rec_val = recall(pred_mask, target_mask)
    fp = np.logical_and(pred_mask, ~target_mask.astype(bool)).sum() / (target_mask.size + 1e-9)
    amps = genome.genes[3:4*K_UNITS:4]
    sparsity = np.abs(amps).sum() / (K_UNITS + 1e-9)
    div = diversity_penalty(genome)  # higher -> worse (more overlap), so subtract

    return iou_val + ALPHA_REC*rec_val - LAMBDA_FP*fp - LAMBDA_SP*sparsity - DELTA_DIV*div

def clip_genes(g: np.ndarray, w: int, h: int):
    for i in range(0, 4*K_UNITS, 4):
        g[i]   = np.clip(g[i],   0, w-1)
        g[i+1] = np.clip(g[i+1], 0, h-1)
        g[i+2] = np.clip(g[i+2], math.log(0.02*min(w,h)), math.log(0.5*min(w,h)))
        g[i+3] = np.clip(g[i+3], 0.0, 2.0)
    g[4*K_UNITS] = np.clip(g[4*K_UNITS], *BIAS_RANGE)

def tournament_select(pop: List[RBFGenome], fits: List[float]) -> RBFGenome:
    idxs = np.random.choice(len(pop), size=TOURNAMENT_K, replace=False)
    best = max(idxs, key=lambda j: fits[j])
    return RBFGenome(pop[best].genes.copy(), pop[best].w, pop[best].h)

def crossover(p1: RBFGenome, p2: RBFGenome):
    g1, g2 = p1.genes.copy(), p2.genes.copy()
    if np.random.rand() < CROSSOVER_PROB:
        alpha = np.random.rand(g1.size)
        a = alpha * g1 + (1-alpha) * g2
        b = alpha * g2 + (1-alpha) * g1
        g1, g2 = a, b
    return RBFGenome(g1, p1.w, p1.h), RBFGenome(g2, p2.w, p2.h)

def mutate(ind: RBFGenome):
    g = ind.genes
    w, h = ind.w, ind.h
    for i in range(0, 4*K_UNITS, 4):
        if np.random.rand() < MUTATION_RATE: g[i]   += np.random.normal(0, MUTATION_SIGMA_POS)
        if np.random.rand() < MUTATION_RATE: g[i+1] += np.random.normal(0, MUTATION_SIGMA_POS)
        if np.random.rand() < MUTATION_RATE: g[i+2] += np.random.normal(0, MUTATION_SIGMA_SIG)
        if np.random.rand() < MUTATION_RATE: g[i+3] += np.random.normal(0, MUTATION_SIGMA_A)
    if np.random.rand() < MUTATION_RATE: g[4*K_UNITS] += np.random.normal(0, 0.05)
    clip_genes(g, w, h)

def connected_components(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not visited[y, x]:
                q = [(y, x)]
                visited[y, x] = True
                pts = []
                while q:
                    cy, cx = q.pop()
                    pts.append((cy, cx))
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                comps.append(np.array(pts, dtype=np.int32))
    return comps

def fit_circle_kasa(points: np.ndarray):
    if points.shape[0] < 3:
        return float('nan'), float('nan'), float('nan')
    x = points[:,1].astype(np.float64)
    y = points[:,0].astype(np.float64)
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x*x + y*y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = math.sqrt(max(c + cx*cx + cy*cy, 0.0))
    return cx, cy, r

def run_once():
    ensure_dirs() # make output dir
    img = load_first_image(IMG_DIR) # load first image
    img = downscale(img, DOWNSCALE_TO) # downscale if needed
    w, h = img.size # get size
    target = binarize_target(img)

    pop = [init_genome(w, h, K_UNITS) for _ in range(POP_SIZE)]
    fits = [fitness(ind, target) for ind in pop]
    best_idx = int(np.argmax(fits))
    best, best_fit = pop[best_idx], fits[best_idx]
    print(f"Init best fitness: {best_fit:.4f}")

    for gen in range(1, GENERATIONS+1):
        new_pop = []
        # elitism
        elite = RBFGenome(pop[best_idx].genes.copy(), w, h)
        new_pop.append(elite)
        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(pop, fits)
            p2 = tournament_select(pop, fits)
            c1, c2 = crossover(p1, p2)
            mutate(c1); mutate(c2)
            new_pop.extend([c1, c2])
        pop = new_pop[:POP_SIZE]
        fits = [fitness(ind, target) for ind in pop]
        best_idx = int(np.argmax(fits))
        if fits[best_idx] > best_fit:
            best_fit = fits[best_idx]
            best = pop[best_idx]
        if gen % 10 == 0:
            print(f"Gen {gen:03d} | best={best_fit:.4f}")

    # Final forward and adaptive threshold
    out = rbfn_forward(best)
    thr = otsu_threshold(out)
    if not (0.05 <= thr <= 0.95):
        thr = THRESH_OUT_FALLBACK
    pred_mask = (out >= thr)

    # Components and circle fitting
    comps = connected_components(pred_mask)
    # Keep components with area >= 0.001 of image
    min_area = 0.001 * (w*h)
    comps = [c for c in comps if c.shape[0] >= min_area]

    circles = []
    for comp in comps:
        cx, cy, r = fit_circle_kasa(comp)
        if not np.isnan(cx) and r > 2:
            circles.append((cx, cy, r))

    print(f"Detected {len(circles)} circle(s).")
    for i, (cx, cy, r) in enumerate(circles, 1):
        print(f"  {i}: center=({cx:.1f}, {cy:.1f}), r={r:.1f}")

    # Save overlays
    import matplotlib.pyplot as plt
    theta = np.linspace(0, 2*np.pi, 256)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig1 = plt.figure()
    plt.imshow(img, cmap="gray")
    for cx, cy, r in circles:
        xs = cx + r*np.cos(theta)
        ys = cy + r*np.sin(theta)
        plt.plot(xs, ys, linewidth=2)
    plt.title(f"Detections: {len(circles)}")
    det_path = os.path.join(OUT_DIR, "detections.png")
    fig1.savefig(det_path, bbox_inches="tight")
    plt.close(fig1)

    fig2 = plt.figure(figsize=(9,3))
    plt.subplot(1,3,1); plt.imshow(target, cmap="gray"); plt.title("Target mask"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(out, cmap="gray"); plt.title(f"RBFNN output (thr={thr:.2f})"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(pred_mask, cmap="gray"); plt.title("Predicted mask"); plt.axis("off")
    viz_path = os.path.join(OUT_DIR, "evolution_result.png")
    fig2.savefig(viz_path, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved visualization to: {det_path} | {viz_path}")

if __name__ == "__main__":
    run_once()
