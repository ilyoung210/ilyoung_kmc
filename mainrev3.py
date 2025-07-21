import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Slider
from matplotlib.patches import Patch, Polygon
from matplotlib.lines import Line2D
import logging
import sys
import csv
import math
import os
from collections import deque
import random

# Voronoi 등에 필요한 라이브러리
from scipy.spatial import cKDTree
from skimage.measure import find_contours

################################################################################
# [0] PDF / 그래프 사이즈, 폰트 등을 통일할 수 있는 전역 설정
################################################################################
PDF_FIG_SIZE = (5.0, 4.0)        # PDF에 들어갈 그림 사이즈 (가로, 세로 인치)
PDF_LABEL_FONT_SIZE = 14         # 축 라벨 폰트 크기
PDF_TICK_FONT_SIZE = 12          # 축 눈금 폰트 크기
PDF_LEGEND_FONT_SIZE = 12        # 범례 폰트 크기

################################################################################
# [A] 전역 설정: 그림 크기, 폰트 사이즈, 색상 팔레트
################################################################################
# (아래 FIG_SIZE, TICK_FONT_SIZE, LABEL_FONT_SIZE 등은
#  "실시간 모드"에서 화면 표시용 크기)
FIG_SIZE = (8.0, 6.0)      # 실시간 모드 plot창 기본 크기 (애니메이션 표시용)
TICK_FONT_SIZE = 14        # 축 눈금 폰트 사이즈(실시간 모드)
LABEL_FONT_SIZE = 16       # 축 라벨/제목 폰트 사이즈(실시간 모드)
LEGEND_FONT_SIZE = 14      # 범례 폰트 사이즈(실시간 모드)

COLOR_LIST_NO_BLUE = ['g','r','m','c','y','k','orange','purple']  # 기타 팔레트
TEMP_COLOR  = 'b'  # 온도축/온도 그래프 파란색

################################################################################
# [B] Potts 4상 (Up=0, Down=1, Mono=2, Tetra=3) 색상
################################################################################
import matplotlib.colors as mcolors
q = 4  # 총 상의 개수

COLOR_UP    = plt.cm.Accent(0/(q-1))  # Up → "O-↑"
COLOR_DOWN  = plt.cm.Accent(1/(q-1))  # Down → "O-↓"
COLOR_MONO  = plt.cm.Accent(2/(q-1))  # Mono → "m"
COLOR_TETRA = plt.cm.Accent(3/(q-1))  # Tetra → "t"

PHASE_LABEL_UP    = "O-↑"
PHASE_LABEL_DOWN  = "O-↓"
PHASE_LABEL_MONO  = "m"
PHASE_LABEL_TETRA = "t"

phase_names = {
    0: 'Up',
    1: 'Down',
    2: 'Mono',
    3: 'Tetra'
}

BFS_CMAP = mcolors.ListedColormap(
    [COLOR_UP, COLOR_DOWN, COLOR_MONO, COLOR_TETRA],
    name='bfs_map'
)

################################################################################
# [C] 시뮬레이션 상수 및 기본 함수
################################################################################
logging.basicConfig(
    filename='simulation.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s'
)

TIME_SCALE = 3.0e16
k_B = 1.380649e-23
E_field = 0.0

# HZO 파라미터
E_p = 4.00e-20
E_m = -0.55e-20
E_t = 6.89e-20
J   = 0.1e-20

a_1 = 3.2e-24
a_2 = 0.0
a_3 = 4.8e-23

sigma_1 = 8.85e-19
sigma_2 = 1.1e-18
sigma_3 = 8.296e-19

interface_pairs = {
    frozenset({0,1}): 0.0,
    frozenset({0,2}): 3.95e-20,
    frozenset({1,2}): 3.95e-20,
    frozenset({0,3}): 5.56e-21,
    frozenset({1,3}): 5.56e-21,
    frozenset({2,3}): 3.19e-20
}

N_for_analysis = 50
h = 100
nu0 = 1.0e13

CELL_SIZE_ANG = 5.0

transition_types = [
    ('Up','Down'),('Down','Up'),
    ('Tetra','Up'),('Tetra','Down'),
    ('Tetra','Mono'),('Mono','Up'),
    ('Mono','Down'),('Mono','Tetra'),
    ('Up','Mono'),('Down','Mono'),
    ('Up','Tetra'),('Down','Tetra'),
]

# -----------------------------------------------------------------------------
# Material parameter sets for doping interpolation
# -----------------------------------------------------------------------------
hf_par = {
    "E_p": 3.5e-20,  "E_m": -0.50e-20, "E_t": 6.0e-20,   "J": 0.08e-20,
    "a_1": 3.0e-24,  "a_2": 0.0,       "a_3": 4.5e-23,
    "sigma_1": 8.0e-19, "sigma_2": 1.0e-18, "sigma_3": 7.5e-19,
    "interface_pairs": {
        frozenset({0,1}): 0.0,
        frozenset({0,2}): 3.5e-20,
        frozenset({1,2}): 3.5e-20,
        frozenset({0,3}): 4.5e-21,
        frozenset({1,3}): 4.5e-21,
        frozenset({2,3}): 2.8e-20
    }
}

hzo_par = {
    "E_p": 4.0e-20,  "E_m": -0.55e-20, "E_t": 6.89e-20, "J": 0.1e-20,
    "a_1": 3.2e-24,  "a_2": 0.0,       "a_3": 4.8e-23,
    "sigma_1": 8.85e-19, "sigma_2": 1.1e-18, "sigma_3": 8.296e-19,
    "interface_pairs": {
        frozenset({0,1}): 0.0,
        frozenset({0,2}): 3.95e-20,
        frozenset({1,2}): 3.95e-20,
        frozenset({0,3}): 5.56e-21,
        frozenset({1,3}): 5.56e-21,
        frozenset({2,3}): 3.19e-20
    }
}

zr_par = {
    "E_p": 4.5e-20,  "E_m": -0.60e-20, "E_t": 7.5e-20,  "J": 0.12e-20,
    "a_1": 3.5e-24,  "a_2": 0.0,       "a_3": 5.0e-23,
    "sigma_1": 9.0e-19, "sigma_2": 1.2e-18, "sigma_3": 9.0e-19,
    "interface_pairs": {
        frozenset({0,1}): 0.0,
        frozenset({0,2}): 4.4e-20,
        frozenset({1,2}): 4.4e-20,
        frozenset({0,3}): 6.0e-21,
        frozenset({1,3}): 6.0e-21,
        frozenset({2,3}): 3.6e-20
    }
}

si_hf_par = {
    "E_p": 3.0e-20,  "E_m": -0.40e-20, "E_t": 5.5e-20,  "J": 0.07e-20,
    "a_1": 2.8e-24,  "a_2": 0.0,       "a_3": 4.0e-23,
    "sigma_1": 7.5e-19, "sigma_2": 0.95e-18, "sigma_3": 7.0e-19,
    "interface_pairs": {
        frozenset({0,1}): 0.0,
        frozenset({0,2}): 3.2e-20,
        frozenset({1,2}): 3.2e-20,
        frozenset({0,3}): 4.0e-21,
        frozenset({1,3}): 4.0e-21,
        frozenset({2,3}): 2.5e-20
    }
}

# 가상의 Al-doped HfO2 파라미터 (예시)
al_hf_par = {
    "E_p": 3.2e-20,  "E_m": -0.45e-20, "E_t": 6.2e-20,  "J": 0.09e-20,
    "a_1": 2.9e-24,  "a_2": 0.0,       "a_3": 4.2e-23,
    "sigma_1": 7.8e-19, "sigma_2": 1.05e-18, "sigma_3": 7.2e-19,
    "interface_pairs": {
        frozenset({0,1}): 0.0,
        frozenset({0,2}): 3.4e-20,
        frozenset({1,2}): 3.4e-20,
        frozenset({0,3}): 4.5e-21,
        frozenset({1,3}): 4.5e-21,
        frozenset({2,3}): 2.6e-20
    }
}

def interpolate_params(parA, parB, ratio):
    new_par = {}
    for k in parA.keys():
        if k == "interface_pairs":
            new_ip = {}
            for fs in parA[k].keys():
                valA = parA[k][fs]
                valB = parB[k].get(fs, valA)
                new_ip[fs] = valA*(1-ratio) + valB*ratio
            new_par[k] = new_ip
        else:
            new_par[k] = parA[k]*(1-ratio) + parB[k]*ratio
    return new_par

def apply_global_params(par):
    global E_p, E_m, E_t, J
    global a_1, a_2, a_3
    global sigma_1, sigma_2, sigma_3
    global interface_pairs

    E_p = par["E_p"]
    E_m = par["E_m"]
    E_t = par["E_t"]
    J   = par["J"]

    a_1 = par["a_1"]
    a_2 = par["a_2"]
    a_3 = par["a_3"]

    sigma_1 = par["sigma_1"]
    sigma_2 = par["sigma_2"]
    sigma_3 = par["sigma_3"]

    interface_pairs = par["interface_pairs"]

def set_material_params_zr_doped(doping_zr):
    if doping_zr <= 0:
        apply_global_params(hf_par)
        return
    if doping_zr >= 100:
        apply_global_params(zr_par)
        return
    if doping_zr == 50:
        apply_global_params(hzo_par)
        return
    if doping_zr < 50:
        ratio = doping_zr / 50.0
        new_par = interpolate_params(hf_par, hzo_par, ratio)
        apply_global_params(new_par)
    else:
        ratio = (doping_zr - 50) / 50.0
        new_par = interpolate_params(hzo_par, zr_par, ratio)
        apply_global_params(new_par)

def set_material_params_si_doped(doping_si):
    if doping_si <= 0:
        apply_global_params(hf_par)
        return
    if doping_si >= 100:
        apply_global_params(si_hf_par)
        return
    ratio = doping_si / 100.0
    new_par = interpolate_params(hf_par, si_hf_par, ratio)
    apply_global_params(new_par)

def set_material_params_al_doped(doping_al):
    if doping_al <= 0:
        apply_global_params(hf_par)
        return
    if doping_al >= 100:
        apply_global_params(al_hf_par)
        return
    ratio = doping_al / 100.0
    new_par = interpolate_params(hf_par, al_hf_par, ratio)
    apply_global_params(new_par)

def initialize_barrier_file(filename='barrier_values.csv'):
    with open(filename,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(["Temperature(K)","Old_State","New_State","Barrier(J)"])

def select_material():
    print("어떤 dopant를 사용하겠습니까?")
    print("1) Zr")
    print("2) Si")
    print("3) Al")
    choice = input("번호를 선택하세요: ").strip()

    doping_str = input("dopant 양을 0~100(%)로 입력하세요: ").strip()
    try:
        doping_val = float(doping_str)
    except Exception:
        doping_val = 0.0
    if doping_val < 0:
        doping_val = 0.0
    if doping_val > 100:
        doping_val = 100.0

    if choice == '1':
        print(f"Zr doping {doping_val}% 로 진행합니다.")
        set_material_params_zr_doped(doping_val)
        logging.info(f"[Material] Using Zr-doped HfO2 with doping={doping_val}%")
    elif choice == '2':
        print(f"Si doping {doping_val}% 로 진행합니다.")
        set_material_params_si_doped(doping_val)
        logging.info(f"[Material] Using Si-doped HfO2 with doping={doping_val}%")
    elif choice == '3':
        print(f"Al doping {doping_val}% 로 진행합니다.")
        set_material_params_al_doped(doping_val)
        logging.info(f"[Material] Using Al-doped HfO2 with doping={doping_val}%")
    else:
        print("잘못된 입력. 기본값(HZO)로 진행합니다.")
        apply_global_params(hzo_par)
        logging.info("[Material] Invalid input => default(HZO) used.")
    return choice, doping_val

def is_Up(s):   return (s==0)
def is_Down(s): return (s==1)
def is_M(s):    return (s==2)
def is_T(s):    return (s==3)

def P_val(s):
    if s==0: return +1
    if s==1: return -1
    return 0

def bulk_energy(s,J,T,h):
    if is_Up(s) or is_Down(s):
        return E_p - T*a_1 + sigma_1/(h/5)
    elif is_M(s):
        return E_m - T*a_2 + sigma_2/(h/5)
    elif is_T(s):
        return E_t - T*a_3 + sigma_3/(h/5)
    return 0.0

def field_energy(s,J):
    return -E_field * P_val(s)

def potts_interaction_energy(s1,s2,J):
    if (is_Up(s1) and is_Up(s2)) or (is_Down(s1) and is_Down(s2)):
        return -J
    elif (is_Up(s1) and is_Down(s2)) or (is_Down(s1) and is_Up(s2)):
        return +J
    return 0.0

def interface_energy(s1,s2,J):
    if s1==s2:
        return 0.0
    return interface_pairs.get(frozenset({s1,s2}), 0.0)

def local_energy(spins, x, y, T, J):
    Nx, Ny = spins.shape
    s= spins[x,y]
    E_loc = bulk_energy(s,J,T,h) + field_energy(s,J)
    neigh = [((x-1)%Nx, y), ((x+1)%Nx, y), (x, (y-1)%Ny), (x, (y+1)%Ny)]
    E_nb=0.0
    for (xx,yy) in neigh:
        s2= spins[xx,yy]
        E_nb += potts_interaction_energy(s,s2,J)
        E_nb += interface_energy(s,s2,J)
    return E_loc + E_nb

def delta_energy(spins, x, y, new_s, T, J):
    old_s = spins[x, y]
    E_old = local_energy(spins, x, y, T, J)
    spins[x, y] = new_s
    E_new = local_energy(spins, x, y, T, J)
    spins[x,y]= old_s
    return E_new - E_old

def total_energy(spins, T, J):
    E_tot = 0.0
    Nx, Ny = spins.shape
    for x in range(Nx):
        for y in range(Ny):
            s= spins[x,y]
            E_tot += bulk_energy(s,J,T,h)
            E_tot += field_energy(s,J)
            s_r= spins[x,(y+1)%Ny]
            s_d= spins[(x+1)%Nx,y]
            E_tot += potts_interaction_energy(s,s_r,J)
            E_tot += potts_interaction_energy(s,s_d,J)
            E_tot += interface_energy(s,s_r,J)
            E_tot += interface_energy(s,s_d,J)
    return E_tot

def polarization(spins):
    return np.mean([P_val(s) for s in spins.flatten()])

def phase_fractions(spins):
    bc= np.bincount(spins.flatten(), minlength=4)
    return bc/spins.size

def get_transition_barrier(old_state,new_state,T,J):
    # 임의 예시값 (필요시 수정 가능)
    if (is_Up(old_state) and is_Down(new_state)) or (is_Down(old_state) and is_Up(new_state)):
        return 0.0
    elif is_T(old_state) and (is_Up(new_state) or is_Down(new_state)):
        return 2.816e-21
    elif is_T(old_state) and is_M(new_state):
        return 1.728e-21
    elif is_M(old_state) and (is_Up(new_state) or is_Down(new_state)):
        return 7.1e-21
    elif ((is_Up(old_state) or is_Down(old_state)) and is_T(new_state)):
        return 2.816e-21
    elif is_M(old_state) and is_T(new_state):
        return 8.728e-21
    elif ((is_Up(old_state) or is_Down(old_state)) and is_M(new_state)):
        return 1.728e-21
    return 1e30

def attempt_flip(spins, x, y, T, J):
    """
    flip 시도 후 (accepted, used_barrier) 반환
    accepted = True / False
    used_barrier = 실제로 accepted일 때 사용한 barrier값, 아니면 None
    """
    old_s= spins[x,y]

    if is_T(old_s):
        poss=[0,1,2]
    elif is_M(old_s):
        poss=[0,1,3]
    elif is_Up(old_s):
        poss=[1,2,3]
    elif is_Down(old_s):
        poss=[0,2,3]
    else:
        return (False, None)

    if not poss:
        return (False, None)

    new_s= np.random.choice(poss)
    dE = delta_energy(spins, x, y, new_s, T, J)
    barrier= get_transition_barrier(old_s,new_s,T,J)

    accepted = False
    if dE < 0:
        # E가 내려가는 전이
        if abs(dE) >= barrier:
            spins[x,y]= new_s
            accepted = True
        else:
            prob= np.exp(-dE/(k_B*T)) if T>1e-9 else 0.0
            if np.random.rand()< prob:
                spins[x,y]= new_s
                accepted = True
    else:
        # E가 올라가는 전이
        prob= np.exp(-dE/(k_B*T)) if T>1e-9 else 0.0
        if np.random.rand()< prob:
            spins[x,y]= new_s
            accepted = True

    if accepted:
        return (True, barrier)
    else:
        return (False, None)

def compute_total_rate(spins, T, J):
    """
    *참고용* 함수.
    이전에는 dt = M / sum_rates 형태로 썼으나,
    지금은 이벤트별로 시간을 더하는 방식으로 변경했으므로
    실제 시간 계산에는 사용하지 않습니다.
    """
    tot=0.0
    Nx, Ny = spins.shape
    for x in range(Nx):
        for y in range(Ny):
            old_s= spins[x,y]
            for new_s in range(q):
                if new_s==old_s:
                    continue
                barrier= get_transition_barrier(old_s,new_s,T,J)
                if barrier<1e30:
                    rate= nu0*np.exp(-barrier/(k_B*T)) if T>1e-9 else 0.0
                    tot+= rate
    return tot

def compute_fluctuation_susceptibility(spins,T,J, n_subsample=10):
    Nsite= spins.shape[0]*spins.shape[1]
    ft_vals=[]
    for _ in range(n_subsample):
        x = np.random.randint(spins.shape[0])
        y = np.random.randint(spins.shape[1])
        attempt_flip(spins, x, y, T, J)  # 샘플링을 위한 임시 flip
        frac_tetra= phase_fractions(spins)[3]
        ft_vals.append(frac_tetra)
    arr= np.array(ft_vals)
    m1= arr.mean()
    m2= (arr**2).mean()
    chi= (Nsite/(k_B*T))*(m2- m1**2) if T>1e-9 else 0.0
    return chi

################################################################################
# [D] BFS + 결정립 분석 함수들
################################################################################
def label_connected_components(spin_arr):
    Nx, Ny = spin_arr.shape
    label_arr = np.zeros((Nx, Ny), dtype=int)
    regions= {}
    cur_label=0

    def bfs_region(x0,y0,s):
        nonlocal cur_label
        cur_label+=1
        queue= deque()
        queue.append((x0,y0))
        label_arr[x0,y0]= cur_label
        regpix= [(x0,y0)]
        while queue:
            x,y= queue.popleft()
            for (nx, ny) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                nx %= Nx
                ny %= Ny
                if spin_arr[nx,ny]==s and label_arr[nx,ny]==0:
                    label_arr[nx,ny]= cur_label
                    queue.append((nx,ny))
                    regpix.append((nx,ny))
        regions[cur_label]= {"spin": s, "pixels": regpix}

    for x in range(Nx):
        for y in range(Ny):
            if label_arr[x,y]==0:
                s= spin_arr[x,y]
                bfs_region(x,y,s)

    return label_arr, regions

def compute_equiv_diameter(pixel_count):
    area_A2= pixel_count*(CELL_SIZE_ANG**2)
    from math import pi, sqrt
    d= math.sqrt(4.0* area_A2/pi)
    return d

def get_region_boundary_pixels(label_arr, lab_id, region_pixels):
    Nx, Ny = label_arr.shape
    bcoords=[]
    for (x,y) in region_pixels:
        neighbors= [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
        is_boundary=False
        for(nx,ny) in neighbors:
            nx %= Nx
            ny %= Ny
            if label_arr[nx,ny]!=lab_id:
                is_boundary=True
                break
        if is_boundary:
            bcoords.append((x,y))
    return bcoords

def marker_for_radius(r):
    if r<10:
        return 'o'
    elif r<20:
        return '^'
    elif r<30:
        return 's'
    else:
        return '*'

################################################################################
# [E] BFS 시각화 + Grain size 분석
################################################################################
def analyze_grains_simple(spin_arr, title_prefix=""):
    label_arr, regions= label_connected_components(spin_arr)
    radius_all= []

    for _, info in regions.items():
        pixcount= len(info["pixels"])
        d_angs= compute_equiv_diameter(pixcount)
        r_angs= d_angs/2.0
        radius_all.append(r_angs)

    global_max= max(radius_all) if radius_all else 1.0

    fig_heat, ax_heat= plt.subplots(figsize=PDF_FIG_SIZE)
    ax_heat.imshow(spin_arr, cmap=BFS_CMAP, origin='upper', vmin=0, vmax=3, aspect='equal')
    ax_heat.axis('off')
    if title_prefix:
        ax_heat.set_title(f"{title_prefix} Heatmap + BFS", fontsize=PDF_LABEL_FONT_SIZE)

    for lbid, inf2 in regions.items():
        ccount= len(inf2["pixels"])
        dd= compute_equiv_diameter(ccount)
        rr= dd/2.0
        bcoords= get_region_boundary_pixels(label_arr, lbid, inf2["pixels"])
        if bcoords:
            mk= marker_for_radius(rr)
            arr_b= np.array(bcoords)
            ys= arr_b[:,1]
            xs= arr_b[:,0]
            ax_heat.plot(ys, xs, marker=mk, color='k', markersize=3, linestyle='none')

    fig_dist, ax_dist= plt.subplots(figsize=PDF_FIG_SIZE)
    if radius_all:
        bins= np.linspace(0, global_max,30)
        ax_dist.hist(radius_all, bins=bins, color='gray', edgecolor='k')
        mean_r= np.mean(radius_all)
        ax_dist.axvline(mean_r, color='r', linestyle='--', label=f"Mean={mean_r:.2f}Å")
        ax_dist.set_xlim(0, global_max*1.05)
    if title_prefix:
        ax_dist.set_title(f"{title_prefix} Grain Size Dist (BFS)", fontsize=PDF_LABEL_FONT_SIZE)
    ax_dist.set_xlabel("Radius(Å)", fontsize=PDF_LABEL_FONT_SIZE)
    ax_dist.set_ylabel("Count", fontsize=PDF_LABEL_FONT_SIZE)
    ax_dist.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
    ax_dist.legend(fontsize=PDF_LEGEND_FONT_SIZE)

    return fig_heat, fig_dist, radius_all

def analyze_grains_with_markers(spin_arr, step_label=""):
    label_arr, regions= label_connected_components(spin_arr)
    radius_dict= {0:[],1:[],2:[],3:[]}
    radius_all= []
    grain_info= []

    for labid, info in regions.items():
        s= info["spin"]
        pixcount= len(info["pixels"])
        d_angs= compute_equiv_diameter(pixcount)
        r_angs= d_angs/2.0
        radius_dict[s].append(r_angs)
        radius_all.append(r_angs)

        bcoords= get_region_boundary_pixels(label_arr, labid, info["pixels"])
        grain_info.append((r_angs,s,bcoords))

    global_max= max(radius_all) if radius_all else 1.0

    fig_heat, ax_heat= plt.subplots(figsize=PDF_FIG_SIZE)
    ax_heat.imshow(spin_arr, cmap=BFS_CMAP, origin='upper', vmin=0, vmax=3, aspect='equal')
    ax_heat.axis('off')
    if step_label:
        ax_heat.set_title(f"Step={step_label} BFS Heatmap", fontsize=PDF_LABEL_FONT_SIZE)
    else:
        ax_heat.set_title("BFS Heatmap", fontsize=PDF_LABEL_FONT_SIZE)

    for (r_angs, s, bcoords) in grain_info:
        if not bcoords:
            continue
        mk= marker_for_radius(r_angs)
        arr_b= np.array(bcoords)
        ys= arr_b[:,1]
        xs= arr_b[:,0]
        ax_heat.plot(ys, xs, marker=mk, color='k', markersize=3, linestyle='none')

    phase_legend= [
        Patch(facecolor=COLOR_TETRA, edgecolor='k', label=PHASE_LABEL_TETRA),
        Patch(facecolor=COLOR_MONO,  edgecolor='k', label=PHASE_LABEL_MONO),
        Patch(facecolor=COLOR_UP,    edgecolor='k', label=PHASE_LABEL_UP),
        Patch(facecolor=COLOR_DOWN,  edgecolor='k', label=PHASE_LABEL_DOWN),
    ]
    size_legend= [
        Line2D([],[], marker='o', color='k', markersize=7, linestyle='none', label='r<10Å'),
        Line2D([],[], marker='^', color='k', markersize=7, linestyle='none', label='10Å<=r<20Å'),
        Line2D([],[], marker='s', color='k', markersize=7, linestyle='none', label='20Å<=r<30Å'),
        Line2D([],[], marker='*', color='k', markersize=7, linestyle='none', label='r>=30Å'),
    ]
    ax_heat.legend(handles=(phase_legend+size_legend), loc='upper left', fontsize=PDF_LEGEND_FONT_SIZE)

    fig_all, ax_all= plt.subplots(figsize=PDF_FIG_SIZE)
    if len(radius_all)>0:
        bins= np.linspace(0, global_max,30)
        ax_all.hist(radius_all, bins=bins, color='gray', edgecolor='k')
        mean_r= np.mean(radius_all)
        ax_all.axvline(mean_r, color='r', linestyle='--', label=f"Mean={mean_r:.2f}Å")
        ax_all.set_xlim(0, global_max*1.05)
    if step_label:
        ax_all.set_title(f"Step={step_label} BFS Grain Size(All)", fontsize=PDF_LABEL_FONT_SIZE)
    else:
        ax_all.set_title("BFS Grain Size(All)", fontsize=PDF_LABEL_FONT_SIZE)
    ax_all.set_xlabel("Radius(Å)", fontsize=PDF_LABEL_FONT_SIZE)
    ax_all.set_ylabel("Count", fontsize=PDF_LABEL_FONT_SIZE)
    ax_all.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
    ax_all.legend(fontsize=PDF_LEGEND_FONT_SIZE)

    fig_phases, axs_ph= plt.subplots(2,2, figsize=(2*PDF_FIG_SIZE[0],2*PDF_FIG_SIZE[1]))

    ph_names= [PHASE_LABEL_UP, PHASE_LABEL_DOWN, PHASE_LABEL_MONO, PHASE_LABEL_TETRA]
    ph_colors= [COLOR_UP, COLOR_DOWN, COLOR_MONO, COLOR_TETRA]
    for i,name in enumerate(ph_names):
        rr= radius_dict[i]
        axp= axs_ph[i//2][i%2]
        bins_r= np.linspace(0, global_max, 30)
        if len(rr)>0:
            axp.hist(rr, bins=bins_r, color=ph_colors[i], edgecolor='k', alpha=0.8)
            mean_rp= np.mean(rr)
            axp.axvline(mean_rp, color='r', linestyle='--', label=f"Mean={mean_rp:.2f}Å")
            axp.set_xlim(0, global_max*1.05)
        axp.set_title(name, fontsize=PDF_LABEL_FONT_SIZE)
        axp.set_xlabel("Radius(Å)", fontsize=PDF_TICK_FONT_SIZE)
        axp.set_ylabel("Count", fontsize=PDF_TICK_FONT_SIZE)
        axp.legend(fontsize=PDF_LEGEND_FONT_SIZE)

    plt.tight_layout()
    return fig_heat, fig_all, fig_phases

################################################################################
# [F] **새로운** BFS+Voronoi (혼합) - 큰 결정립 일부분할
################################################################################
def random_kcenters_subdivision(coords, n_sub):
    chosen_centers = [random.choice(coords)]
    while len(chosen_centers)< n_sub:
        dist_array= []
        for p in coords:
            dmin= min((p[0]-c[0])**2+(p[1]-c[1])**2 for c in chosen_centers)
            dist_array.append(dmin)
        idx= np.argmax(dist_array)
        chosen_centers.append(coords[idx])

    sub_assignment= {}
    for i in range(n_sub):
        sub_assignment[i]= []
    for p in coords:
        dd= [ (p[0]-c[0])**2+(p[1]-c[1])**2 for c in chosen_centers]
        sub_idx= np.argmin(dd)
        sub_assignment[sub_idx].append(p)
    return sub_assignment

def custom_voronoi_labeling_mixed_bfs(spin_arr,
                                      scale_factor=50.0,
                                      large_threshold=200,
                                      n_subgrains=3,
                                      seed=0):
    random.seed(seed)
    label_bfs, regions= label_connected_components(spin_arr)

    sub_centers= []
    sub_ids= []
    spin_map= {}
    id_counter= 0

    for lbid, info in regions.items():
        s= info["spin"]
        pix= info["pixels"]
        pcount= len(pix)

        if pcount>= large_threshold and n_subgrains>1:
            arr_pix= list(pix)
            sub_dict= random_kcenters_subdivision(arr_pix, n_subgrains)
            for sub_i, sub_coords in sub_dict.items():
                if not sub_coords:
                    continue
                id_counter+=1
                sub_id= id_counter
                spin_map[sub_id] = s

                scount= len(sub_coords)
                n_sub= max(1, int(round(scount/ scale_factor)))
                if n_sub> scount:
                    n_sub= scount
                chosen= random.sample(sub_coords, n_sub)
                for (xx,yy) in chosen:
                    sub_centers.append((xx+0.5, yy+0.5))
                    sub_ids.append(sub_id)
        else:
            id_counter+=1
            sub_id= id_counter
            spin_map[sub_id] = s

            scount= len(pix)
            n_sub= max(1, int(round(scount/scale_factor)))
            if n_sub> scount:
                n_sub= scount
            chosen= random.sample(pix, n_sub)
            for (xx,yy) in chosen:
                sub_centers.append((xx+0.5, yy+0.5))
                sub_ids.append(sub_id)

    if not sub_centers:
        final_label= np.ones_like(spin_arr, dtype=int)
        return final_label, {}

    sub_centers_arr = np.array(sub_centers)
    Nx, Ny = spin_arr.shape
    tree = cKDTree(sub_centers_arr)
    coords = np.indices((Nx, Ny)).reshape(2, -1).T
    dists, inds = tree.query(coords)
    final_label = np.array([sub_ids[i] for i in inds]).reshape(Nx, Ny)
    return final_label, spin_map

def plot_mixed_bfs_overlay(final_label,
                           spin_map,
                           step_label="",
                           scale_factor=50.0):
    Nx, Ny = final_label.shape
    fig, ax = plt.subplots(figsize=PDF_FIG_SIZE)
    ax.imshow(np.ones((Nx, Ny, 3)), origin='upper', vmin=0, vmax=1, aspect='equal')
    if step_label:
        ax.set_title(f"Step={step_label} MixedBFS Voronoi", fontsize=PDF_LABEL_FONT_SIZE)
    else:
        ax.set_title(f"MixedBFS Voronoi", fontsize=PDF_LABEL_FONT_SIZE)
    ax.axis('off')

    color_map= {
        0: COLOR_UP,
        1: COLOR_DOWN,
        2: COLOR_MONO,
        3: COLOR_TETRA
    }

    overlay_rgb = np.zeros((Nx, Ny, 3), dtype=np.float32)
    unique_ids = np.unique(final_label)
    for x in range(Nx):
        for y in range(Ny):
            sub_id= final_label[x,y]
            spin_s= spin_map.get(sub_id,2)
            cval= color_map.get(spin_s, (1,1,1,1))
            overlay_rgb[x,y]= cval[:3]

    ax.imshow(overlay_rgb, origin='upper', aspect='equal')

    for sid in unique_ids:
        mask= (final_label==sid)
        cts= find_contours(mask.astype(float), 0.5)
        for contour in cts:
            poly= Polygon(contour[:, ::-1], closed=True, fill=False,
                          edgecolor='white', linewidth=0.8)
            ax.add_patch(poly)

    custom_leg= [
        Patch(facecolor=COLOR_UP[:3],    edgecolor='black', label=PHASE_LABEL_UP),
        Patch(facecolor=COLOR_DOWN[:3],  edgecolor='black', label=PHASE_LABEL_DOWN),
        Patch(facecolor=COLOR_MONO[:3],  edgecolor='black', label=PHASE_LABEL_MONO),
        Patch(facecolor=COLOR_TETRA[:3], edgecolor='black', label=PHASE_LABEL_TETRA),
    ]
    ax.legend(handles=custom_leg, loc='upper left', fontsize=PDF_LEGEND_FONT_SIZE)
    return fig

def analyze_mixed_bfs_map(final_label,
                          spin_map,
                          step_label="",
                          scale_factor=50.0):
    unique_ids= np.unique(final_label)
    radius_dict= {0:[],1:[],2:[],3:[]}
    radius_all= []

    for sid in unique_ids:
        pcount= np.count_nonzero(final_label==sid)
        spin_s= spin_map.get(sid, 2)
        d= compute_equiv_diameter(pcount)
        r= d/2.0
        radius_dict[spin_s].append(r)
        radius_all.append(r)

    if not radius_all:
        fig_dummy= plt.figure()
        return fig_dummy, fig_dummy, fig_dummy

    global_max= max(radius_all)

    fig_all, ax_all= plt.subplots(figsize=PDF_FIG_SIZE)
    bins= np.linspace(0, global_max,30)
    ax_all.hist(radius_all, bins=bins, color='gray', edgecolor='k')
    mean_r= np.mean(radius_all)
    ax_all.axvline(mean_r, color='r', linestyle='--', label=f"Mean={mean_r:.2f}Å")
    ax_all.set_xlim(0, global_max*1.05)
    if step_label:
        ax_all.set_title(f"Step={step_label} MixedBFS(All)", fontsize=PDF_LABEL_FONT_SIZE)
    else:
        ax_all.set_title("MixedBFS Grain Size(All)", fontsize=PDF_LABEL_FONT_SIZE)
    ax_all.set_xlabel("Radius(Å)", fontsize=PDF_LABEL_FONT_SIZE)
    ax_all.set_ylabel("Count", fontsize=PDF_LABEL_FONT_SIZE)
    ax_all.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
    ax_all.legend(fontsize=PDF_LEGEND_FONT_SIZE)

    fig_phases, axs_ph= plt.subplots(2,2, figsize=(2*PDF_FIG_SIZE[0], 2*PDF_FIG_SIZE[1]))

    ph_names= [PHASE_LABEL_UP, PHASE_LABEL_DOWN, PHASE_LABEL_MONO, PHASE_LABEL_TETRA]
    ph_colors= [COLOR_UP, COLOR_DOWN, COLOR_MONO, COLOR_TETRA]
    for i,name in enumerate(ph_names):
        rr= radius_dict[i]
        axp= axs_ph[i//2][i%2]
        bins_r= np.linspace(0, global_max,30)
        if len(rr)>0:
            axp.hist(rr, bins=bins_r, color=ph_colors[i], edgecolor='k', alpha=0.8)
            mean_rp= np.mean(rr)
            axp.axvline(mean_rp, color='r', linestyle='--', label=f"Mean={mean_rp:.2f}Å")
            axp.set_xlim(0, global_max*1.05)
        axp.set_title(name, fontsize=PDF_LABEL_FONT_SIZE)
        axp.set_xlabel("Radius(Å)", fontsize=PDF_TICK_FONT_SIZE)
        axp.set_ylabel("Count", fontsize=PDF_TICK_FONT_SIZE)
        axp.legend(fontsize=PDF_LEGEND_FONT_SIZE)

    plt.tight_layout()
    fig_dummy= plt.figure()
    plt.close(fig_dummy)

    return fig_all, fig_phases, fig_dummy

################################################################################
# [G] Temperature Profile 모드 (실시간 + 오프라인)
################################################################################
def parse_temperature_profile():
    print("\n온도 프로파일을 입력하세요.")
    print("(예) 300 1000 30 => 300K에서 1000K까지 30스텝 선형증가")
    print("구간 여러 줄 입력 후, 빈 줄로 종료. 예:\n 300 1000 30\n 1000 500 10\n (빈 줄)\n")

    lines=[]
    while True:
        line= input()
        if not line.strip():
            break
        lines.append(line.strip())

    segs=[]
    for ln in lines:
        arr= ln.split()
        if len(arr)!=3:
            print("입력 오류. 예) 300 1000 30")
            sys.exit()
        sT= float(arr[0])
        eT= float(arr[1])
        sc= int(arr[2])
        segs.append((sT,eT,sc))

    if not segs:
        print("입력 없음. 종료.")
        sys.exit()

    return segs

def T_func_profile(segs, stp):
    cum=0
    for (startT,endT,stepc) in segs:
        if stp< cum+ stepc:
            local_s= stp- cum
            dT= (endT - startT)/stepc
            return max(startT + local_s*dT, 0.0)
        cum+= stepc
    return max(segs[-1][1], 0.0)


def parse_temperature_profile_en():
    print("\nEnter temperature profile segments.")
    print("Example: 300 1000 30 => linear increase from 300K to 1000K over 30 steps")
    print("Enter multiple lines, finish with an empty line.\n")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line.strip())
    segs = []
    for ln in lines:
        arr = ln.split()
        if len(arr) != 3:
            print("Input error. Example: 300 1000 30")
            sys.exit()
        sT = float(arr[0])
        eT = float(arr[1])
        sc = int(arr[2])
        segs.append((sT, eT, sc))
    if not segs:
        print("No input. Exiting.")
        sys.exit()
    return segs

def save_barrier_values(T,J,filename='barrier_values.csv'):
    with open(filename,'a',newline='') as f:
        w= csv.writer(f)
        def s_to_i(ss):
            if ss=='Up':      return 0
            elif ss=='Down':  return 1
            elif ss=='Mono':  return 2
            elif ss=='Tetra': return 3
            else: return -1
        for (old_str,new_str) in transition_types:
            old_code= s_to_i(old_str)
            new_code= s_to_i(new_str)
            barrier= get_transition_barrier(old_code,new_code,T,J)
            w.writerow([T, old_str, new_str, barrier])

def get_minmax_temperature_from_segments(segments):
    temps = []
    for (sT, eT, sc) in segments:
        temps.append(sT)
        temps.append(eT)
    return (min(temps), max(temps))

################################################################################
# [H] 오프라인 시뮬레이션 + PDF/GIF
################################################################################
def compute_bfs_radius_all(spin_arr):
    """
    BFS를 통해 결정립 찾아서 등가지름 계산 후,
    phase별/total 평균 반지름(Å)을 구합니다.
    """
    label_arr, regions = label_connected_components(spin_arr)
    radius_dict = {0:[], 1:[], 2:[], 3:[]}
    for _, info in regions.items():
        phase_s = info["spin"]
        pixcount= len(info["pixels"])
        d_angs  = compute_equiv_diameter(pixcount)
        r_angs  = d_angs/2.0
        radius_dict[phase_s].append(r_angs)

    all_vals = []
    for sidx in [0,1,2,3]:
        all_vals.extend(radius_dict[sidx])

    def safe_mean(lst):
        if len(lst)==0:
            return 0.0
        return float(np.mean(lst))

    bfs_up    = safe_mean(radius_dict[0])
    bfs_down  = safe_mean(radius_dict[1])
    bfs_mono  = safe_mean(radius_dict[2])
    bfs_tetra = safe_mean(radius_dict[3])
    bfs_total = safe_mean(all_vals) if len(all_vals)>0 else 0.0

    return bfs_up, bfs_down, bfs_mono, bfs_tetra, bfs_total

def run_profile_simulation_offline(segments,
                                   gif_filename="Temperature_animation.gif",
                                   pdf_filename="Temperature_result.pdf",
                                   global_minT=None,
                                   global_maxT=None):
    total_steps= sum(sc for (sT,eT,sc) in segments)
    seg_ranges=[]
    cum=0
    for i,(sT,eT,sc) in enumerate(segments):
        seg_ranges.append((i,sT,eT,cum,cum+sc))
        cum+= sc

    seg_index_for_step= [None]*(total_steps+1)
    for (idx, stT,enT, stS, stE) in seg_ranges:
        for stp in range(stS, stE):
            if stp<= total_steps:
                seg_index_for_step[stp]= idx

    spins_off= np.random.randint(q,size=(N_for_analysis,N_for_analysis))

    step_list=[]
    time_list=[]
    en_list=[]
    pol_list=[]
    up_list=[]
    down_list=[]
    mono_list=[]
    tetra_list=[]
    temp_list=[]
    chi_list=[]
    spin_snapshots=[]

    bfs_up_list=[]
    bfs_down_list=[]
    bfs_mono_list=[]
    bfs_tetra_list=[]
    bfs_total_list=[]

    current_time=0.0
    n_subsample=10

    initialize_barrier_file()

    if global_minT is None or global_maxT is None:
        global_minT, global_maxT = get_minmax_temperature_from_segments(segments)
    t_ax_min = max(0, global_minT - 50)
    t_ax_max = global_maxT + 50

    for step in range(total_steps+1):
        T_now= T_func_profile(segments, step)
        T_now= max(T_now, 0.0)

        # KMC 이벤트별 시간 합산
        accepted_time_sum=0.0
        for _ in range(N_for_analysis*N_for_analysis):
            x= np.random.randint(N_for_analysis)
            y= np.random.randint(N_for_analysis)
            accepted, used_barrier = attempt_flip(spins_off, x, y, T_now, J)
            if accepted:
                rate= nu0*np.exp(-used_barrier/(k_B*T_now)) if T_now>1e-9 else 0.0
                if rate>1e-30:
                    dt_event= 1.0 / rate
                    accepted_time_sum += dt_event

        dt= accepted_time_sum
        current_time += dt * TIME_SCALE

        # barrier 값 저장
        save_barrier_values(T_now,J)

        E_now = total_energy(spins_off, T_now, J)
        P_now= polarization(spins_off)
        ph_now= phase_fractions(spins_off)

        if step<len(seg_index_for_step):
            seg_idx= seg_index_for_step[step]
        else:
            seg_idx= None

        if seg_idx is not None:
            (sID, stT,enT, stS, stE)= seg_ranges[seg_idx]
            if abs(stT-enT)<1e-9:
                chi_now=0.0
            else:
                if step==0:
                    chi_now=0.0
                else:
                    chi_now= compute_fluctuation_susceptibility(spins_off, T_now, J, n_subsample)
        else:
            chi_now=0.0

        bfs_up, bfs_down, bfs_mono, bfs_tetra, bfs_all = compute_bfs_radius_all(spins_off)

        step_list.append(step)
        time_list.append(current_time)
        en_list.append(E_now)
        pol_list.append(P_now)
        up_list.append(ph_now[0])
        down_list.append(ph_now[1])
        mono_list.append(ph_now[2])
        tetra_list.append(ph_now[3])
        temp_list.append(T_now)
        chi_list.append(chi_now)

        bfs_up_list.append(bfs_up)
        bfs_down_list.append(bfs_down)
        bfs_mono_list.append(bfs_mono)
        bfs_tetra_list.append(bfs_tetra)
        bfs_total_list.append(bfs_all)

        spin_snapshots.append(np.copy(spins_off))

        logging.info(
            f"[OfflineSim] step={step}, seg={seg_idx}, T={T_now:.1f}, dt={dt:.3e}, time={current_time:.3e}s, "
            f"E={E_now:.3e}, P={P_now:.3e}, up={ph_now[0]:.3f}, down={ph_now[1]:.3f}, mono={ph_now[2]:.3f}, tetra={ph_now[3]:.3f}, "
            f"chi={chi_now:.2e}, BFSup={bfs_up:.2f}, BFSdown={bfs_down:.2f}, BFSmono={bfs_mono:.2f}, BFStet={bfs_tetra:.2f}, BFSall={bfs_all:.2f}"
        )

    # 애니메이션 (GIF)
    fig_off, axs_off= plt.subplots(2,2, figsize=(8,6), constrained_layout=True)
    ax_spin= axs_off[0][0]
    ax_en=   axs_off[0][1]
    ax_pol=  axs_off[1][0]
    ax_ph=   axs_off[1][1]

    im_off= ax_spin.imshow(spin_snapshots[0], cmap=BFS_CMAP, vmin=0, vmax=q-1, aspect='equal')
    ax_spin.axis('off')
    legend_e= [
        Patch(facecolor=COLOR_TETRA, label=PHASE_LABEL_TETRA),
        Patch(facecolor=COLOR_MONO,  label=PHASE_LABEL_MONO),
        Patch(facecolor=COLOR_UP,    label=PHASE_LABEL_UP),
        Patch(facecolor=COLOR_DOWN,  label=PHASE_LABEL_DOWN),
    ]
    ax_spin.legend(handles=legend_e, loc='upper left', fontsize=12)

    line_en_off,= ax_en.plot([],[],'r-', lw=2)
    ax_en.set_xlabel("Step", fontsize=LABEL_FONT_SIZE)
    ax_en.set_ylabel("Energy (J)", fontsize=LABEL_FONT_SIZE)
    ax_en.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    ax_en.grid(True, linestyle=':')

    line_pol_off,= ax_pol.plot([],[],'b-', lw=2)
    ax_pol.set_xlabel("Step", fontsize=LABEL_FONT_SIZE)
    ax_pol.set_ylabel("Polarization", fontsize=LABEL_FONT_SIZE)
    ax_pol.set_ylim(-1,1)
    ax_pol.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    ax_pol.grid(True, linestyle=':')

    line_up_off,=   ax_ph.plot([],[],'-', color=COLOR_UP,    lw=2)
    line_down_off,= ax_ph.plot([],[],'-', color=COLOR_DOWN,  lw=2)
    line_mono_off,= ax_ph.plot([],[],'-', color=COLOR_MONO,  lw=2)
    line_tetra_off,= ax_ph.plot([],[],'-', color=COLOR_TETRA, lw=2)
    ax_ph.set_ylim(-0.05,1.05)
    ax_ph.set_xlabel("Step", fontsize=LABEL_FONT_SIZE)
    ax_ph.set_ylabel("Phase Fraction", fontsize=LABEL_FONT_SIZE)
    ax_ph.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    ax_ph.grid(True, linestyle=':')

    ax_temp_off= ax_ph.twinx()
    line_temp_off,= ax_temp_off.plot([],[],'--', color=TEMP_COLOR, lw=2)
    ax_temp_off.set_ylabel("Temperature (K)", fontsize=LABEL_FONT_SIZE, color=TEMP_COLOR)
    ax_temp_off.tick_params(axis='y', labelsize=TICK_FONT_SIZE, labelcolor=TEMP_COLOR)
    ax_temp_off.set_ylim(t_ax_min, t_ax_max)

    def init_off():
        line_en_off.set_data([],[])
        line_pol_off.set_data([],[])
        line_up_off.set_data([],[])
        line_down_off.set_data([],[])
        line_mono_off.set_data([],[])
        line_tetra_off.set_data([],[])
        line_temp_off.set_data([],[])
        return (im_off, line_en_off, line_pol_off,
                line_up_off, line_down_off, line_mono_off, line_tetra_off, line_temp_off)

    def update_off(f):
        if f< len(step_list):
            im_off.set_data(spin_snapshots[f])
            xx= step_list[:f+1]
            yyE= en_list[:f+1]
            yyP= pol_list[:f+1]
            upA= up_list[:f+1]
            dwA= down_list[:f+1]
            moA= mono_list[:f+1]
            teA= tetra_list[:f+1]
            ttT= temp_list[:f+1]

            line_en_off.set_data(xx, yyE)
            ax_en.set_xlim(0, step_list[-1])
            if yyE:
                ax_en.set_ylim(0, max(yyE)*1.1)

            line_pol_off.set_data(xx, yyP)
            ax_pol.set_xlim(0, step_list[-1])

            line_up_off.set_data(xx, upA)
            line_down_off.set_data(xx, dwA)
            line_mono_off.set_data(xx, moA)
            line_tetra_off.set_data(xx, teA)
            ax_ph.set_xlim(0, step_list[-1])

            line_temp_off.set_data(xx, ttT)
            ax_temp_off.set_xlim(0, step_list[-1])
        return (im_off, line_en_off, line_pol_off,
                line_up_off, line_down_off, line_mono_off, line_tetra_off, line_temp_off)

    offline_ani= FuncAnimation(fig_off, update_off, init_func=init_off,
                               frames=len(step_list), interval=300, blit=False, repeat=False)
    try:
        offline_ani.save(gif_filename, writer=PillowWriter(fps=5), dpi=100)
        print(f"{gif_filename} saved (offline).")
        logging.info(f"[OfflineSim] {gif_filename} saved.")
    except Exception as e:
        print("GIF save error:", e)

    # === PDF 저장(모두 한 파일) + 각 그림별로 따로 PDF 저장 ===
    # 폴더 생성
    split_folder = "split_figures"
    if not os.path.exists(split_folder):
        os.makedirs(split_folder)

    try:
        with PdfPages(pdf_filename) as pdf:
            ################################################################
            # 1) Energy vs Step
            ################################################################
            figE= plt.figure(figsize=PDF_FIG_SIZE)
            plt.plot(step_list, en_list, 'r-', lw=2)
            plt.xlabel("Step", fontsize=PDF_LABEL_FONT_SIZE)
            plt.ylabel("Energy (J)", fontsize=PDF_LABEL_FONT_SIZE)
            plt.grid(True, linestyle=':')
            plt.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
            if en_list:
                plt.ylim(0, max(en_list)*1.1)
            pdf.savefig(figE,bbox_inches='tight')

            # 개별 PDF로도 저장
            figE.savefig(os.path.join(split_folder, "figure_1_Energy_vs_Step.pdf"), bbox_inches='tight')
            plt.close(figE)

            ################################################################
            # 2) Polarization vs Step
            ################################################################
            figP= plt.figure(figsize=PDF_FIG_SIZE)
            plt.plot(step_list, pol_list, 'b-', lw=2)
            plt.xlabel("Step", fontsize=PDF_LABEL_FONT_SIZE)
            plt.ylabel("Polarization", fontsize=PDF_LABEL_FONT_SIZE)
            plt.ylim(-1,1)
            plt.grid(True, linestyle=':')
            plt.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
            pdf.savefig(figP,bbox_inches='tight')

            figP.savefig(os.path.join(split_folder, "figure_2_Polarization_vs_Step.pdf"), bbox_inches='tight')
            plt.close(figP)

            ################################################################
            # 3) Phase fraction vs Step (+ Temperature)
            ################################################################
            figPh= plt.figure(figsize=PDF_FIG_SIZE)
            ax_phs= figPh.add_subplot(111)
            ax_phs.plot(step_list, up_list,   '-', color=COLOR_UP,    lw=2, label=PHASE_LABEL_UP)
            ax_phs.plot(step_list, down_list, '-', color=COLOR_DOWN,  lw=2, label=PHASE_LABEL_DOWN)
            ax_phs.plot(step_list, mono_list, '-', color=COLOR_MONO,  lw=2, label=PHASE_LABEL_MONO)
            ax_phs.plot(step_list, tetra_list,'-', color=COLOR_TETRA, lw=2, label=PHASE_LABEL_TETRA)
            ax_phs.set_xlabel("Step", fontsize=PDF_LABEL_FONT_SIZE)
            ax_phs.set_ylabel("Phase Fraction", fontsize=PDF_LABEL_FONT_SIZE)
            ax_phs.set_ylim(-0.05,1.05)
            ax_phs.grid(True, linestyle=':')
            ax_phs.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)

            ax_temp_s= ax_phs.twinx()
            ax_temp_s.plot(step_list, temp_list, '--', color=TEMP_COLOR, lw=2)
            ax_temp_s.set_ylabel("Temperature (K)", fontsize=PDF_LABEL_FONT_SIZE, color=TEMP_COLOR)
            ax_temp_s.tick_params(axis='y', labelsize=PDF_TICK_FONT_SIZE, labelcolor=TEMP_COLOR)
            ax_temp_s.set_ylim(t_ax_min, t_ax_max)

            ax_phs.legend(loc='upper left', fontsize=PDF_LEGEND_FONT_SIZE)
            pdf.savefig(figPh,bbox_inches='tight')

            figPh.savefig(os.path.join(split_folder, "figure_3_PhaseFrac_vs_Step.pdf"), bbox_inches='tight')
            plt.close(figPh)

            ################################################################
            # 4) Time-based (Energy vs time)
            ################################################################
            figEt= plt.figure(figsize=PDF_FIG_SIZE)
            plt.plot(time_list, en_list,'r-', lw=2)
            plt.xlabel("Time (s)", fontsize=PDF_LABEL_FONT_SIZE)
            plt.ylabel("Energy (J)", fontsize=PDF_LABEL_FONT_SIZE)
            plt.grid(True, linestyle=':')
            plt.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
            if en_list:
                plt.ylim(0, max(en_list)*1.1)
            pdf.savefig(figEt,bbox_inches='tight')

            figEt.savefig(os.path.join(split_folder, "figure_4_Energy_vs_Time.pdf"), bbox_inches='tight')
            plt.close(figEt)

            ################################################################
            # 5) Time-based (Polarization vs time)
            ################################################################
            figPt= plt.figure(figsize=PDF_FIG_SIZE)
            plt.plot(time_list, pol_list,'b-', lw=2)
            plt.xlabel("Time (s)", fontsize=PDF_LABEL_FONT_SIZE)
            plt.ylabel("Polarization", fontsize=PDF_LABEL_FONT_SIZE)
            plt.ylim(-1,1)
            plt.grid(True, linestyle=':')
            plt.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
            pdf.savefig(figPt,bbox_inches='tight')

            figPt.savefig(os.path.join(split_folder, "figure_5_Polarization_vs_Time.pdf"), bbox_inches='tight')
            plt.close(figPt)

            ################################################################
            # 6) Time-based (Phase fraction vs time)
            ################################################################
            figPh_t= plt.figure(figsize=PDF_FIG_SIZE)
            ax_ph_t= figPh_t.add_subplot(111)
            ax_ph_t.plot(time_list, up_list,    '-', color=COLOR_UP,    lw=2, label=PHASE_LABEL_UP)
            ax_ph_t.plot(time_list, down_list,  '-', color=COLOR_DOWN,  lw=2, label=PHASE_LABEL_DOWN)
            ax_ph_t.plot(time_list, mono_list,  '-', color=COLOR_MONO,  lw=2, label=PHASE_LABEL_MONO)
            ax_ph_t.plot(time_list, tetra_list, '-', color=COLOR_TETRA, lw=2, label=PHASE_LABEL_TETRA)
            ax_ph_t.set_xlabel("Time (s)", fontsize=PDF_LABEL_FONT_SIZE)
            ax_ph_t.set_ylabel("Phase Fraction", fontsize=PDF_LABEL_FONT_SIZE)
            ax_ph_t.set_ylim(-0.05,1.05)
            ax_ph_t.grid(True, linestyle=':')
            ax_ph_t.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)

            ax_temp_t= ax_ph_t.twinx()
            ax_temp_t.plot(time_list, temp_list,'--', color=TEMP_COLOR, lw=2)
            ax_temp_t.set_ylabel("Temperature (K)", fontsize=PDF_LABEL_FONT_SIZE, color=TEMP_COLOR)
            ax_temp_t.tick_params(axis='y', labelcolor=TEMP_COLOR, labelsize=PDF_TICK_FONT_SIZE)
            ax_temp_t.set_ylim(t_ax_min, t_ax_max)

            ax_ph_t.legend(loc='upper left', fontsize=PDF_LEGEND_FONT_SIZE)
            pdf.savefig(figPh_t,bbox_inches='tight')

            figPh_t.savefig(os.path.join(split_folder, "figure_6_PhaseFrac_vs_Time.pdf"), bbox_inches='tight')
            plt.close(figPh_t)

            # === 최종 스텝 ===
            final_snap= spin_snapshots[-1]
            final_step= total_steps

            ################################################################
            # (1) Raw Heatmap
            ################################################################
            fig_raw_fin= plt.figure(figsize=PDF_FIG_SIZE)
            plt.imshow(final_snap, cmap=BFS_CMAP, vmin=0, vmax=q-1, aspect='equal')
            plt.title(f"Final Step={final_step} Raw Heatmap", fontsize=PDF_LABEL_FONT_SIZE)
            plt.axis('off')
            pdf.savefig(fig_raw_fin, bbox_inches='tight')
            fig_raw_fin.savefig(os.path.join(split_folder, "figure_7_Final_Heatmap.pdf"), bbox_inches='tight')
            plt.close(fig_raw_fin)

            ################################################################
            # (2) BFS 분석
            ################################################################
            fig_bfs_heat, fig_bfs_all, fig_bfs_ph= analyze_grains_with_markers(final_snap, step_label=str(final_step))
            pdf.savefig(fig_bfs_heat, bbox_inches='tight')
            pdf.savefig(fig_bfs_all, bbox_inches='tight')
            pdf.savefig(fig_bfs_ph, bbox_inches='tight')

            fig_bfs_heat.savefig(os.path.join(split_folder, "figure_8_BFS_Heatmap.pdf"), bbox_inches='tight')
            fig_bfs_all.savefig(os.path.join(split_folder, "figure_9_BFS_GrainSizeAll.pdf"), bbox_inches='tight')
            fig_bfs_ph.savefig(os.path.join(split_folder, "figure_10_BFS_GrainSizePhases.pdf"), bbox_inches='tight')

            plt.close(fig_bfs_heat)
            plt.close(fig_bfs_all)
            plt.close(fig_bfs_ph)

            ################################################################
            # (3) 혼합 BFS-Voronoi
            ################################################################
            mixed_label, spin_map= custom_voronoi_labeling_mixed_bfs(
                final_snap,
                scale_factor=50.0,
                large_threshold=200,
                n_subgrains=3,
                seed=0
            )
            fig_mixed= plot_mixed_bfs_overlay(mixed_label, spin_map,
                                              step_label=str(final_step),
                                              scale_factor=50.0)
            pdf.savefig(fig_mixed, bbox_inches='tight')
            fig_mixed.savefig(os.path.join(split_folder, "figure_11_MixedBFS_Overlay.pdf"), bbox_inches='tight')
            plt.close(fig_mixed)

            fig_mixed_all, fig_mixed_ph, _= analyze_mixed_bfs_map(
                mixed_label, spin_map,
                step_label=str(final_step),
                scale_factor=50.0
            )
            pdf.savefig(fig_mixed_all, bbox_inches='tight')
            pdf.savefig(fig_mixed_ph, bbox_inches='tight')

            fig_mixed_all.savefig(os.path.join(split_folder, "figure_12_MixedBFS_GrainSizeAll.pdf"), bbox_inches='tight')
            fig_mixed_ph.savefig(os.path.join(split_folder, "figure_13_MixedBFS_GrainSizePhases.pdf"), bbox_inches='tight')

            plt.close(fig_mixed_all)
            plt.close(fig_mixed_ph)

            ################################################################
            # (4) BFS Grain size vs Step
            ################################################################
            fig_bfs, ax_bfs = plt.subplots(figsize=PDF_FIG_SIZE)
            ax_bfs.plot(step_list, bfs_up_list,    color=COLOR_UP,    lw=1.5, label="Up")
            ax_bfs.plot(step_list, bfs_down_list,  color=COLOR_DOWN,  lw=1.5, label="Down")
            ax_bfs.plot(step_list, bfs_mono_list,  color=COLOR_MONO,  lw=1.5, label="m")
            ax_bfs.plot(step_list, bfs_tetra_list, color=COLOR_TETRA, lw=1.5, label="t")
            ax_bfs.plot(step_list, bfs_total_list, color='k',         lw=2.0, label="total")

            ax_bfs.set_xlabel("Step", fontsize=PDF_LABEL_FONT_SIZE)
            ax_bfs.set_ylabel("grain radius (Å)", fontsize=PDF_LABEL_FONT_SIZE)
            ax_bfs.grid(True, linestyle=':')
            ax_bfs.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)

            max_bfs_val = 0.0
            if bfs_total_list:
                max_bfs_val = max( bfs_up_list + bfs_down_list + bfs_mono_list + bfs_tetra_list + bfs_total_list )
            ax_bfs.set_ylim(0, max_bfs_val*1.1 if max_bfs_val>0 else 1.0)

            ax_bfsT = ax_bfs.twinx()
            ax_bfsT.plot(step_list, temp_list, '--', color=TEMP_COLOR, lw=2)
            ax_bfsT.set_ylabel("Temperature (K)", fontsize=PDF_LABEL_FONT_SIZE, color=TEMP_COLOR)
            ax_bfsT.tick_params(axis='y', labelsize=PDF_TICK_FONT_SIZE, labelcolor=TEMP_COLOR)
            ax_bfsT.set_ylim(t_ax_min, t_ax_max)

            ax_bfs.legend(loc='upper left', fontsize=PDF_LEGEND_FONT_SIZE)
            pdf.savefig(fig_bfs,bbox_inches='tight')
            fig_bfs.savefig(os.path.join(split_folder, "figure_14_BFS_Radius_vs_Step.pdf"), bbox_inches='tight')
            plt.close(fig_bfs)

            # (추가) 특정 step에서의 heatmap 등
            print("\n추출하고 싶은 Heatmap + Grain Step 번호를 띄어쓰기로 입력 (예: 30 60 90).")
            print("(아무것도 입력 안 하고 엔터 => 생략)\n")
            extra_steps_in= input("Steps: ").strip()
            if extra_steps_in:
                try:
                    step_candidates= [int(x) for x in extra_steps_in.split()]
                    for stv in step_candidates:
                        if 0<=stv< len(spin_snapshots):
                            snap_st= spin_snapshots[stv]
                            fig_raw_st= plt.figure(figsize=PDF_FIG_SIZE)
                            plt.imshow(snap_st, cmap=BFS_CMAP, vmin=0, vmax=q-1, aspect='equal')
                            plt.title(f"Step={stv} Raw Heatmap", fontsize=PDF_LABEL_FONT_SIZE)
                            plt.axis('off')
                            pdf.savefig(fig_raw_st, bbox_inches='tight')
                            fig_raw_st.savefig(os.path.join(split_folder, f"ExtraStep_{stv}_Heatmap.pdf"), bbox_inches='tight')
                            plt.close(fig_raw_st)

                            fig_bfsH_s, fig_bfsA_s, fig_bfsP_s= analyze_grains_with_markers(snap_st, step_label=str(stv))
                            pdf.savefig(fig_bfsH_s, bbox_inches='tight')
                            fig_bfsH_s.savefig(os.path.join(split_folder, f"ExtraStep_{stv}_BFS_Heatmap.pdf"), bbox_inches='tight')
                            plt.close(fig_bfsH_s)

                            pdf.savefig(fig_bfsA_s, bbox_inches='tight')
                            fig_bfsA_s.savefig(os.path.join(split_folder, f"ExtraStep_{stv}_BFS_GrainSizeAll.pdf"), bbox_inches='tight')
                            plt.close(fig_bfsA_s)

                            pdf.savefig(fig_bfsP_s, bbox_inches='tight')
                            fig_bfsP_s.savefig(os.path.join(split_folder, f"ExtraStep_{stv}_BFS_GrainSizePhases.pdf"), bbox_inches='tight')
                            plt.close(fig_bfsP_s)

                            lbl_mix, sp_map= custom_voronoi_labeling_mixed_bfs(
                                snap_st,
                                scale_factor=50.0,
                                large_threshold=200,
                                n_subgrains=3,
                                seed=0
                            )
                            fig_mix_st= plot_mixed_bfs_overlay(lbl_mix, sp_map,
                                                               step_label=str(stv),
                                                               scale_factor=50.0)
                            pdf.savefig(fig_mix_st, bbox_inches='tight')
                            fig_mix_st.savefig(os.path.join(split_folder, f"ExtraStep_{stv}_MixedBFS_Overlay.pdf"), bbox_inches='tight')
                            plt.close(fig_mix_st)

                            fig_mix_all, fig_mix_ph, _= analyze_mixed_bfs_map(
                                lbl_mix, sp_map,
                                step_label=str(stv),
                                scale_factor=50.0
                            )
                            pdf.savefig(fig_mix_all, bbox_inches='tight')
                            fig_mix_all.savefig(os.path.join(split_folder, f"ExtraStep_{stv}_MixedBFS_SizeAll.pdf"), bbox_inches='tight')
                            plt.close(fig_mix_all)

                            pdf.savefig(fig_mix_ph, bbox_inches='tight')
                            fig_mix_ph.savefig(os.path.join(split_folder, f"ExtraStep_{stv}_MixedBFS_SizePhases.pdf"), bbox_inches='tight')
                            plt.close(fig_mix_ph)
                        else:
                            print(f"Step={stv}는 범위를 벗어납니다 (0 ~ {len(spin_snapshots)-1}).")
                except:
                    pass

            print(f"{pdf_filename} saved (offline).")

    except Exception as e:
        print("PDF save error:", e)

################################################################################
# [I-2] 산포(Scatter) 모드 다중 반복 시뮬레이션
################################################################################

def simulate_single_run_record(Nx, Ny, segments, seed):
    """Run one scatter simulation with a fixed seed and record heatmaps."""
    np.random.seed(seed)
    spins = np.random.randint(q, size=(Nx, Ny))
    total_steps = sum(sc for (_, _, sc) in segments)
    snaps = []
    for step in range(total_steps + 1):
        T_now = T_func_profile(segments, step)
        T_now = max(T_now, 0.0)
        for _ in range(Nx * Ny):
            x = np.random.randint(Nx)
            y = np.random.randint(Ny)
            attempt_flip(spins, x, y, T_now, J)
        snaps.append(np.copy(spins))
    return snaps

def create_heatmap_gif_for_seeds(Nx, Ny, segments, seeds, filename, labels):
    """Create GIF showing heatmaps for given seeds.

    The lattice is transposed so that Nx corresponds to the horizontal axis.
    Figure size adapts to the rectangle aspect to fill each page correctly.
    """
    if not seeds:
        return
    run_snaps = [simulate_single_run_record(Nx, Ny, segments, sd) for sd in seeds]
    steps = len(run_snaps[0])
    # aspect ratio = height / width with Nx as horizontal length
    ar = Ny / float(Nx) if Nx > 0 else 1.0
    orientation_vertical = Nx >= Ny
    base = 4.0
    ratio = Nx / float(Ny) if Ny > 0 else 1.0
    if ratio >= 1:
        wfac = ratio
        hfac = 1.0
    else:
        wfac = 1.0
        hfac = 1.0 / ratio
    if orientation_vertical:
        rows = len(seeds)
        cols = 1
    else:
        cols = len(seeds)
        rows = 1
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * base * wfac, rows * base * hfac))
    axes = np.atleast_1d(axes)
    ims = []
    temp_texts = []
    for ax, snap, lbl in zip(axes, run_snaps, labels):
        im = ax.imshow(
            snap[0].T,
            cmap=BFS_CMAP,
            vmin=0,
            vmax=q - 1,
            animated=True,
            aspect='equal',
        )
        ax.axis('off')
        ax.set_title(lbl, fontsize=6)
        txt = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                      ha='left', va='top', color='white', fontsize=6,
                      bbox=dict(facecolor='black', alpha=0.5))
        ims.append(im)
        temp_texts.append(txt)

    def init():
        for im, snap, txt in zip(ims, run_snaps, temp_texts):
            im.set_data(snap[0].T)
            txt.set_text(f"T={T_func_profile(segments,0):.1f}K")
        return ims + temp_texts

    def update(frame):
        T_now = T_func_profile(segments, frame)
        for im, snap, ax, lbl, txt in zip(ims, run_snaps, axes, labels, temp_texts):
            if frame < len(snap):
                im.set_data(snap[frame].T)
            ax.set_title(lbl, fontsize=6)
            txt.set_text(f"T={T_now:.1f}K")
        return ims + temp_texts

    ani = FuncAnimation(fig, update, init_func=init, frames=steps, interval=300, blit=False, repeat=False)
    try:
        ani.save(filename, writer=PillowWriter(fps=5), dpi=100)
        print(f"{filename} saved.")
    except Exception as e:
        print("GIF save error:", e)
    plt.close(fig)

def create_heatmap_pdf_for_seeds(Nx, Ny, segments, seeds, labels, filename):
    """Create a PDF with final heatmaps for given seeds."""
    if not seeds:
        return
    run_snaps = [simulate_single_run_record(Nx, Ny, segments, sd) for sd in seeds]
    ar = Ny / float(Nx) if Nx > 0 else 1.0
    ratio = Nx / float(Ny) if Ny > 0 else 1.0
    if ratio >= 1:
        wfac = ratio
        hfac = 1.0
    else:
        wfac = 1.0
        hfac = 1.0 / ratio
    base = 4.0
    with PdfPages(filename) as pdf:
        for snap, lbl in zip(run_snaps, labels):
            fig, ax = plt.subplots(figsize=(base * wfac, base * hfac))
            ax.imshow(
                snap[-1].T,
                cmap=BFS_CMAP,
                vmin=0,
                vmax=q - 1,
                aspect='equal',
            )
            ax.axis("off")
            ax.set_title(lbl, fontsize=8)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
################################################################################
def run_scatter_mode(
    Nx,
    Ny,
    segments,
    repeat_count,
    dopant_choice=None,
    doping_base=0.0,
    doping_var=0.0,
    h_var=0.0,
):
    total_steps = sum(sc for (_, _, sc) in segments)
    O_list = []
    T_list = []
    M_list = []
    P_list = []
    snap_list = []
    seed_list = []
    h_list = []

    original_h = h
    for r in range(repeat_count):
        # Randomize thickness if variation requested
        if h_var > 0:
            sigma_h = h_var / 2.0
            h_rand = np.random.normal(original_h, sigma_h)
            h_rand = np.clip(h_rand, original_h - h_var, original_h + h_var)
            set_global_h(h_rand)
        else:
            set_global_h(original_h)
        h_list.append(h / 10.0)

        # Randomize dopant amount if requested
        doping_val_run = doping_base
        if doping_var > 0 and dopant_choice in ['1', '2', '3']:
            sigma_d = doping_var / 2.0
            doping_val_run = np.random.normal(doping_base, sigma_d)
            low = max(0.0, doping_base - doping_var)
            high = min(100.0, doping_base + doping_var)
            doping_val_run = float(np.clip(doping_val_run, low, high))
            if dopant_choice == '1':
                set_material_params_zr_doped(doping_val_run)
            elif dopant_choice == '2':
                set_material_params_si_doped(doping_val_run)
            elif dopant_choice == '3':
                set_material_params_al_doped(doping_val_run)

        seed = random.randint(0, 2**32 - 1)
        seed_list.append(seed)
        np.random.seed(seed)
        spins = np.random.randint(q, size=(Nx, Ny))
        for step in range(total_steps + 1):
            T_now = T_func_profile(segments, step)
            T_now = max(T_now, 0.0)
            for _ in range(Nx * Ny):
                x = np.random.randint(Nx)
                y = np.random.randint(Ny)
                attempt_flip(spins, x, y, T_now, J)

        ph = phase_fractions(spins)
        pol = polarization(spins)
        O_list.append(ph[0] + ph[1])
        T_list.append(ph[3])
        M_list.append(ph[2])
        P_list.append(pol)
        snap_list.append(np.copy(spins))

        logging.info(
            f"[Scatter] run={r}, O={ph[0]+ph[1]:.3f}, T={ph[3]:.3f}, M={ph[2]:.3f}, P={pol:.3f}"
        )

    # restore original global parameters
    set_global_h(original_h)
    if dopant_choice == '1':
        set_material_params_zr_doped(doping_base)
    elif dopant_choice == '2':
        set_material_params_si_doped(doping_base)
    elif dopant_choice == '3':
        set_material_params_al_doped(doping_base)

    os.makedirs("count", exist_ok=True)
    os.makedirs("polarization", exist_ok=True)
    os.makedirs("O_phase", exist_ok=True)

    bins = np.linspace(0, 1, 20)
    fig_O, ax_O = plt.subplots(figsize=PDF_FIG_SIZE)
    ax_O.hist(O_list, bins=bins, color='gray', edgecolor='k')
    if O_list:
        mean_O = np.mean(O_list)
        ax_O.axvline(mean_O, color='r', linestyle='--', label=f"Mean={mean_O:.2f}")
        ax_O.legend(fontsize=PDF_LEGEND_FONT_SIZE)
    ax_O.set_xlabel('O fraction', fontsize=PDF_LABEL_FONT_SIZE)
    ax_O.set_ylabel('Count', fontsize=PDF_LABEL_FONT_SIZE)
    ax_O.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
    ax_O.set_title('O Phase Histogram', fontsize=PDF_LABEL_FONT_SIZE)
    fig_O.savefig(os.path.join('count', 'O_histogram.pdf'), bbox_inches='tight')
    plt.close(fig_O)

    fig_T, ax_T = plt.subplots(figsize=PDF_FIG_SIZE)
    ax_T.hist(T_list, bins=bins, color='gray', edgecolor='k')
    if T_list:
        mean_T = np.mean(T_list)
        ax_T.axvline(mean_T, color='r', linestyle='--', label=f"Mean={mean_T:.2f}")
        ax_T.legend(fontsize=PDF_LEGEND_FONT_SIZE)
    ax_T.set_xlabel('T fraction', fontsize=PDF_LABEL_FONT_SIZE)
    ax_T.set_ylabel('Count', fontsize=PDF_LABEL_FONT_SIZE)
    ax_T.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
    ax_T.set_title('T Phase Histogram', fontsize=PDF_LABEL_FONT_SIZE)
    fig_T.savefig(os.path.join('count', 'T_histogram.pdf'), bbox_inches='tight')
    plt.close(fig_T)

    fig_M, ax_M = plt.subplots(figsize=PDF_FIG_SIZE)
    ax_M.hist(M_list, bins=bins, color='gray', edgecolor='k')
    if M_list:
        mean_M = np.mean(M_list)
        ax_M.axvline(mean_M, color='r', linestyle='--', label=f"Mean={mean_M:.2f}")
        ax_M.legend(fontsize=PDF_LEGEND_FONT_SIZE)
    ax_M.set_xlabel('M fraction', fontsize=PDF_LABEL_FONT_SIZE)
    ax_M.set_ylabel('Count', fontsize=PDF_LABEL_FONT_SIZE)
    ax_M.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
    ax_M.set_title('M Phase Histogram', fontsize=PDF_LABEL_FONT_SIZE)
    fig_M.savefig(os.path.join('count', 'M_histogram.pdf'), bbox_inches='tight')
    plt.close(fig_M)

    fig_P, ax_P = plt.subplots(figsize=PDF_FIG_SIZE)
    ax_P.hist(P_list, bins=20, color='gray', edgecolor='k')
    if P_list:
        mean_P = np.mean(P_list)
        ax_P.axvline(mean_P, color='r', linestyle='--', label=f"Mean={mean_P:.2f}")
        ax_P.legend(fontsize=PDF_LEGEND_FONT_SIZE)
    ax_P.set_xlabel('Polarization', fontsize=PDF_LABEL_FONT_SIZE)
    ax_P.set_ylabel('Count', fontsize=PDF_LABEL_FONT_SIZE)
    ax_P.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
    ax_P.set_title('Polarization Histogram', fontsize=PDF_LABEL_FONT_SIZE)
    fig_P.savefig(os.path.join('count', 'Polarization_histogram.pdf'), bbox_inches='tight')
    plt.close(fig_P)

    idx_sorted = np.argsort(P_list)
    n = len(P_list)
    n10 = max(1, n // 10)
    low_idx = idx_sorted[:n10]
    high_idx = idx_sorted[-n10:]
    mid_start = max(0, n // 2 - n10 // 2)
    mid_idx = idx_sorted[mid_start:mid_start + n10]

    idx_O_sorted = np.argsort(O_list)
    low_O = idx_O_sorted[:n10]
    high_O = idx_O_sorted[-n10:]
    mid_O_start = max(0, n // 2 - n10 // 2)
    mid_O = idx_O_sorted[mid_O_start:mid_O_start + n10]

    def plot_heatmaps(indices, title):
        """Return list of figures with up to two heatmaps per page."""
        if len(indices) == 0:
            return []

        figs = []
        ar = Ny / float(Nx) if Nx > 0 else 1.0
        orientation_vertical = Nx >= Ny
        base = 4.0
        ratio = Nx / float(Ny) if Ny > 0 else 1.0
        if ratio >= 1:
            wfac = ratio
            hfac = 1.0
        else:
            wfac = 1.0
            hfac = 1.0 / ratio
        per_page = 1

        for start in range(0, len(indices), per_page):
            subset = indices[start:start + per_page]
            rows = 1
            cols = 1

            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(cols * base * wfac, rows * base * hfac),
            )
            axes = np.atleast_1d(axes).ravel()

            for i, idx in enumerate(subset):
                ax = axes[i]
                ax.imshow(
                    snap_list[idx].T,
                    cmap=BFS_CMAP,
                    vmin=0,
                    vmax=q - 1,
                    aspect='equal',
                )
                ax.axis("off")
                ax.set_title(
                    f"run {idx} (h={h_list[idx]:.1f} nm) "
                    f"O={O_list[idx]*100:.1f}% M={M_list[idx]*100:.1f}% T={T_list[idx]*100:.1f}%",
                    fontsize=6,
                )
            for j in range(len(subset), len(axes)):
                axes[j].axis("off")
            figs.append(fig)

        return figs

    fig_low_list = plot_heatmaps(low_idx, 'Low Polarization (bottom 10%)')
    if fig_low_list:
        with PdfPages(os.path.join('polarization', 'low.pdf')) as pdf:
            for fg in fig_low_list:
                pdf.savefig(fg, bbox_inches='tight')
                plt.close(fg)

    fig_mid_list = plot_heatmaps(mid_idx, 'Middle 10% Polarization')
    if fig_mid_list:
        with PdfPages(os.path.join('polarization', 'mid.pdf')) as pdf:
            for fg in fig_mid_list:
                pdf.savefig(fg, bbox_inches='tight')
                plt.close(fg)

    fig_high_list = plot_heatmaps(high_idx, 'High Polarization (top 10%)')
    if fig_high_list:
        with PdfPages(os.path.join('polarization', 'high.pdf')) as pdf:
            for fg in fig_high_list:
                pdf.savefig(fg, bbox_inches='tight')
                plt.close(fg)

    fig_O_low_list = plot_heatmaps(low_O, 'Low O fraction (bottom 10%)')
    if fig_O_low_list:
        with PdfPages(os.path.join('O_phase', 'O_phase_min.pdf')) as pdf:
            for fg in fig_O_low_list:
                pdf.savefig(fg, bbox_inches='tight')
                plt.close(fg)

    fig_O_mid_list = plot_heatmaps(mid_O, 'Middle 10% O fraction')
    if fig_O_mid_list:
        with PdfPages(os.path.join('O_phase', 'O_phase_mid.pdf')) as pdf:
            for fg in fig_O_mid_list:
                pdf.savefig(fg, bbox_inches='tight')
                plt.close(fg)

    fig_O_high_list = plot_heatmaps(high_O, 'High O fraction (top 10%)')
    if fig_O_high_list:
        with PdfPages(os.path.join('O_phase', 'O_phase_max.pdf')) as pdf:
            for fg in fig_O_high_list:
                pdf.savefig(fg, bbox_inches='tight')
                plt.close(fg)

    # ---- Grain size histograms based on final states ----
    grain_O = []
    grain_T = []
    grain_M = []
    grain_all = []

    def gather_grains(spin_map):
        label_arr, regions = label_connected_components(spin_map)
        for info in regions.values():
            phase_s = info["spin"]
            pixcount = len(info["pixels"])
            r = compute_equiv_diameter(pixcount) / 2.0
            grain_all.append(r)
            if phase_s in (0, 1):
                grain_O.append(r)
            elif phase_s == 2:
                grain_M.append(r)
            elif phase_s == 3:
                grain_T.append(r)

    for snap in snap_list:
        gather_grains(snap)

    os.makedirs("grain", exist_ok=True)

    if grain_all:
        max_r = max(grain_all)
    else:
        max_r = 1.0
    bins_r = np.linspace(0, max_r, 30)

    def save_grain_hist(data, fname, title):
        fig, ax = plt.subplots(figsize=PDF_FIG_SIZE)
        if data:
            ax.hist(data, bins=bins_r, color='gray', edgecolor='k')
            mean_val = np.mean(data)
            ax.axvline(mean_val, color='r', linestyle='--', label=f"Mean={mean_val:.2f}Å")
            ax.set_xlim(0, max_r * 1.05)
            ax.legend(fontsize=PDF_LEGEND_FONT_SIZE)
        ax.set_xlabel('Radius(Å)', fontsize=PDF_LABEL_FONT_SIZE)
        ax.set_ylabel('Count', fontsize=PDF_LABEL_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
        ax.set_title(title, fontsize=PDF_LABEL_FONT_SIZE)
        fig.savefig(os.path.join('grain', fname), bbox_inches='tight')
        plt.close(fig)

    def save_grain_hist_stack(O_data, M_data, T_data, fname, title):
        fig, ax = plt.subplots(figsize=PDF_FIG_SIZE)
        if O_data or M_data or T_data:
            counts_O, _ = np.histogram(O_data, bins=bins_r)
            counts_M, _ = np.histogram(M_data, bins=bins_r)
            counts_T, _ = np.histogram(T_data, bins=bins_r)
            width = np.diff(bins_r)
            left = bins_r[:-1]
            ax.bar(left, counts_O, width=width, color=COLOR_UP, align='edge', label='O')
            ax.bar(left, counts_M, width=width, bottom=counts_O, color=COLOR_MONO, align='edge', label='M')
            bottom_TM = counts_O + counts_M
            ax.bar(left, counts_T, width=width, bottom=bottom_TM, color=COLOR_TETRA, align='edge', label='T')
            total_counts = counts_O + counts_M + counts_T
            if total_counts.any():
                mean_all = np.average((left + width/2), weights=total_counts)
                ax.axvline(mean_all, color='r', linestyle='--', label=f"Mean={mean_all:.2f}Å")
            ax.set_xlim(0, max_r * 1.05)
            ax.legend(fontsize=PDF_LEGEND_FONT_SIZE)
        ax.set_xlabel('Radius(Å)', fontsize=PDF_LABEL_FONT_SIZE)
        ax.set_ylabel('Count', fontsize=PDF_LABEL_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
        ax.set_title(title, fontsize=PDF_LABEL_FONT_SIZE)
        fig.savefig(os.path.join('grain', fname), bbox_inches='tight')
        plt.close(fig)

    save_grain_hist(grain_O, 'O_grain_histogram.pdf', 'O Phase Grain Size')
    save_grain_hist(grain_T, 'T_grain_histogram.pdf', 'T Phase Grain Size')
    save_grain_hist(grain_M, 'M_grain_histogram.pdf', 'M Phase Grain Size')
    save_grain_hist_stack(grain_O, grain_M, grain_T,
                          'All_grain_histogram.pdf',
                          'All Phases Grain Size')

    # ---- Thickness distribution histogram ----
    os.makedirs("count", exist_ok=True)
    fig_h, ax_h = plt.subplots(figsize=PDF_FIG_SIZE)
    if h_list:
        ax_h.hist(h_list, bins=20, color='gray', edgecolor='k')
        mean_h = np.mean(h_list)
        ax_h.axvline(mean_h, color='r', linestyle='--', label=f"Mean={mean_h:.2f} nm")
        ax_h.legend(fontsize=PDF_LEGEND_FONT_SIZE)
    ax_h.set_xlabel('h (nm)', fontsize=PDF_LABEL_FONT_SIZE)
    ax_h.set_ylabel('Count', fontsize=PDF_LABEL_FONT_SIZE)
    ax_h.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
    ax_h.set_title('Thickness Distribution', fontsize=PDF_LABEL_FONT_SIZE)
    fig_h.savefig(os.path.join('count', 'h_histogram.pdf'), bbox_inches='tight')
    plt.close(fig_h)
    with open(os.path.join('count', 'h_values.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'h_nm'])
        for i, hv in enumerate(h_list):
            w.writerow([i, hv])

    # --- create GIFs and PDF for runs grouped by O fraction extremes ---
    if O_list:
        order = np.argsort(O_list)
        n = len(O_list)
        n10 = max(1, n // 10)
        idx_high = order[-n10:]
        idx_low = order[:n10]
        mid_start = max(0, n // 2 - n10 // 2)
        idx_mid = order[mid_start:mid_start + n10]

        def make_labels(indices):
            return [
                f"run {i} (h={h_list[i]:.1f} nm) "
                f"O={O_list[i]*100:.1f}% M={M_list[i]*100:.1f}% T={T_list[i]*100:.1f}%"
                for i in indices
            ]

        # GIFs for each group
        create_heatmap_gif_for_seeds(
            Nx,
            Ny,
            segments,
            [seed_list[i] for i in idx_high],
            'O_phase_max.gif',
            make_labels(idx_high),
        )
        create_heatmap_gif_for_seeds(
            Nx,
            Ny,
            segments,
            [seed_list[i] for i in idx_mid],
            'O_phase_mid.gif',
            make_labels(idx_mid),
        )
        create_heatmap_gif_for_seeds(
            Nx,
            Ny,
            segments,
            [seed_list[i] for i in idx_low],
            'O_phase_min.gif',
            make_labels(idx_low),
        )

        # Combined PDF for the same groups
        create_heatmap_pdf_for_seeds(
            Nx,
            Ny,
            segments,
            [seed_list[i] for i in idx_high],
            make_labels(idx_high),
            os.path.join('O_phase', 'O_phase_max.pdf'),
        )
        create_heatmap_pdf_for_seeds(
            Nx,
            Ny,
            segments,
            [seed_list[i] for i in idx_mid],
            make_labels(idx_mid),
            os.path.join('O_phase', 'O_phase_mid.pdf'),
        )
        create_heatmap_pdf_for_seeds(
            Nx,
            Ny,
            segments,
            [seed_list[i] for i in idx_low],
            make_labels(idx_low),
            os.path.join('O_phase', 'O_phase_min.pdf'),
        )

def create_auto_scatter_script(
    Nx,
    Ny,
    segments,
    repeat_count,
    dopant_choice,
    doping_val,
    doping_var=0.0,
    h_var=0.0,
    filename="auto_scatter.py",
):
    """Generate a standalone script to rerun scatter mode with given params."""
    try:
        with open(filename, "w") as f:
            f.write("import mainrev1\n")
            if dopant_choice == '1':
                f.write(f"mainrev1.set_material_params_zr_doped({doping_val})\n")
            elif dopant_choice == '2':
                f.write(f"mainrev1.set_material_params_si_doped({doping_val})\n")
            elif dopant_choice == '3':
                f.write(f"mainrev1.set_material_params_al_doped({doping_val})\n")
            else:
                f.write("mainrev1.apply_global_params(mainrev1.hzo_par)\n")
            f.write(f"segments = {repr(segments)}\n")
            f.write(
                f"mainrev1.run_scatter_mode({Nx}, {Ny}, segments, {repeat_count}, '{dopant_choice}', {doping_val}, {doping_var}, {h_var})\n"
            )
            f.write("print('Auto scatter simulation finished.')\n")
        print(f"Auto script saved: {filename}")
    except Exception as e:
        print('Failed to create auto script:', e)
# [I] Multi-h & Multi-size Tc 계산 모드
################################################################################
def set_global_h(new_h):
    global h
    h = new_h
    logging.info(f"[Global] h set to {h}")

def get_equil_sweeps(N):
    base_sweeps=20
    delta= 0.4*(N-50)
    sweeps= base_sweeps+ delta
    if sweeps<1:
        sweeps=1
    return int(round(sweeps))

def compute_binder_and_chi(spins_original, T, J, n_equil_sweeps=20):
    temp_spins = np.copy(spins_original)
    Nx, Ny = temp_spins.shape
    Nsite = Nx * Ny
    sample_tfrac=[]
    for _ in range(n_equil_sweeps):
        for __ in range(Nx * Ny):
            x = np.random.randint(Nx)
            y = np.random.randint(Ny)
            attempt_flip(temp_spins, x, y, T, J)
        frac_tetra= phase_fractions(temp_spins)[3]
        sample_tfrac.append(frac_tetra)
    arr= np.array(sample_tfrac)
    m1= arr.mean()
    m2= (arr**2).mean()
    m4= (arr**4).mean()
    eps=1e-14
    denom= 3.0*(m2**2)+ eps
    binder= 1.0 - (m4/denom)
    chi= (Nsite/(k_B*T))*(m2- m1**2) if T>1e-9 else 0.0
    return binder, chi

def compute_final_phase(spins_original, T, J, n_equil_sweeps=20):
    temp_spins = np.copy(spins_original)
    Nx, Ny = temp_spins.shape
    for _ in range(n_equil_sweeps):
        for __ in range(Nx * Ny):
            x = np.random.randint(Nx)
            y = np.random.randint(Ny)
            attempt_flip(temp_spins, x, y, T, J)
    return phase_fractions(temp_spins)

def run_profile_simulation_Tc_multiN_multiH(segments,
                                           pdf_filename="Tc_multiN_multiH_result.pdf",
                                           N_list=[50,75,100],
                                           H_list=[50,100,150]):
    results= {}
    total_steps= sum(sc for (sT,eT,sc) in segments)
    up_segments=[]
    down_segments=[]
    cum=0
    for (sT,eT,sc) in segments:
        seg_steps= range(cum, cum+sc)
        if eT> sT:
            up_segments.append((seg_steps.start, seg_steps.stop-1))
        elif eT< sT:
            down_segments.append((seg_steps.start, seg_steps.stop-1))
        cum+= sc

    T_c_up_dict= {}
    T_c_down_dict= {}
    total_combinations= len(H_list)*len(N_list)*total_steps
    progress_step=0

    global_minT, global_maxT = get_minmax_temperature_from_segments(segments)
    t_ax_min = max(0, global_minT - 50)
    t_ax_max = global_maxT + 50

    print("시뮬레이션을 시작합니다... (Multi-h & Multi-size)")

    for h_val in H_list:
        set_global_h(h_val)
        results[h_val]= {}
        for Nsize in N_list:
            spins_off= np.random.randint(q,size=(Nsize,Nsize))

            step_list=[]
            temp_list=[]
            time_list=[]
            chi_list=[]
            binder_list=[]
            up_list=[]
            down_list=[]
            mono_list=[]
            tetra_list=[]
            final_up_list=[]
            final_down_list=[]
            final_mono_list=[]
            final_tetra_list=[]

            current_time=0.0
            cum=0
            n_equil_sweeps= get_equil_sweeps(Nsize)

            for (sT,eT,sc) in segments:
                dT= (eT- sT)/sc
                for local_stp in range(sc):
                    step= cum+ local_stp
                    T_now= sT + local_stp*dT
                    T_now= max(T_now, 0.0)

                    # KMC: 이벤트별 시간 합산
                    accepted_time_sum=0.0
                    for _a in range(Nsize*Nsize):
                        x = np.random.randint(Nsize)
                        y = np.random.randint(Nsize)
                        accepted, used_barrier = attempt_flip(spins_off, x, y, T_now, J)
                        if accepted and used_barrier is not None:
                            rate= nu0*np.exp(-used_barrier/(k_B*T_now)) if T_now>1e-9 else 0.0
                            if rate>1e-30:
                                dt_event= 1.0 / rate
                                accepted_time_sum += dt_event

                    dt= accepted_time_sum
                    current_time+= dt * TIME_SCALE

                    ph_now= phase_fractions(spins_off)
                    BL_now, Chi_now= compute_binder_and_chi(spins_off, T_now,J,n_equil_sweeps)
                    ph_final= compute_final_phase(spins_off, T_now,J,n_equil_sweeps)

                    step_list.append(step)
                    temp_list.append(T_now)
                    time_list.append(current_time)
                    chi_list.append(Chi_now)
                    binder_list.append(BL_now)
                    up_list.append(ph_now[0])
                    down_list.append(ph_now[1])
                    mono_list.append(ph_now[2])
                    tetra_list.append(ph_now[3])
                    final_up_list.append(ph_final[0])
                    final_down_list.append(ph_final[1])
                    final_mono_list.append(ph_final[2])
                    final_tetra_list.append(ph_final[3])

                    progress_step+=1
                    prog_ratio= progress_step/total_combinations*100
                    if (progress_step % (max(1,total_combinations//10)))==0:
                        print(f"진행상황: {round(prog_ratio)}% 완료...")

                cum+= sc

            results[h_val][Nsize]= {
                'step':  np.array(step_list),
                'temp':  np.array(temp_list),
                'time':  np.array(time_list),
                'chi':   np.array(chi_list),
                'binder':np.array(binder_list),
                'up':    np.array(up_list),
                'down':  np.array(down_list),
                'mono':  np.array(mono_list),
                'tetra': np.array(tetra_list),
                'f_up':  np.array(final_up_list),
                'f_down':np.array(final_down_list),
                'f_mono':np.array(final_mono_list),
                'f_tetra':np.array(final_tetra_list),
            }

    # 온도 상승/하강 구간별 chi 최대 -> Tc
    for i_up,(st_s, st_e) in enumerate(up_segments):
        T_c_up_dict[i_up]= {}
        for h_val in H_list:
            tlist=[]
            for Nsize in sorted(results[h_val].keys()):
                data= results[h_val][Nsize]
                stp_arr= data['step']
                chi_arr= data['chi']
                T_arr= data['temp']
                seg_idx= np.where((stp_arr>=st_s)&(stp_arr<=st_e))[0]
                if len(seg_idx)>0:
                    seg_chi= chi_arr[seg_idx]
                    max_i= np.argmax(seg_chi)
                    real_idx= seg_idx[max_i]
                    Tc_here= T_arr[real_idx]
                    tlist.append(Tc_here)
            if tlist:
                T_c_up_dict[i_up][h_val]= np.mean(tlist)
            else:
                T_c_up_dict[i_up][h_val]= None

    for i_down,(st_s, st_e) in enumerate(down_segments):
        T_c_down_dict[i_down]= {}
        for h_val in H_list:
            tlist=[]
            for Nsize in sorted(results[h_val].keys()):
                data= results[h_val][Nsize]
                stp_arr= data['step']
                chi_arr= data['chi']
                T_arr= data['temp']
                seg_idx= np.where((stp_arr>=st_s)&(stp_arr<=st_e))[0]
                if len(seg_idx)>0:
                    seg_chi= chi_arr[seg_idx]
                    max_i= np.argmax(seg_chi)
                    real_idx= seg_idx[max_i]
                    Tc_here= T_arr[real_idx]
                    tlist.append(Tc_here)
            if tlist:
                T_c_down_dict[i_down][h_val]= np.mean(tlist)
            else:
                T_c_down_dict[i_down][h_val]= None

    # PDF 생성
    try:
        with PdfPages(pdf_filename) as pdf:
            for h_val in H_list:
                local_colors= COLOR_LIST_NO_BLUE
                ls_styles= ['-','--',':','-.']

                ################################################################
                # chi vs step
                ################################################################
                fig_chi, ax_chi= plt.subplots(figsize=PDF_FIG_SIZE)
                ax_chi.set_xlabel("Step", fontsize=PDF_LABEL_FONT_SIZE)
                ax_chi.set_ylabel(r"$\chi$", fontsize=PDF_LABEL_FONT_SIZE)
                ax_chi.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
                ax_chi.grid(True, linestyle=':')

                ax_chi_T= ax_chi.twinx()
                ax_chi_T.set_ylabel("T (K)", fontsize=PDF_LABEL_FONT_SIZE, color=TEMP_COLOR)
                ax_chi_T.tick_params(axis='y', labelcolor=TEMP_COLOR, labelsize=PDF_TICK_FONT_SIZE)
                ax_chi_T.set_ylim(t_ax_min, t_ax_max)

                all_stp=[]
                for idx, Nsize in enumerate(sorted(results[h_val].keys())):
                    data= results[h_val][Nsize]
                    stp= data['step']
                    T_ary= data['temp']
                    chiA= data['chi']
                    ccol= local_colors[idx%len(local_colors)]
                    lsty= ls_styles[idx%len(ls_styles)]
                    ax_chi.plot(stp, chiA, color=ccol, ls=lsty, lw=1.5, label=f"N={Nsize}")
                    ax_chi_T.plot(stp, T_ary,'--', color=TEMP_COLOR, lw=1.0)
                    all_stp.extend(stp.tolist())
                if all_stp:
                    ax_chi.set_xlim(min(all_stp), max(all_stp))
                    ax_chi_T.set_xlim(min(all_stp), max(all_stp))
                ax_chi.legend(loc='upper left', fontsize=PDF_LEGEND_FONT_SIZE)
                pdf.savefig(fig_chi, bbox_inches='tight')
                plt.close(fig_chi)

                ################################################################
                # phase fraction vs step
                ################################################################
                fig_pf, ax_pf= plt.subplots(figsize=PDF_FIG_SIZE)
                ax_pf.set_xlabel("Step", fontsize=PDF_LABEL_FONT_SIZE)
                ax_pf.set_ylabel("Phase Fraction", fontsize=PDF_LABEL_FONT_SIZE)
                ax_pf.set_ylim(-0.05,1.05)
                ax_pf.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
                ax_pf.grid(True, linestyle=':')

                ax_pf_T= ax_pf.twinx()
                ax_pf_T.set_ylabel("T (K)", fontsize=PDF_LABEL_FONT_SIZE, color=TEMP_COLOR)
                ax_pf_T.tick_params(axis='y', labelcolor=TEMP_COLOR, labelsize=PDF_TICK_FONT_SIZE)
                ax_pf_T.set_ylim(t_ax_min, t_ax_max)

                all_stp_pf=[]
                for idx, Nsize in enumerate(sorted(results[h_val].keys())):
                    data= results[h_val][Nsize]
                    stp= data['step']
                    T_ary= data['temp']
                    upA= data['up']
                    dwA= data['down']
                    moA= data['mono']
                    teA= data['tetra']
                    lsty= ls_styles[idx%len(ls_styles)]
                    ax_pf.plot(stp, upA,   color=COLOR_UP,    ls=lsty,lw=1.5)
                    ax_pf.plot(stp, dwA,   color=COLOR_DOWN,  ls=lsty,lw=1.5)
                    ax_pf.plot(stp, moA,   color=COLOR_MONO,  ls=lsty,lw=1.5)
                    ax_pf.plot(stp, teA,   color=COLOR_TETRA, ls=lsty,lw=1.5)
                    ax_pf_T.plot(stp, T_ary,'--', color=TEMP_COLOR, lw=1.0)
                    all_stp_pf.extend(stp.tolist())
                if all_stp_pf:
                    ax_pf.set_xlim(min(all_stp_pf), max(all_stp_pf))
                    ax_pf_T.set_xlim(min(all_stp_pf), max(all_stp_pf))
                custom_phase_leg= [
                    Line2D([],[], color=COLOR_TETRA, lw=2, label=PHASE_LABEL_TETRA),
                    Line2D([],[], color=COLOR_MONO,  lw=2, label=PHASE_LABEL_MONO),
                    Line2D([],[], color=COLOR_UP,    lw=2, label=PHASE_LABEL_UP),
                    Line2D([],[], color=COLOR_DOWN,  lw=2, label=PHASE_LABEL_DOWN),
                ]
                ax_pf.legend(handles=custom_phase_leg, loc='upper left', fontsize=PDF_LEGEND_FONT_SIZE)
                pdf.savefig(fig_pf, bbox_inches='tight')
                plt.close(fig_pf)

                ################################################################
                # binder vs step
                ################################################################
                fig_binder, ax_binder= plt.subplots(figsize=PDF_FIG_SIZE)
                ax_binder.set_xlabel("Step", fontsize=PDF_LABEL_FONT_SIZE)
                ax_binder.set_ylabel("Binder", fontsize=PDF_LABEL_FONT_SIZE)
                ax_binder.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
                ax_binder.grid(True, linestyle=':')

                ax_binder_T= ax_binder.twinx()
                ax_binder_T.set_ylabel("T (K)", fontsize=PDF_LABEL_FONT_SIZE, color=TEMP_COLOR)
                ax_binder_T.tick_params(axis='y', labelcolor=TEMP_COLOR, labelsize=PDF_TICK_FONT_SIZE)
                ax_binder_T.set_ylim(t_ax_min, t_ax_max)

                all_stp_bd=[]
                for idx, Nsize in enumerate(sorted(results[h_val].keys())):
                    data= results[h_val][Nsize]
                    stp= data['step']
                    T_ary= data['temp']
                    bdA= data['binder']
                    ccol= local_colors[idx % len(local_colors)]
                    lsty= ls_styles[idx % len(ls_styles)]
                    ax_binder.plot(stp, bdA, color=ccol, ls=lsty, lw=1.5, label=f"N={Nsize}")
                    ax_binder_T.plot(stp, T_ary,'--', color=TEMP_COLOR, lw=1.0)
                    all_stp_bd.extend(stp.tolist())
                if all_stp_bd:
                    ax_binder.set_xlim(min(all_stp_bd), max(all_stp_bd))
                    ax_binder_T.set_xlim(min(all_stp_bd), max(all_stp_bd))
                ax_binder.legend(loc='upper left', fontsize=PDF_LEGEND_FONT_SIZE)
                pdf.savefig(fig_binder, bbox_inches='tight')
                plt.close(fig_binder)

                ################################################################
                # final phase fraction vs step
                ################################################################
                fig_pfF, ax_pfF= plt.subplots(figsize=PDF_FIG_SIZE)
                ax_pfF.set_xlabel("Step", fontsize=PDF_LABEL_FONT_SIZE)
                ax_pfF.set_ylabel("Final Phase Fraction", fontsize=PDF_LABEL_FONT_SIZE)
                ax_pfF.set_ylim(-0.05,1.05)
                ax_pfF.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
                ax_pfF.grid(True, linestyle=':')

                ax_pfF_T= ax_pfF.twinx()
                ax_pfF_T.set_ylabel("T (K)", fontsize=PDF_LABEL_FONT_SIZE, color=TEMP_COLOR)
                ax_pfF_T.tick_params(axis='y', labelcolor=TEMP_COLOR, labelsize=PDF_TICK_FONT_SIZE)
                ax_pfF_T.set_ylim(t_ax_min, t_ax_max)

                all_stp_pfF=[]
                for idx, Nsize in enumerate(sorted(results[h_val].keys())):
                    data= results[h_val][Nsize]
                    stp= data['step']
                    T_ary= data['temp']
                    f_upA= data['f_up']
                    f_downA= data['f_down']
                    f_monoA= data['f_mono']
                    f_tetraA= data['f_tetra']
                    lsty= ls_styles[idx % len(ls_styles)]
                    ax_pfF.plot(stp, f_upA,    color=COLOR_UP,    ls=lsty,lw=1.5)
                    ax_pfF.plot(stp, f_downA,  color=COLOR_DOWN,  ls=lsty,lw=1.5)
                    ax_pfF.plot(stp, f_monoA,  color=COLOR_MONO,  ls=lsty,lw=1.5)
                    ax_pfF.plot(stp, f_tetraA, color=COLOR_TETRA, ls=lsty,lw=1.5)
                    ax_pfF_T.plot(stp, T_ary,'--', color=TEMP_COLOR, lw=1.0)
                    all_stp_pfF.extend(stp.tolist())
                if all_stp_pfF:
                    ax_pfF.set_xlim(min(all_stp_pfF), max(all_stp_pfF))
                    ax_pfF_T.set_xlim(min(all_stp_pfF), max(all_stp_pfF))
                custom_leg_pfF= [
                    Line2D([],[], color=COLOR_TETRA, lw=2, label=PHASE_LABEL_TETRA),
                    Line2D([],[], color=COLOR_MONO,  lw=2, label=PHASE_LABEL_MONO),
                    Line2D([],[], color=COLOR_UP,    lw=2, label=PHASE_LABEL_UP),
                    Line2D([],[], color=COLOR_DOWN,  lw=2, label=PHASE_LABEL_DOWN),
                ]
                ax_pfF.legend(handles=custom_leg_pfF, loc='upper left', fontsize=PDF_LEGEND_FONT_SIZE)
                pdf.savefig(fig_pfF, bbox_inches='tight')
                plt.close(fig_pfF)

            # *** 하나의 그래프로 Tc_up / Tc_down 모두 표시 ***
            fig_Tc, ax_Tc = plt.subplots(figsize=PDF_FIG_SIZE)
            color_cycle = ['red','green','blue','orange','purple','brown','pink','gray','cyan','lime']
            marker_cycle= ['o','s','^','D','v','h','>','<','p','x']

            # up
            for i_up in sorted(T_c_up_dict.keys()):
                x_list=[]
                y_list=[]
                for idx,h_val in enumerate(H_list):
                    val= T_c_up_dict[i_up].get(h_val,None)
                    if val is not None:
                        x_list.append(h_val)
                        y_list.append(val)
                if x_list:
                    ccol= color_cycle[i_up % len(color_cycle)]
                    mk= marker_cycle[i_up % len(marker_cycle)]
                    ax_Tc.plot(x_list,y_list,color=ccol, marker=mk, lw=2,
                               label=f"Tc_up seg#{i_up+1}")

            # down
            for i_down in sorted(T_c_down_dict.keys()):
                x_list=[]
                y_list=[]
                for idx,h_val in enumerate(H_list):
                    val= T_c_down_dict[i_down].get(h_val,None)
                    if val is not None:
                        x_list.append(h_val)
                        y_list.append(val)
                if x_list:
                    ccol= color_cycle[(i_down+5) % len(color_cycle)]
                    mk= marker_cycle[(i_down+5) % len(marker_cycle)]
                    ax_Tc.plot(x_list,y_list,color=ccol, marker=mk, lw=2,
                               label=f"Tc_down seg#{i_down+1}")

            ax_Tc.set_xlabel("h (Å)", fontsize=PDF_LABEL_FONT_SIZE)
            ax_Tc.set_ylabel("Tc (K)", fontsize=PDF_LABEL_FONT_SIZE)
            ax_Tc.tick_params(axis='both', labelsize=PDF_TICK_FONT_SIZE)
            ax_Tc.grid(True, linestyle=':')
            ax_Tc.legend(loc='best', fontsize=PDF_LEGEND_FONT_SIZE)
            pdf.savefig(fig_Tc,bbox_inches='tight')
            plt.close(fig_Tc)

        print(f"\n시뮬레이션이 끝났습니다. PDF 저장 완료: {pdf_filename}")

    except Exception as e:
        print("PDF save error:", e)

################################################################################
# [J] 메인
################################################################################
def parse_n_values():
    print("chi 계산 시 사용할 격자크기 N을 여러 개 입력해주세요 (예: 50 75 100):")
    line = input().strip()
    vals = [int(x) for x in line.split()]
    return vals

def parse_h_values():
    print("시뮬레이션에서 사용할 h(Å 단위) 여러 개를 입력해주세요 (예: 5 10 15).")
    print("예) 5 => 실제 50Å, 10 => 100Å 등.")
    line = input().strip()
    vals = [float(x)*10 for x in line.split()]
    return vals

def main():
    initialize_barrier_file()
    dop_choice, dop_val = select_material()

    print("\nSelect mode:")
    print("1) Slider Mode (real-time)")
    print("2) Temperature Profile Mode (real-time + offline)")
    print("3) Multi-h & Multi-size Tc 계산 모드")
    print("4) Scatter Analysis Mode")

    mode= input("Mode? ")
    if mode not in ['1','2','3','4']:
        print("Invalid. Exiting.")
        sys.exit()

    if mode=='1':
        spins= np.random.randint(q, size=(N_for_analysis,N_for_analysis))
        while True:
            steps_in= input("How many steps? ")
            try:
                num_steps= int(steps_in)
                if num_steps>0:
                    break
            except:
                pass
            print("Must be positive integer!")

        fig, axs= plt.subplots(2,2, figsize=FIG_SIZE, constrained_layout=True)
        ax_spin= axs[0][0]
        ax_en=   axs[0][1]
        ax_pol=  axs[1][0]
        ax_ph=   axs[1][1]

        im= ax_spin.imshow(spins, cmap=BFS_CMAP, vmin=0, vmax=3, aspect='equal')
        ax_spin.axis('off')
        legend_e= [
            Patch(facecolor=COLOR_TETRA, label=PHASE_LABEL_TETRA),
            Patch(facecolor=COLOR_MONO,  label=PHASE_LABEL_MONO),
            Patch(facecolor=COLOR_UP,    label=PHASE_LABEL_UP),
            Patch(facecolor=COLOR_DOWN,  label=PHASE_LABEL_DOWN),
        ]
        ax_spin.legend(handles=legend_e, loc='upper left', fontsize=12)

        line_en,= ax_en.plot([],[],'r-', lw=2, label="Energy")
        ax_en.set_xlabel("Step (0-based)", fontsize=LABEL_FONT_SIZE)
        ax_en.set_ylabel("Energy (J)", fontsize=LABEL_FONT_SIZE)
        ax_en.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax_en.grid(True, linestyle=':')
        ax_en.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE)

        line_pol,= ax_pol.plot([],[],'b-', lw=2, label="Polarization")
        ax_pol.set_xlabel("Step (0-based)", fontsize=LABEL_FONT_SIZE)
        ax_pol.set_ylabel("Polarization", fontsize=LABEL_FONT_SIZE)
        ax_pol.set_ylim(-1,1)
        ax_pol.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax_pol.grid(True, linestyle=':')
        ax_pol.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE)

        line_up,=   ax_ph.plot([],[],'-', color=COLOR_UP,   lw=2, label=PHASE_LABEL_UP)
        line_down,= ax_ph.plot([],[],'-', color=COLOR_DOWN, lw=2, label=PHASE_LABEL_DOWN)
        line_mono,= ax_ph.plot([],[],'-', color=COLOR_MONO, lw=2, label=PHASE_LABEL_MONO)
        line_tetra,= ax_ph.plot([],[],'-', color=COLOR_TETRA,lw=2, label=PHASE_LABEL_TETRA)
        ax_ph.set_ylim(-0.05,1.05)
        ax_ph.set_xlabel("Step (0-based)", fontsize=LABEL_FONT_SIZE)
        ax_ph.set_ylabel("Phase Fraction", fontsize=LABEL_FONT_SIZE)
        ax_ph.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax_ph.grid(True, linestyle=':')
        ax_ph.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE)

        ax_ph_t= ax_ph.twinx()
        ax_ph_t.set_ylabel("T (K)", fontsize=LABEL_FONT_SIZE, color=TEMP_COLOR)
        ax_ph_t.tick_params(axis='y', labelcolor=TEMP_COLOR, labelsize=TICK_FONT_SIZE)
        ax_ph_t.set_ylim(250,1050)

        slid_ax= fig.add_axes([0.25,0.01,0.5,0.02])
        temp_slider= Slider(slid_ax,'Temp(K)',300,1000,valinit=600,valstep=10)
        temp_slider.label.set_size(LABEL_FONT_SIZE)

        step_count=0
        current_time=0.0
        step_list=[]
        time_list=[]
        en_list=[]
        pol_list=[]
        up_list=[]
        down_list=[]
        mono_list=[]
        tetra_list=[]
        temp_list=[]

        def init_anim():
            line_en.set_data([],[])
            line_pol.set_data([],[])
            line_up.set_data([],[])
            line_down.set_data([],[])
            line_mono.set_data([],[])
            line_tetra.set_data([],[])
            return (im, line_en, line_pol, line_up, line_down, line_mono, line_tetra)

        def update_anim(frame):
            nonlocal step_count, current_time
            if step_count> num_steps:
                anim.event_source.stop()
                return (im, line_en, line_pol, line_up, line_down, line_mono, line_tetra)

            T_now= float(temp_slider.val)
            T_now= max(T_now, 0.0)

            # 이벤트별 시간 합산
            accepted_time_sum= 0.0
            for _ in range(N_for_analysis*N_for_analysis):
                x= np.random.randint(N_for_analysis)
                y= np.random.randint(N_for_analysis)
                accepted, used_barrier = attempt_flip(spins, x, y, T_now, J)
                if accepted and used_barrier is not None:
                    rate= nu0*np.exp(-used_barrier/(k_B*T_now)) if T_now>1e-9 else 0.0
                    if rate>1e-30:
                        accepted_time_sum += (1.0/rate)

            dt= accepted_time_sum
            current_time+= dt* TIME_SCALE

            E_now = total_energy(spins, T_now, J)
            P_now= polarization(spins)
            ph_now= phase_fractions(spins)

            # BFS
            bfs_up, bfs_down, bfs_mono_val, bfs_tetra_val, bfs_all = compute_bfs_radius_all(spins)

            logging.info(
                f"[Slider] step={step_count}, T={T_now:.1f}, dt={dt:.3e}, time={current_time:.3e}s, "
                f"E={E_now:.3e}, P={P_now:.3e}, up={ph_now[0]:.3f}, down={ph_now[1]:.3f}, mono={ph_now[2]:.3f}, tetra={ph_now[3]:.3f}, "
                f"BFSup={bfs_up:.2f}, BFSdown={bfs_down:.2f}, BFSmono={bfs_mono_val:.2f}, BFStet={bfs_tetra_val:.2f}, BFSall={bfs_all:.2f}"
            )

            step_list.append(step_count)
            time_list.append(current_time)
            en_list.append(E_now)
            pol_list.append(P_now)
            up_list.append(ph_now[0])
            down_list.append(ph_now[1])
            mono_list.append(ph_now[2])
            tetra_list.append(ph_now[3])
            temp_list.append(T_now)

            im.set_data(spins)

            line_en.set_data(step_list,en_list)
            ax_en.set_xlim(0,num_steps)
            if en_list:
                ax_en.set_ylim(0,max(en_list)*1.1)

            line_pol.set_data(step_list,pol_list)
            ax_pol.set_xlim(0,num_steps)

            line_up.set_data(step_list, up_list)
            line_down.set_data(step_list, down_list)
            line_mono.set_data(step_list, mono_list)
            line_tetra.set_data(step_list, tetra_list)
            ax_ph.set_xlim(0,num_steps)

            step_count+=1
            line_temp_profile,= ax_ph_t.plot(step_list, temp_list,'--', color=TEMP_COLOR, lw=2)
            ax_ph_t.set_xlim(0,num_steps)

            return (im, line_en, line_pol, line_up, line_down, line_mono, line_tetra, line_temp_profile)

        anim= FuncAnimation(fig, update_anim, init_func=init_anim,
                            frames=num_steps+1, interval=300, blit=False, repeat=False)
        plt.show()

    elif mode=='2':
        segs= parse_temperature_profile()
        global_minT, global_maxT = get_minmax_temperature_from_segments(segs)
        t_ax_min = max(0, global_minT - 50)
        t_ax_max = global_maxT + 50

        total_steps= sum(x[2] for x in segs)
        spins= np.random.randint(q, size=(N_for_analysis,N_for_analysis))

        fig, axs= plt.subplots(2,2, figsize=FIG_SIZE, constrained_layout=True)
        ax_spin= axs[0][0]
        ax_en=   axs[0][1]
        ax_pol=  axs[1][0]
        ax_ph=   axs[1][1]

        im= ax_spin.imshow(spins, cmap=BFS_CMAP, vmin=0, vmax=q-1, aspect='equal')
        ax_spin.axis('off')
        legend_e= [
            Patch(facecolor=COLOR_TETRA, label=PHASE_LABEL_TETRA),
            Patch(facecolor=COLOR_MONO,  label=PHASE_LABEL_MONO),
            Patch(facecolor=COLOR_UP,    label=PHASE_LABEL_UP),
            Patch(facecolor=COLOR_DOWN,  label=PHASE_LABEL_DOWN),
        ]
        ax_spin.legend(handles=legend_e, loc='upper left', fontsize=12)

        line_en,= ax_en.plot([],[],'r-', lw=2)
        ax_en.set_xlabel("Step", fontsize=LABEL_FONT_SIZE)
        ax_en.set_ylabel("Energy (J)", fontsize=LABEL_FONT_SIZE)
        ax_en.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax_en.grid(True, linestyle=':')

        line_pol,= ax_pol.plot([],[],'b-', lw=2)
        ax_pol.set_xlabel("Step", fontsize=LABEL_FONT_SIZE)
        ax_pol.set_ylabel("Polarization", fontsize=LABEL_FONT_SIZE)
        ax_pol.set_ylim(-1,1)
        ax_pol.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax_pol.grid(True, linestyle=':')

        line_up,=   ax_ph.plot([],[],'-', color=COLOR_UP,   lw=2)
        line_down,= ax_ph.plot([],[],'-', color=COLOR_DOWN, lw=2)
        line_mono,= ax_ph.plot([],[],'-', color=COLOR_MONO, lw=2)
        line_tetra,= ax_ph.plot([],[],'-', color=COLOR_TETRA,lw=2)
        ax_ph.set_ylim(-0.05,1.05)
        ax_ph.set_xlabel("Step", fontsize=LABEL_FONT_SIZE)
        ax_ph.set_ylabel("Phase Fraction", fontsize=LABEL_FONT_SIZE)
        ax_ph.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax_ph.grid(True, linestyle=':')

        ax_ph_t= ax_ph.twinx()
        ax_ph_t.set_ylabel("T (K)", fontsize=LABEL_FONT_SIZE, color=TEMP_COLOR)
        ax_ph_t.tick_params(axis='y', labelcolor=TEMP_COLOR, labelsize=TICK_FONT_SIZE)
        ax_ph_t.set_ylim(t_ax_min, t_ax_max)

        step_count=0
        current_time=0.0
        step_list=[]
        time_list=[]
        en_list=[]
        pol_list=[]
        up_list=[]
        down_list=[]
        mono_list=[]
        tetra_list=[]
        temp_list=[]

        def init_prof_mode():
            line_en.set_data([],[])
            line_pol.set_data([],[])
            line_up.set_data([],[])
            line_down.set_data([],[])
            line_mono.set_data([],[])
            line_tetra.set_data([],[])
            return (im, line_en, line_pol, line_up, line_down, line_mono, line_tetra)

        def update_prof_mode(frame):
            nonlocal step_count, current_time
            if step_count> total_steps:
                prof_anim.event_source.stop()
                return (im, line_en, line_pol, line_up, line_down, line_mono, line_tetra)

            T_now= T_func_profile(segs, step_count)
            T_now= max(T_now, 0.0)
            save_barrier_values(T_now,J)

            # KMC 이벤트별 시간 합산
            accepted_time_sum=0.0
            for _ in range(N_for_analysis*N_for_analysis):
                x = np.random.randint(N_for_analysis)
                y = np.random.randint(N_for_analysis)
                accepted, used_barrier = attempt_flip(spins, x, y, T_now, J)
                if accepted and used_barrier is not None:
                    rate= nu0*np.exp(-used_barrier/(k_B*T_now)) if T_now>1e-9 else 0.0
                    if rate>1e-30:
                        accepted_time_sum+= 1.0/rate

            dt= accepted_time_sum
            current_time+= dt * TIME_SCALE

            E_now = total_energy(spins, T_now, J)
            P_now= polarization(spins)
            ph_now= phase_fractions(spins)

            bfs_up, bfs_down, bfs_mono_val, bfs_tetra_val, bfs_all = compute_bfs_radius_all(spins)

            logging.info(
                f"[TempProfile] step={step_count}, T={T_now:.1f}, dt={dt:.3e}, time={current_time:.3e}s, "
                f"E={E_now:.3e}, P={P_now:.3e}, up={ph_now[0]:.3f}, down={ph_now[1]:.3f}, mono={ph_now[2]:.3f}, tetra={ph_now[3]:.3f}, "
                f"BFSup={bfs_up:.2f}, BFSdown={bfs_down:.2f}, BFSmono={bfs_mono_val:.2f}, BFStet={bfs_tetra_val:.2f}, BFSall={bfs_all:.2f}"
            )

            step_list.append(step_count)
            time_list.append(current_time)
            en_list.append(E_now)
            pol_list.append(P_now)
            up_list.append(ph_now[0])
            down_list.append(ph_now[1])
            mono_list.append(ph_now[2])
            tetra_list.append(ph_now[3])
            temp_list.append(T_now)

            im.set_data(spins)
            ax_spin.set_title(f"T={T_now:.1f}K", fontsize=12)

            line_en.set_data(step_list,en_list)
            ax_en.set_xlim(0,total_steps)
            if en_list:
                ax_en.set_ylim(0, max(en_list)*1.1)

            line_pol.set_data(step_list,pol_list)
            ax_pol.set_xlim(0,total_steps)

            line_up.set_data(step_list, up_list)
            line_down.set_data(step_list, down_list)
            line_mono.set_data(step_list, mono_list)
            line_tetra.set_data(step_list, tetra_list)
            ax_ph.set_xlim(0,total_steps)

            step_count+=1
            line_temp_profile,= ax_ph_t.plot(step_list, temp_list,'--', color=TEMP_COLOR, lw=2)
            ax_ph_t.set_xlim(0,total_steps)

            return (im, line_en, line_pol, line_up, line_down, line_mono, line_tetra, line_temp_profile)

        prof_anim= FuncAnimation(fig, update_prof_mode, init_func=init_prof_mode,
                                 frames= total_steps+1, interval=300, blit=False, repeat=False)
        plt.show()

        # 오프라인 시뮬
        run_profile_simulation_offline(
            segments= segs,
            gif_filename="Temperature_animation.gif",
            pdf_filename="Temperature_result.pdf",
            global_minT= global_minT,
            global_maxT= global_maxT
        )

    elif mode=='3':
        segs= parse_temperature_profile()
        N_list= parse_n_values()
        H_list= parse_h_values()
        print("\n시뮬레이션을 시작합니다.\n")
        run_profile_simulation_Tc_multiN_multiH(
            segments= segs,
            pdf_filename="Tc_multiN_multiH_result.pdf",
            N_list=N_list,
            H_list=H_list
        )

    elif mode=='4':
        Nx_in = input("Lattice size Nx: ").strip()
        Ny_in = input("Lattice size Ny: ").strip()
        try:
            Nx = int(Nx_in)
        except Exception:
            Nx = N_for_analysis
        try:
            Ny = int(Ny_in)
        except Exception:
            Ny = Nx
        segments = parse_temperature_profile_en()
        rep_in = input("How many repetitions? ").strip()
        try:
            rep_count = int(rep_in)
        except Exception:
            rep_count = 1

        h_var = 0.0
        if input("Apply thickness variation? (y/n): ").strip().lower() == 'y':
            hv = input("Variation range in nm (±): ").strip()
            try:
                h_var = float(hv) * 10.0
            except Exception:
                h_var = 0.0

        doping_var = 0.0
        if input("Apply dopant variation? (y/n): ").strip().lower() == 'y':
            dv = input("Variation range in % (±): ").strip()
            try:
                doping_var = float(dv)
            except Exception:
                doping_var = 0.0

        print("\nStarting scatter mode...\n")
        run_scatter_mode(
            Nx,
            Ny,
            segments,
            rep_count,
            dop_choice,
            dop_val,
            doping_var,
            h_var,
        )

    print("All done.")
    logging.info("Simulation ended.")

if __name__=="__main__":
    main()
