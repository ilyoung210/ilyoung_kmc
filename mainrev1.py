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

# 3D 플롯에 필요
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as m3d

# Voronoi 등에 필요한 라이브러리
from scipy.spatial import cKDTree
from skimage.measure import find_contours

# 추가: Z축 formatting
from matplotlib.ticker import ScalarFormatter

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

# -- 여기서부터 새로 추가된 진한 파랑/빨강을 위한 커스텀 colormap 정의 --
COOLWARM_CUSTOM = mcolors.LinearSegmentedColormap.from_list(
    'coolwarm_custom',
    [
        (0, '#00008B'),  # 좀 더 진한 파란색
        (0.5, '#FFFFFF'),
        (1, '#8B0000')   # 좀 더 진한 빨간색
    ]
)

################################################################################
# [C] 시뮬레이션 상수 및 기본 함수 (초기값: HZO 기반)
################################################################################
logging.basicConfig(
    filename='simulation.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s'
)

TIME_SCALE = 3.0e16
k_B = 1.380649e-23
E_field = 0.0

# --- 전역 물성 파라미터(초기: HZO).---
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

h = 100
nu0 = 1.0e13
CELL_SIZE_ANG = 5.0
N_for_analysis = 50

transition_types = [
    ('Up','Down'),('Down','Up'),
    ('Tetra','Up'),('Tetra','Down'),
    ('Tetra','Mono'),('Mono','Up'),
    ('Mono','Down'),('Mono','Tetra'),
    ('Up','Mono'),('Down','Mono'),
    ('Up','Tetra'),('Down','Tetra'),
]

################################################################################
# [C-1] HfO2, HZO, ZrO2, SiHfO2 파라미터 (예시 임의 값) + 보간 함수
################################################################################
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

al_hf_par = {
    "E_p": 3.8e-20,  "E_m": -0.45e-20, "E_t": 6.2e-20,  "J": 0.09e-20,
    "a_1": 3.1e-24,  "a_2": 0.0,       "a_3": 4.2e-23,
    "sigma_1": 8.2e-19, "sigma_2": 1.05e-18, "sigma_3": 7.7e-19,
    "interface_pairs": {
        frozenset({0,1}): 0.0,
        frozenset({0,2}): 3.6e-20,
        frozenset({1,2}): 3.6e-20,
        frozenset({0,3}): 4.7e-21,
        frozenset({1,3}): 4.7e-21,
        frozenset({2,3}): 2.9e-20
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
                new_val = valA*(1-ratio) + valB*ratio
                new_ip[fs] = new_val
            new_par[k] = new_ip
        else:
            valA = parA[k]
            valB = parB[k]
            new_val = valA*(1-ratio) + valB*ratio
            new_par[k] = new_val
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
    if doping_zr<=0:
        apply_global_params(hf_par)
        return
    if doping_zr>=100:
        apply_global_params(zr_par)
        return
    if doping_zr==50:
        apply_global_params(hzo_par)
        return

    if doping_zr<50:
        ratio = doping_zr/50.0
        new_par = interpolate_params(hf_par, hzo_par, ratio)
        apply_global_params(new_par)
    else:
        ratio = (doping_zr - 50)/50.0
        new_par = interpolate_params(hzo_par, zr_par, ratio)
        apply_global_params(new_par)

def set_material_params_si_doped(doping_si):
    if doping_si<=0:
        apply_global_params(hf_par)
        return
    if doping_si>=100:
        apply_global_params(si_hf_par)
        return

    ratio = doping_si/100.0
    new_par = interpolate_params(hf_par, si_hf_par, ratio)
    apply_global_params(new_par)

def set_material_params_al_doped(doping_al):
    if doping_al<=0:
        apply_global_params(hf_par)
        return
    if doping_al>=100:
        apply_global_params(al_hf_par)
        return

    ratio = doping_al/100.0
    new_par = interpolate_params(hf_par, al_hf_par, ratio)
    apply_global_params(new_par)

################################################################################
# [C-2] 초기화 함수
################################################################################
def initialize_barrier_file(filename='barrier_values.csv'):
    with open(filename,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(["Temperature(K)","Old_State","New_State","Barrier(J)"])

################################################################################
# [C-3] 물성 선택 함수
################################################################################
def select_material():
    print("어떤 물질/도핑으로 시작하시겠습니까?")
    print("1) Zr-doped HfO2")
    print("2) Si-doped HfO2")
    print("3) Al-doped HfO2")
    choice = input("번호를 선택하세요: ").strip()

    if choice == '1':
        print("Zr-doped HfO2를 선택하였습니다.")
        doping_str = input("Zr 농도를 0~100(%) 범위로 입력하세요 (예: 50 => HZO, 0 => HfO2, 100 => ZrO2): ").strip()
        try:
            doping_val = float(doping_str)
        except:
            doping_val = 50.0
        if doping_val<0: doping_val=0.0
        if doping_val>100: doping_val=100.0
        set_material_params_zr_doped(doping_val)
        logging.info(f"[Material] Using Zr-doped HfO2 with doping={doping_val}%")
    elif choice == '2':
        print("Si-doped HfO2를 선택하였습니다.")
        doping_str = input("Si 농도를 0~100(%) 범위로 입력하세요 (예: 0 => HfO2, 100 => SiHfO2(가정)): ").strip()
        try:
            doping_val = float(doping_str)
        except:
            doping_val = 0.0
        if doping_val<0: doping_val=0.0
        if doping_val>100: doping_val=100.0
        set_material_params_si_doped(doping_val)
        logging.info(f"[Material] Using Si-doped HfO2 with doping={doping_val}%")
    elif choice == '3':
        print("Al-doped HfO2를 선택하였습니다.")
        doping_str = input("Al 농도를 0~100(%) 범위로 입력하세요 (예: 0 => HfO2, 100 => AlHfO2(가정)): ").strip()
        try:
            doping_val = float(doping_str)
        except:
            doping_val = 0.0
        if doping_val<0: doping_val=0.0
        if doping_val>100: doping_val=100.0
        set_material_params_al_doped(doping_val)
        logging.info(f"[Material] Using Al-doped HfO2 with doping={doping_val}%")
    else:
        print("잘못된 입력. 기본값(HZO)로 진행합니다.")
        apply_global_params(hzo_par)
        logging.info("[Material] Invalid input => default(HZO) used.")

################################################################################
# [D] 스핀 상태 / 에너지 함수들
################################################################################
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
    # Up-Up or Down-Down => -J, Up-Down => +J
    if (is_Up(s1) and is_Up(s2)) or (is_Down(s1) and is_Down(s2)):
        return -J
    elif (is_Up(s1) and is_Down(s2)) or (is_Down(s1) and is_Up(s2)):
        return +J
    return 0.0

def interface_energy(s1,s2,J):
    if s1==s2:
        return 0.0
    return interface_pairs.get(frozenset({s1,s2}), 0.0)

def local_energy(spins,x,y,N,T,J):
    s= spins[x,y]
    E_loc = bulk_energy(s,J,T,h) + field_energy(s,J)
    neigh = [((x-1)%N,y),((x+1)%N,y),(x,(y-1)%N),(x,(y+1)%N)]
    E_nb=0.0
    for (xx,yy) in neigh:
        s2= spins[xx,yy]
        E_nb += potts_interaction_energy(s,s2,J)
        E_nb += interface_energy(s,s2,J)
    return E_loc + E_nb

def delta_energy(spins,x,y,new_s,N,T,J):
    old_s= spins[x,y]
    E_old= local_energy(spins,x,y,N,T,J)
    spins[x,y]= new_s
    E_new= local_energy(spins,x,y,N,T,J)
    spins[x,y]= old_s
    return E_new - E_old

def total_energy(spins,N,T,J):
    E_tot=0.0
    Nsize= spins.shape[0]
    for x in range(Nsize):
        for y in range(Nsize):
            s= spins[x,y]
            E_tot += bulk_energy(s,J,T,h)
            E_tot += field_energy(s,J)
            s_r= spins[x,(y+1)%Nsize]
            s_d= spins[(x+1)%Nsize,y]
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
    # 예시 값(임의)
    if (is_Up(old_state) and is_Down(new_state)) or (is_Down(old_state) and is_Up(new_state)):
        # Up <-> Down은 barrier=0 (예시)
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

def attempt_flip(spins,x,y,N,T,J):
    old_s= spins[x,y]

    if is_T(old_s):
        poss=[0,1,2]  # T -> Up/Down/Mono
    elif is_M(old_s):
        poss=[0,1,3]  # M -> Up/Down/T
    elif is_Up(old_s):
        poss=[1,2,3]  # Up-> Down/Mono/T
    elif is_Down(old_s):
        poss=[0,2,3]  # Down-> Up/Mono/T
    else:
        return (False, None)

    if not poss:
        return (False, None)

    new_s= np.random.choice(poss)
    dE= delta_energy(spins,x,y,new_s,N,T,J)
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

def compute_total_rate(spins,N,T,J):
    tot=0.0
    Nsize = spins.shape[0]
    for x in range(Nsize):
        for y in range(Nsize):
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
        x= np.random.randint(spins.shape[0])
        y= np.random.randint(spins.shape[1])
        attempt_flip(spins,x,y, spins.shape[0],T,J)  # 샘플링을 위한 임시 flip
        frac_tetra= phase_fractions(spins)[3]
        ft_vals.append(frac_tetra)
    arr= np.array(ft_vals)
    m1= arr.mean()
    m2= (arr**2).mean()
    chi= (Nsite/(k_B*T))*(m2- m1**2) if T>1e-9 else 0.0
    return chi

################################################################################
# [E] BFS + 결정립 분석 함수들
################################################################################
def label_connected_components(spin_arr):
    N= spin_arr.shape[0]
    label_arr= np.zeros((N,N), dtype=int)
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
            for (nx,ny) in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                nx%=N; ny%=N
                if spin_arr[nx,ny]==s and label_arr[nx,ny]==0:
                    label_arr[nx,ny]= cur_label
                    queue.append((nx,ny))
                    regpix.append((nx,ny))
        regions[cur_label]= {"spin": s, "pixels": regpix}

    for x in range(N):
        for y in range(N):
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
    N= label_arr.shape[0]
    bcoords=[]
    for (x,y) in region_pixels:
        neighbors= [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
        is_boundary=False
        for(nx,ny) in neighbors:
            nx%=N; ny%=N
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
# [F] BFS 시각화 + Grain size 분석
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
    ax_heat.imshow(spin_arr, cmap=BFS_CMAP, origin='upper', vmin=0, vmax=3)
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
    ax_heat.imshow(spin_arr, cmap=BFS_CMAP, origin='upper', vmin=0, vmax=3)
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
    if step_label:
        fig_phases.suptitle(f"Step={step_label} BFS Grain Size(Phase-wise)", fontsize=PDF_LABEL_FONT_SIZE)
    else:
        fig_phases.suptitle("BFS Grain Size(Phase-wise)", fontsize=PDF_LABEL_FONT_SIZE)

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
# [G] BFS+Voronoi (혼합) - 큰 결정립 내부 세분화
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

    sub_centers_arr= np.array(sub_centers)
    N= spin_arr.shape[0]
    tree= cKDTree(sub_centers_arr)
    coords= np.indices((N,N)).reshape(2,-1).T
    dists, inds= tree.query(coords)
    final_label= np.array([ sub_ids[i] for i in inds]).reshape(N,N)
    return final_label, spin_map

def plot_mixed_bfs_overlay(final_label,
                           spin_map,
                           step_label="",
                           scale_factor=50.0):
    N= final_label.shape[0]
    fig, ax= plt.subplots(figsize=PDF_FIG_SIZE)
    ax.imshow(np.ones((N,N,3)), origin='upper', vmin=0, vmax=1)
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

    overlay_rgb= np.zeros((N,N,3), dtype=np.float32)
    unique_ids= np.unique(final_label)
    for x in range(N):
        for y in range(N):
            sub_id= final_label[x,y]
            spin_s= spin_map.get(sub_id,2)
            cval= color_map.get(spin_s, (1,1,1,1))
            overlay_rgb[x,y]= cval[:3]

    ax.imshow(overlay_rgb, origin='upper')

    from skimage.measure import find_contours
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
    if step_label:
        fig_phases.suptitle(f"Step={step_label} MixedBFS(Phase-wise)", fontsize=PDF_LABEL_FONT_SIZE)
    else:
        fig_phases.suptitle("MixedBFS(Phase-wise)", fontsize=PDF_LABEL_FONT_SIZE)

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
# [H] Temperature Profile 파싱/함수
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
# [H-2] BFS Grain-size vs step 계산
################################################################################
def compute_bfs_radius_all(spins):
    label_arr, regions = label_connected_components(spins)
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

################################################################################
# [H-3] 오프라인 시뮬레이션 + PDF/GIF
################################################################################
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
            accepted, used_barrier= attempt_flip(spins_off,x,y,N_for_analysis,T_now,J)
            if accepted:
                rate= nu0*np.exp(-used_barrier/(k_B*T_now)) if (T_now>1e-9 and used_barrier is not None) else 0.0
                if rate>1e-30:
                    dt_event= 1.0 / rate
                    accepted_time_sum += dt_event

        dt= accepted_time_sum
        current_time += dt * TIME_SCALE

        save_barrier_values(T_now,J)

        E_now= total_energy(spins_off, N_for_analysis, T_now, J)
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

    # --- GIF 생성 ---
    fig_off, axs_off= plt.subplots(2,2, figsize=(8,6), constrained_layout=True)
    ax_spin= axs_off[0][0]
    ax_en=   axs_off[0][1]
    ax_pol=  axs_off[1][0]
    ax_ph=   axs_off[1][1]

    im_off= ax_spin.imshow(spin_snapshots[0], cmap=BFS_CMAP, vmin=0, vmax=q-1)
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
    split_folder = "split_figures"
    if not os.path.exists(split_folder):
        os.makedirs(split_folder)

    try:
        with PdfPages(pdf_filename) as pdf:
            # 필요하면 중간에 원하는 그림들 추가 가능
            print(f"{pdf_filename} saved (offline).")
    except Exception as e:
        print("PDF save error:", e)

################################################################################
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
    temp_spins= np.copy(spins_original)
    N= temp_spins.shape[0]
    Nsite= N*N
    sample_tfrac=[]
    for _ in range(n_equil_sweeps):
        for __ in range(N*N):
            x= np.random.randint(N)
            y= np.random.randint(N)
            attempt_flip(temp_spins,x,y,N,T,J)
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
    temp_spins= np.copy(spins_original)
    N= temp_spins.shape[0]
    for _ in range(n_equil_sweeps):
        for __ in range(N*N):
            x= np.random.randint(N)
            y= np.random.randint(N)
            attempt_flip(temp_spins,x,y,N,T,J)
    return phase_fractions(temp_spins)

def run_profile_simulation_Tc_multiN_multiH(segments,
                                           pdf_filename="Tc_multiN_multiH_result.pdf",
                                           N_list=[50,75,100],
                                           H_list=[50,100,150]):
    # (기능 미구현 - 필요시 사용자 맞춤 추가)
    pass

################################################################################
# [J] (수정된) 모드 4: 실시간 3D 에너지 프로파일 + 2D Contour + NEB 다이어그램
################################################################################

def local_energy_of_site(spins, x, y, new_s, N, T, J):
    """
    해당 (x,y)에 new_s 상을 '가정'했을 때의 local_energy.
    spins를 임시로 바꿨다가 되돌린다.
    """
    old_s = spins[x,y]
    spins[x,y] = new_s
    E_loc = local_energy(spins, x, y, N, T, J)
    spins[x,y] = old_s
    return E_loc

def find_surrounded_m_site(spins):
    """
    '사발이 M phase로 둘러싸인 상태' = 자기 자신도 M 이고, 이웃 4개도 모두 M인 첫 위치를 찾는다.
    못찾으면 None
    """
    N = spins.shape[0]
    for x in range(N):
        for y in range(N):
            if is_M(spins[x,y]):
                # 이웃 검사
                neigh = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
                all_m = True
                for (nx, ny) in neigh:
                    nx%=N; ny%=N
                    if not is_M(spins[nx,ny]):
                        all_m = False
                        break
                if all_m:
                    return (x,y)
    return None

def mode_4_realtime_energy_profile():
    """
    - 온도 프로파일 입력 → 매 step KMC 진행
    - 한 화면에서: (1×4) 서브플롯
      1) BFS(Spin) Heatmap
      2) 3D Surface Plot (local energy) [색상: COOLWARM_CUSTOM], 범례고정
      3) 2D Contour Plot (local energy) [색상: COOLWARM_CUSTOM, 컬러바 고정]
      4) NEB 다이어그램(M->O->T 경로, Barrier)
    """
    segs= parse_temperature_profile()
    total_steps= sum(s[2] for s in segs)

    # 온도 범위
    global_minT, global_maxT = get_minmax_temperature_from_segments(segs)
    t_ax_min = max(0, global_minT - 50)
    t_ax_max = global_maxT + 50

    # 시뮬용 초기 spins
    Nsize = N_for_analysis
    spins = np.random.randint(q, size=(Nsize, Nsize))

    # 애니메이션용 기록
    step_list= []
    temp_list= []
    spin_snapshots= []
    energy_mats= []
    current_time= 0.0

    # 3D/Contour 에서 z(에너지) 스케일을 고정하기 위해 임의 상한 지정 (원하는 값으로 조정 가능)
    # 예: 0 ~ 2.0e-19 J 사이로 고정
    ZMAX_FIXED = 2.0e-19

    # figure 설정: 1행 4열
    fig_4 = plt.figure(figsize=(20,5))
    ax_heat = fig_4.add_subplot(1,4,1)
    ax_surf = fig_4.add_subplot(1,4,2, projection='3d')
    ax_cont = fig_4.add_subplot(1,4,3)
    ax_neb  = fig_4.add_subplot(1,4,4)

    # 1) Heatmap 초기
    im_heat = ax_heat.imshow(spins, cmap=BFS_CMAP, vmin=0, vmax=q-1)
    ax_heat.axis('off')
    ax_heat.set_title("Heatmap (Step=0)", fontsize=12)

    # 2) 3D Surface 초기
    X = np.arange(Nsize)
    Y = np.arange(Nsize)
    Xg, Yg = np.meshgrid(X, Y)
    Z0 = np.zeros((Nsize,Nsize))  # 일단 0으로 초기
    surf = ax_surf.plot_surface(Xg, Yg, Z0, cmap=COOLWARM_CUSTOM, linewidth=0, antialiased=False)
    ax_surf.set_title("Local Energy 3D (Step=0)", fontsize=12)
    ax_surf.set_xlabel("X")
    ax_surf.set_ylabel("Y")
    ax_surf.set_zlabel("Energy (J)")
    # X,Y 범위 [0,Nsize]
    ax_surf.set_xlim(0, Nsize)
    ax_surf.set_ylim(0, Nsize)
    # Z축: 0~ZMAX_FIXED로 고정
    ax_surf.set_zlim(0, ZMAX_FIXED)
    # 3D 그림의 grid 제거
    ax_surf.grid(False)
    # 지수표시 없애기
    ax_surf.zaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax_surf.zaxis.get_major_formatter().set_useOffset(False)

    # 3) 2D contour 초기
    cont_im = ax_cont.imshow(Z0, origin='upper', cmap=COOLWARM_CUSTOM, vmin=0, vmax=ZMAX_FIXED)
    ax_cont.set_title("Local Energy Contour (Step=0)", fontsize=12)
    ax_cont.axis('off')
    ax_cont.contour(Z0, levels=10, colors='k', alpha=0.7)

    # 컬러바(Contour기준) 고정 범위
    cbar = plt.colorbar(cont_im, ax=ax_cont, shrink=0.7, pad=0.02)
    cbar.set_label("Local Energy (J)", fontsize=LABEL_FONT_SIZE)
    cbar.set_clim(0, ZMAX_FIXED)
    cbar.update_normal(cont_im)

    # 4) NEB 다이어그램
    ax_neb.set_title("NEB(M->O->T)", fontsize=12)
    ax_neb.set_xlabel("Reaction Coord", fontsize=LABEL_FONT_SIZE)
    ax_neb.set_ylabel("Energy (J)", fontsize=LABEL_FONT_SIZE)
    # x좌표: [0,0.5,1,1.5,2]
    x_neb = [0,0.5,1,1.5,2]
    # 초기(0 스텝)에선 임시로 모두 0
    y_neb_init = [0,0,0,0,0]
    line_neb, = ax_neb.plot(x_neb, y_neb_init, '-o', color='purple')
    ax_neb.set_xlim(-0.1, 2.1)
    ax_neb.set_ylim(0, ZMAX_FIXED*1.2)  # 적당히
    ax_neb.grid(True, linestyle=':')

    # '사발이 M으로 둘러싸인 셀' 한 곳 찾기
    m_site = find_surrounded_m_site(spins)

    #================================================
    def init_anim():
        # 아무것도 안 해도 됨
        return (im_heat, surf, cont_im, line_neb)

    def update_anim(frame):
        nonlocal current_time
        if frame> total_steps:
            anim_4.event_source.stop()
            return (im_heat, surf, cont_im, line_neb)

        # 현재 T
        T_now = T_func_profile(segs, frame)
        T_now = max(T_now, 0.0)

        # KMC 스텝
        accepted_time_sum=0.0
        for _ in range(Nsize*Nsize):
            x= np.random.randint(Nsize)
            y= np.random.randint(Nsize)
            accepted, used_barrier= attempt_flip(spins,x,y,Nsize,T_now,J)
            if accepted and used_barrier is not None:
                rate= nu0*np.exp(-used_barrier/(k_B*T_now)) if T_now>1e-9 else 0.0
                if rate>1e-30:
                    dt_event= 1.0/rate
                    accepted_time_sum+= dt_event
        current_time += (accepted_time_sum * TIME_SCALE)

        step_list.append(frame)
        temp_list.append(T_now)

        # Heatmap 업데이트
        im_heat.set_data(spins)
        ax_heat.set_title(f"Heatmap (Step={frame}, T={T_now:.1f}K)", fontsize=12)

        # Local energy matrix
        matE = np.zeros((Nsize,Nsize))
        for xx in range(Nsize):
            for yy in range(Nsize):
                matE[xx,yy] = local_energy(spins, xx, yy, Nsize, T_now, J)
        energy_mats.append(matE.copy())
        spin_snapshots.append(spins.copy())

        # 3D surface 갱신
        ax_surf.clear()
        ax_surf.set_xlabel("X")
        ax_surf.set_ylabel("Y")
        ax_surf.set_zlabel("Energy (J)")
        ax_surf.set_title(f"Local Energy 3D (Step={frame}, T={T_now:.1f}K)", fontsize=12)
        ax_surf.set_xlim(0, Nsize)
        ax_surf.set_ylim(0, Nsize)
        ax_surf.set_zlim(0, ZMAX_FIXED)
        ax_surf.grid(False)
        ax_surf.zaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax_surf.zaxis.get_major_formatter().set_useOffset(False)
        surf_new = ax_surf.plot_surface(Xg, Yg, matE, cmap=COOLWARM_CUSTOM, linewidth=0, antialiased=False)

        # 2D contour 갱신
        ax_cont.clear()
        ax_cont.axis('off')
        ax_cont.set_title(f"Local Energy Contour (Step={frame}, T={T_now:.1f}K)", fontsize=12)
        cont_im_new = ax_cont.imshow(matE, origin='upper', cmap=COOLWARM_CUSTOM,
                                     vmin=0, vmax=ZMAX_FIXED)
        ax_cont.contour(matE, levels=10, colors='k', alpha=0.7)
        # 컬러바 범위 고정
        cont_im_new.set_clim(0, ZMAX_FIXED)
        cbar.update_normal(cont_im_new)

        # NEB( M->O->T ) 업데이트 (m_site가 없으면 생략)
        if m_site is not None:
            (mx,my) = m_site
            # E(M), E(O), E(T)
            E_M = local_energy_of_site(spins, mx, my, 2, Nsize, T_now, J)
            E_O = local_energy_of_site(spins, mx, my, 0, Nsize, T_now, J)  # up=0
            E_T = local_energy_of_site(spins, mx, my, 3, Nsize, T_now, J)  # T=3

            # Barrier
            B_MO = get_transition_barrier(2,0,T_now,J)  # M->O
            B_OT = get_transition_barrier(0,3,T_now,J)  # O->T

            # y = [E(M), E(M)+B_MO, E(O), E(O)+B_OT, E(T)]
            y_neb = [
                E_M,
                E_M + B_MO,
                E_O,
                E_O + B_OT,
                E_T
            ]
            line_neb.set_data(x_neb, y_neb)
            # y축 자동 업데이트 or 고정
            ymax = max(y_neb)*1.2
            if ymax<1e-30: ymax = 1.0
            if ymax< ZMAX_FIXED*1.2:
                ymax = ZMAX_FIXED*1.2
            ax_neb.set_ylim(0, ymax)
        else:
            # 만약 찾지 못했다면?
            line_neb.set_data([],[])

        return (im_heat, surf_new, cont_im_new, line_neb)

    anim_4 = FuncAnimation(fig_4, update_anim, init_func=init_anim,
                           frames= total_steps+1, interval=300, blit=False, repeat=False)

    plt.show()

    # 시뮬레이션 종료 후, 특정 스텝 PDF 저장
    print("\n시뮬레이션 완료. 결과 PDF로 내보낼 Step 번호들(띄어쓰기로) 입력 (예: 30 60):")
    steps_in = input("Steps: ").strip()
    if not steps_in:
        return
    try:
        step_vals = [int(s) for s in steps_in.split()]
        pdf_out = "Realtime_3D_energy_profile.pdf"
        with PdfPages(pdf_out) as pdf:
            for stv in step_vals:
                if 0<=stv<len(spin_snapshots):
                    snap = spin_snapshots[stv]
                    matE = energy_mats[stv]
                    T_here= temp_list[stv]

                    # Heatmap
                    fig_h = plt.figure(figsize=PDF_FIG_SIZE)
                    plt.imshow(snap, cmap=BFS_CMAP, vmin=0, vmax=q-1)
                    plt.title(f"Step={stv}, T={T_here:.1f}K Heatmap", fontsize=PDF_LABEL_FONT_SIZE)
                    plt.axis('off')
                    pdf.savefig(fig_h, bbox_inches='tight')
                    plt.close(fig_h)

                    # 3D surface
                    fig_s = plt.figure(figsize=(6,5))
                    ax_s = fig_s.add_subplot(111, projection='3d')
                    ax_s.plot_surface(Xg, Yg, matE, cmap=COOLWARM_CUSTOM, linewidth=0, antialiased=False)
                    ax_s.set_title(f"Local Energy 3D (Step={stv}, T={T_here:.1f}K)", fontsize=12)
                    ax_s.set_xlabel("X")
                    ax_s.set_ylabel("Y")
                    ax_s.set_zlabel("Energy (J)")
                    ax_s.set_xlim(0, Nsize)
                    ax_s.set_ylim(0, Nsize)
                    ax_s.set_zlim(0, ZMAX_FIXED)
                    ax_s.grid(False)
                    ax_s.zaxis.set_major_formatter(ScalarFormatter(useMathText=False))
                    ax_s.zaxis.get_major_formatter().set_useOffset(False)
                    pdf.savefig(fig_s, bbox_inches='tight')
                    plt.close(fig_s)

                    # 2D contour
                    fig_c = plt.figure(figsize=PDF_FIG_SIZE)
                    ax_c = fig_c.add_subplot(111)
                    ax_c.set_title(f"Local Energy Contour (Step={stv}, T={T_here:.1f}K)", fontsize=PDF_LABEL_FONT_SIZE)
                    ax_c.axis('off')
                    c_im = ax_c.imshow(matE, origin='upper', cmap=COOLWARM_CUSTOM, vmin=0, vmax=ZMAX_FIXED)
                    ax_c.contour(matE, levels=10, colors='k', alpha=0.7)
                    cbar_c = plt.colorbar(c_im, ax=ax_c, shrink=0.8)
                    cbar_c.set_label("Local Energy (J)", fontsize=PDF_LABEL_FONT_SIZE)
                    pdf.savefig(fig_c, bbox_inches='tight')
                    plt.close(fig_c)

                    # NEB (한 번 더 계산)
                    if m_site is not None:
                        (mx,my) = m_site
                        E_M = local_energy_of_site(snap, mx, my, 2, Nsize, T_here, J)
                        E_O = local_energy_of_site(snap, mx, my, 0, Nsize, T_here, J)
                        E_T = local_energy_of_site(snap, mx, my, 3, Nsize, T_here, J)
                        B_MO = get_transition_barrier(2,0,T_here,J)
                        B_OT = get_transition_barrier(0,3,T_here,J)
                        y_neb = [
                            E_M,
                            E_M + B_MO,
                            E_O,
                            E_O + B_OT,
                            E_T
                        ]
                        fig_nb = plt.figure(figsize=PDF_FIG_SIZE)
                        ax_nb = fig_nb.add_subplot(111)
                        ax_nb.set_title(f"NEB(M->O->T) Step={stv}, T={T_here:.1f}K", fontsize=12)
                        ax_nb.set_xlabel("Reaction Coord", fontsize=LABEL_FONT_SIZE)
                        ax_nb.set_ylabel("Energy (J)", fontsize=LABEL_FONT_SIZE)
                        x_neb_vals = [0,0.5,1,1.5,2]
                        ax_nb.plot(x_neb_vals, y_neb, '-o', color='purple')
                        ax_nb.set_xlim(-0.1,2.1)
                        ymax_local = max(y_neb)*1.2
                        if ymax_local< ZMAX_FIXED*1.2:
                            ymax_local= ZMAX_FIXED*1.2
                        ax_nb.set_ylim(0, ymax_local)
                        ax_nb.grid(True, linestyle=':')
                        pdf.savefig(fig_nb, bbox_inches='tight')
                        plt.close(fig_nb)

            print(f"PDF 저장 완료: {pdf_out}")
    except:
        pass

################################################################################
# [K] 메인
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
    select_material()

    print("\nSelect mode:")
    print("1) Slider Mode (real-time)")
    print("2) Temperature Profile Mode (real-time + offline)")
    print("3) Multi-h & Multi-size Tc 계산 모드")
    print("4) Real-time 3D energy profile Mode (+ 2D contour, coolwarm cmap, NEB)")
    mode= input("Mode? ")
    if mode not in ['1','2','3','4']:
        print("Invalid. Exiting.")
        sys.exit()

    if mode=='1':
        # 기존 Slider 모드
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

        im= ax_spin.imshow(spins, cmap=BFS_CMAP, vmin=0, vmax=3)
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

            accepted_time_sum= 0.0
            for _ in range(N_for_analysis*N_for_analysis):
                x= np.random.randint(N_for_analysis)
                y= np.random.randint(N_for_analysis)
                accepted, used_barrier= attempt_flip(spins,x,y,N_for_analysis,T_now,J)
                if accepted and used_barrier is not None:
                    rate= nu0*np.exp(-used_barrier/(k_B*T_now)) if T_now>1e-9 else 0.0
                    if rate>1e-30:
                        accepted_time_sum += (1.0/rate)

            dt= accepted_time_sum
            current_time+= dt* TIME_SCALE

            E_now= total_energy(spins,N_for_analysis,T_now,J)
            P_now= polarization(spins)
            ph_now= phase_fractions(spins)

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
        # Temperature Profile Mode (real-time + offline)
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

        im= ax_spin.imshow(spins, cmap=BFS_CMAP, vmin=0, vmax=q-1)
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

            accepted_time_sum=0.0
            for _ in range(N_for_analysis*N_for_analysis):
                x= np.random.randint(N_for_analysis)
                y= np.random.randint(N_for_analysis)
                accepted, used_barrier= attempt_flip(spins,x,y,N_for_analysis,T_now,J)
                if accepted and used_barrier is not None:
                    rate= nu0*np.exp(-used_barrier/(k_B*T_now)) if T_now>1e-9 else 0.0
                    if rate>1e-30:
                        accepted_time_sum+= 1.0/rate

            dt= accepted_time_sum
            current_time+= dt * TIME_SCALE

            E_now= total_energy(spins,N_for_analysis,T_now,J)
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

    else:
        # mode=='4': 수정된 3D 에너지 + 2D Contour + NEB
        mode_4_realtime_energy_profile()

    print("All done.")
    logging.info("Simulation ended.")

if __name__=="__main__":
    main()
