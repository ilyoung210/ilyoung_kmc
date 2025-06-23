# ilyoung_kmc
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
from collections import deque, defaultdict
import random

# 3D 플롯에 필요
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as m3d

# Voronoi 등에 필요한 라이브러리
from scipy.spatial import cKDTree
from skimage.measure import find_contours

# 추가: Z축 formatting
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors

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
FIG_SIZE = (8.0, 6.0)      # 실시간 모드 plot창 기본 크기 (애니메이션 표시용)
TICK_FONT_SIZE = 14
LABEL_FONT_SIZE = 16
LEGEND_FONT_SIZE = 14

COLOR_LIST_NO_BLUE = ['g','r','m','c','y','k','orange','purple']
TEMP_COLOR  = 'b'

################################################################################
# [B] Potts 4상 (Up=0, Down=1, Mono=2, Tetra=3) 색상
################################################################################
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

# BFS_CMAP: 스핀 상(Up/Down/Mono/Tetra) 표시용
BFS_CMAP = mcolors.ListedColormap(
    [COLOR_UP, COLOR_DOWN, COLOR_MONO, COLOR_TETRA],
    name='bfs_map'
)

################################################################################
# (중요) 2D/3D 에너지 플롯에 사용할 컬러맵을 "jet"으로 변경
################################################################################
ENERGY_CMAP = 'jet'  # 'jet'은 파랑~초록~노랑~빨강 등 다채로운 색을 나타냄

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

# (추가) Joule -> meV 변환 계수
J_to_meV = 1.0 / (1.602176634e-22)  # 약 6.241509e18

################################################################################
# [C-1] HfO2, HZO, ZrO2, SiHfO2 파라미터 + 보간 함수
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

################################################################################
# [C-2] barrier CSV 초기화 함수 (PermissionError 예외 처리)
################################################################################
def initialize_barrier_file(filename='barrier_values.csv'):
    """
    barrier_values.csv 파일을 새로 쓰기 시도. 
    PermissionError 시, 경고 메시지 후 건너뜀.
    """
    import csv
    try:
        with open(filename,'w',newline='') as f:
            w=csv.writer(f)
            w.writerow(["T(K)","x","y","old_s","new_s","E_b_fwd","E_b_bwd"])
    except PermissionError:
        print(f"[WARNING] Permission denied for creating {filename}. Barrier file creation skipped.")

################################################################################
# [H-x] 시뮬레이션 끝난 후, barrier_values.csv를 transition type/온도별로 요약
################################################################################
from collections import defaultdict

def postprocess_barrier_data(infile='barrier_values.csv', outfile='barrier_summary.csv'):
    if not os.path.exists(infile):
        print(f"[postprocess] {infile} 파일이 없습니다. 요약 불가.")
        return

    data_dict = defaultdict(list)

    try:
        with open(infile, 'r', newline='') as f:
            rd = csv.reader(f)
            header = next(rd)  # skip header
            for row in rd:
                if len(row)<7:
                    continue
                T_val   = float(row[0])
                x       = int(row[1])
                y       = int(row[2])
                old_s   = int(row[3])
                new_s   = int(row[4])
                e_fwd   = float(row[5])
                e_bwd   = float(row[6])
                data_dict[(old_s, new_s, T_val)].append( (e_fwd, e_bwd) )
    except PermissionError:
        print(f"[WARNING] Permission denied reading {infile}. Skipped postprocessing.")
        return

    summary_rows = []
    for key, val_list in data_dict.items():
        (old_s, new_s, T_val) = key
        arr_f = [x[0] for x in val_list]
        arr_b = [x[1] for x in val_list]
        avg_f = np.mean(arr_f)
        avg_b = np.mean(arr_b)
        ccount= len(val_list)
        tr_str= f"{old_s}->{new_s}"
        summary_rows.append( (tr_str, T_val, avg_f, avg_b, ccount) )

    summary_rows.sort(key=lambda x: (x[0], x[1]))

    try:
        with open(outfile, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["Transition","T(K)","avg_fwd","avg_bwd","count"])
            for row in summary_rows:
                w.writerow(row)
        print(f"[postprocess] barrier 요약 파일 생성 완료: {outfile}")
    except PermissionError:
        print(f"[WARNING] Permission denied writing {outfile}. Skipped summary.")

################################################################################
# [C-3] 물성 선택 함수
################################################################################
def select_material():
    print("어떤 물질/도핑으로 시작하시겠습니까?")
    print("1) Zr-doped HfO2")
    print("2) Si-doped HfO2")
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
    # 절대 bulk 에너지 (site 단위)
    # E_p, E_m, E_t 는 이미 material-dependent 절대값 (J 단위)이고,
    # - T*a_? + sigma_?/(h/5) 항도 그대로 포함 → 최종 절대값
    if is_Up(s) or is_Down(s):
        return E_p - T*a_1 + sigma_1/(h/5)
    elif is_M(s):
        return E_m - T*a_2 + sigma_2/(h/5)
    elif is_T(s):
        return E_t - T*a_3 + sigma_3/(h/5)
    return 0.0

def field_energy(s,J):
    # E_field * P_val(s)도 절대값(여기선 E_field가 외부에서 정해진 값)
    return -E_field * P_val(s)

def potts_interaction_energy(s1,s2,J):
    # s1,s2 둘 다 Up이면 -J, Up-Down이면 +J 등
    if (is_Up(s1) and is_Up(s2)) or (is_Down(s1) and is_Down(s2)):
        return -J
    elif (is_Up(s1) and is_Down(s2)) or (is_Down(s1) and is_Up(s2)):
        return +J
    return 0.0

def bulk_energy_components(s, T, h):
    """Return (bulk_base, helmholtz, surface) for single spin state."""
    if is_Up(s) or is_Down(s):
        base = E_p
        helm = -T * a_1
        surf = sigma_1 / (h/5)
    elif is_M(s):
        base = E_m
        helm = -T * a_2
        surf = sigma_2 / (h/5)
    elif is_T(s):
        base = E_t
        helm = -T * a_3
        surf = sigma_3 / (h/5)
    else:
        base = helm = surf = 0.0
    return base, helm, surf

def interaction_energy_details(spins, x, y, state, N, J):
    """Return potts and interface contributions for (x,y) with given state."""
    neigh = [((x-1)%N,y), ((x+1)%N,y), (x,(y-1)%N), (x,(y+1)%N)]
    potts_vals = []
    interface_vals = []
    for (xx, yy) in neigh:
        s2 = spins[xx, yy]
        potts_vals.append(potts_interaction_energy(state, s2, J))
        interface_vals.append(interface_energy(state, s2, J))
    return potts_vals, interface_vals

def interface_energy(s1,s2,J):
    # s1,s2가 다르면 interface_pairs에서 절대값을 불러옴
    if s1==s2:
        return 0.0
    return interface_pairs.get(frozenset({s1,s2}), 0.0)

def local_energy(spins,x,y,N,T,J):
    """
    (x,y) 한 사이트에 대해 bulk+field + 인접 4방향에 따른 potts+interface 합
    """
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
    """
    시스템 전체의 절대 에너지( bulk + field + potts + interface ) 합
    """
    E_tot=0.0
    Nsize= spins.shape[0]
    for x in range(Nsize):
        for y in range(Nsize):
            s= spins[x,y]
            # bulk+field는 site마다 1번
            E_tot += bulk_energy(s,J,T,h)
            E_tot += field_energy(s,J)

            # potts+interface는 (x+1, y), (x, y+1)만 계산하여 2배 중복 방지
            # 하지만 여기서는 그냥 아래서 한꺼번에 계산해도 되며,
            # 중복 피하려면 아래처럼 한정해서 계산:
    for x in range(Nsize):
        for y in range(Nsize):
            s= spins[x,y]
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

################################################################################
# [추가 함수 1] 전체 에너지를 (bulk, field, potts, interface)로 분해하여 반환
################################################################################
def total_energy_breakdown(spins, N, T, J):
    """
    시스템 전체 에너지를 절대값으로
    (bulk, field, potts, interface, total) 로 분해 계산.
    기존 기능을 보존하기 위해 제공.
    """
    sum_bulk = 0.0
    sum_field = 0.0
    sum_potts = 0.0
    sum_interface = 0.0

    # site별 bulk+field
    for x in range(N):
        for y in range(N):
            s = spins[x,y]
            sum_bulk  += bulk_energy(s,J,T,h)
            sum_field += field_energy(s,J)

    # potts+interface: 중복 방지 위해 (x+1,y), (x,y+1)에 대해서만 더함
    for x in range(N):
        for y in range(N):
            s  = spins[x,y]
            sr = spins[x,(y+1)%N]
            sd = spins[(x+1)%N,y]
            sum_potts     += potts_interaction_energy(s,sr,J)
            sum_potts     += potts_interaction_energy(s,sd,J)
            sum_interface += interface_energy(s,sr,J)
            sum_interface += interface_energy(s,sd,J)

    E_tot = sum_bulk + sum_field + sum_potts + sum_interface
    return sum_bulk, sum_field, sum_potts, sum_interface, E_tot

################################################################################
# [추가 함수 1b] 세부 에너지 분해 (bulk/helmholtz/surface/field/potts/interface)
################################################################################
def total_energy_breakdown_extended(spins, N, T, J):
    """
    bulk 항(E_p, E_m, E_t)을 분리하여 Helmholtz(-T*a_i)와
    surface(sigma_i/(h/5))를 각각 독립적으로 합산.
    반환값: (bulk_base, helmholtz, surface, field, potts, interface, total)
    """
    sum_base = 0.0
    sum_helm = 0.0
    sum_surface = 0.0
    sum_field = 0.0
    sum_potts = 0.0
    sum_interface = 0.0

    for x in range(N):
        for y in range(N):
            s = spins[x, y]
            if is_Up(s) or is_Down(s):
                sum_base += E_p
                sum_helm += -T * a_1
                sum_surface += sigma_1 / (h/5)
            elif is_M(s):
                sum_base += E_m
                sum_helm += -T * a_2
                sum_surface += sigma_2 / (h/5)
            elif is_T(s):
                sum_base += E_t
                sum_helm += -T * a_3
                sum_surface += sigma_3 / (h/5)

            sum_field += field_energy(s,J)

    for x in range(N):
        for y in range(N):
            s = spins[x,y]
            sr = spins[x,(y+1)%N]
            sd = spins[(x+1)%N,y]
            sum_potts     += potts_interaction_energy(s,sr,J)
            sum_potts     += potts_interaction_energy(s,sd,J)
            sum_interface += interface_energy(s,sr,J)
            sum_interface += interface_energy(s,sd,J)

    E_tot = (sum_base + sum_helm + sum_surface +
             sum_field + sum_potts + sum_interface)
    return (sum_base, sum_helm, sum_surface,
            sum_field, sum_potts, sum_interface, E_tot)

################################################################################
# [추가 함수 2] NEB 에너지 항목별 CSV 기록 위한 함수
################################################################################
def initialize_neb_detail_file(filename='neb_energy_details.csv'):
    """
    NEB 에너지 항목별 로그 파일을 생성한다.
    bulk 항(E_p 등)을 분리하여 Helmholtz, surface 에너지까지
    별도 컬럼으로 기록한다.
    """
    try:
        with open(filename,'w',newline='') as f:
            w=csv.writer(f)
            w.writerow([
                "step","neb_index","x","y","T(K)","Transition",
                "BulkBase_from","Helmholtz_from","Surface_from",
                "Field_from","Potts_from","Interface_from","E_from",
                "BulkBase_to","Helmholtz_to","Surface_to",
                "Field_to","Potts_to","Interface_to","E_to","E_diff",
                "BulkTerm_from","FieldTerm_from","PottsTerms_from","InterfaceTerms_from",
                "BulkTerm_to","FieldTerm_to","PottsTerms_to","InterfaceTerms_to"
            ])
    except PermissionError:
        print(f"[WARNING] Permission denied for creating {filename}. NEB detail file creation skipped.")

def log_neb_energy_details(step, i, x, y, T_val,
                           from_label, b0_f, h_f, s_f, f_f, p_f, int_f, E_f,
                           to_label,   b0_t, h_t, s_t, f_t, p_t, int_t, E_t,
                           bulk_term_f, field_term_f, potts_terms_f, int_terms_f,
                           bulk_term_t, field_term_t, potts_terms_t, int_terms_t,
                           filename='neb_energy_details.csv'):
    """
    NEB 상황에서 from->to (예: M->T) 전이 시
    각 항목별 절대 에너지( bulk base, Helmholtz, surface, field,
    potts, interface, total )를 기록한다.
    E_diff = E_to - E_from
    """
    try:
        with open(filename, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                step, i, x, y, T_val, f"{from_label}->{to_label}",
                b0_f, h_f, s_f, f_f, p_f, int_f, E_f,
                b0_t, h_t, s_t, f_t, p_t, int_t, E_t,
                (E_t - E_f),
                f"{bulk_term_f[0]:.3e},{bulk_term_f[1]:.3e},{bulk_term_f[2]:.3e}",
                f"{field_term_f:.3e}",
                ';'.join(f"{v:.3e}" for v in potts_terms_f),
                ';'.join(f"{v:.3e}" for v in int_terms_f),
                f"{bulk_term_t[0]:.3e},{bulk_term_t[1]:.3e},{bulk_term_t[2]:.3e}",
                f"{field_term_t:.3e}",
                ';'.join(f"{v:.3e}" for v in potts_terms_t),
                ';'.join(f"{v:.3e}" for v in int_terms_t)
            ])
    except PermissionError:
        print(f"[WARNING] Could not write to {filename} (Permission denied). Skipped logging NEB energy details.")

################################################################################
# barrier & Metropolis 시도
################################################################################
def get_transition_barrier(old_state,new_state,T,J):
    # 구간별 장벽값(단위 J), 미리 정의
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

def write_dynamic_barrier_to_csv(T, x, y, old_s, new_s, e_b_fwd, e_b_bwd, filename='barrier_values.csv'):
    try:
        with open(filename, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([T, x, y, old_s, new_s, e_b_fwd, e_b_bwd])
    except PermissionError:
        print(f"[WARNING] Could not write to {filename} (Permission denied). Skipped logging barrier.")

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

    new_s= np.random.choice(poss)
    E_old= local_energy(spins, x,y, N, T, J)
    spins[x,y] = new_s
    E_new= local_energy(spins, x,y, N, T, J)
    spins[x,y] = old_s

    dE= E_new - E_old

    e_b0_fwd= get_transition_barrier(old_s, new_s, T, J)
    e_b0_bwd= get_transition_barrier(new_s, old_s, T, J)

    e_b_fwd = e_b0_fwd + max(0, dE)
    e_b_bwd = e_b0_bwd + max(0, -dE)

    # 필요시 장벽 로그
    # write_dynamic_barrier_to_csv(T, x, y, old_s, new_s, e_b_fwd, e_b_bwd)

    # Metropolis-like
    accepted = False
    if dE < 0:
        if abs(dE) >= e_b_fwd:
            spins[x,y]= new_s
            accepted = True
        else:
            prob= np.exp(-dE/(k_B*T)) if T>1e-9 else 0.0
            if np.random.rand()< prob:
                spins[x,y]= new_s
                accepted = True
    else:
        prob= np.exp(-dE/(k_B*T)) if T>1e-9 else 0.0
        if np.random.rand()< prob:
            spins[x,y]= new_s
            accepted = True

    if accepted:
        return (True, e_b_fwd)
    else:
        return (False, None)

################################################################################
# [E] BFS + 결정립 분석 (필요시)
################################################################################
def compute_bfs_radius_all(spins):
    return (0,0,0,0,0)

################################################################################
# [H] Temperature Profile 파싱/함수
################################################################################
def parse_temperature_profile():
    """
    사용자 입력을 통해 온도 프로파일(구간별 시작T, 끝T, 스텝수)을 받아
    리스트로 반환
    """
    print("\n온도 프로파일을 입력하세요.")
    print("(예) 300 1000 30 => 300K에서 1000K까지 30스텝 선형증가")
    print("구간 여러 줄 입력 후, 빈 줄로 종료.\n예:\n 300 1000 30\n 1000 500 10\n (빈 줄)\n")

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
    """
    현재 스텝 stp에 대하여 세그먼트별 (startT->endT) 구간에서
    해당 스텝 범위이면 보간하여 T를 리턴
    """
    cum=0
    for (startT,endT,stepc) in segs:
        if stp< cum+ stepc:
            local_s= stp- cum
            dT= (endT - startT)/stepc
            return max(startT + local_s*dT, 0.0)
        cum+= stepc
    return max(segs[-1][1], 0.0)

def get_minmax_temperature_from_segments(segments):
    temps = []
    for (sT, eT, sc) in segments:
        temps.append(sT)
        temps.append(eT)
    return (min(temps), max(temps))

################################################################################
# NEB Marker & Plot
################################################################################

def find_m_site_with_Tneighbors(spins, T_needed, M_needed):
    N = spins.shape[0]
    for x in range(N):
        for y in range(N):
            if not is_M(spins[x,y]):
                continue
            nbs = [
                spins[(x-1)%N,y],
                spins[(x+1)%N,y],
                spins[x,(y-1)%N],
                spins[x,(y+1)%N]
            ]
            t_count = sum(1 for nb in nbs if is_T(nb))
            m_count = sum(1 for nb in nbs if is_M(nb))
            if t_count == T_needed and m_count == M_needed and (t_count + m_count == 4):
                return (x,y)
    return None

def find_m_site_with_Oneighbors(spins, O_needed, M_needed):
    N = spins.shape[0]
    for x in range(N):
        for y in range(N):
            if not is_M(spins[x,y]):
                continue
            nbs = [
                spins[(x-1)%N,y],
                spins[(x+1)%N,y],
                spins[x,(y-1)%N],
                spins[x,(y+1)%N]
            ]
            o_count = sum(1 for nb in nbs if is_Up(nb))
            m_count = sum(1 for nb in nbs if is_M(nb))
            if o_count == O_needed and m_count == M_needed and (o_count + m_count == 4):
                return (x,y)
    return None

################################################################################
# 메인 NEB 모드
################################################################################
def mode_4_realtime_energy_profile():
    segs= parse_temperature_profile()
    total_steps= sum(s[2] for s in segs)

    global_minT, global_maxT = get_minmax_temperature_from_segments(segs)
    Nsize = N_for_analysis
    spins = np.random.randint(q, size=(Nsize, Nsize))

    step_list= []
    temp_list= []
    spin_snapshots= []
    energy_mats= []
    current_time= 0.0
    ZMAX_FIXED = 2.0e-19

    # 2행 x 4열
    fig_4 = plt.figure(figsize=(16,8))
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    gs = fig_4.add_gridspec(2,4)

    # 상단(0행)
    ax_heat   = fig_4.add_subplot(gs[0,0])
    ax_surf   = fig_4.add_subplot(gs[0,1], projection='3d')
    ax_cont   = fig_4.add_subplot(gs[0,2])
    ax_marker = fig_4.add_subplot(gs[0,3])  # Marker

    # 하단(1행) -> NEB #1 diff, NEB #2 diff, NEB #1 abs, NEB #2 abs
    ax_neb1_diff= fig_4.add_subplot(gs[1,0])
    ax_neb2_diff= fig_4.add_subplot(gs[1,1])
    ax_neb1_abs = fig_4.add_subplot(gs[1,2])
    ax_neb2_abs = fig_4.add_subplot(gs[1,3])

    # Heatmap
    im_heat = ax_heat.imshow(spins, cmap=BFS_CMAP, vmin=0, vmax=q-1)
    ax_heat.axis('off')
    ax_heat.set_title("Heatmap (Step=0)", fontsize=12)

    # 3D Surface
    X = np.arange(Nsize)
    Y = np.arange(Nsize)
    Xg, Yg = np.meshgrid(X, Y)
    Z0 = np.zeros((Nsize,Nsize))
    surf = ax_surf.plot_surface(Xg, Yg, Z0, cmap=ENERGY_CMAP,
                                linewidth=0, antialiased=False)
    ax_surf.set_title("Local Energy 3D (Step=0)", fontsize=12)
    ax_surf.set_xlabel("X")
    ax_surf.set_ylabel("Y")
    ax_surf.set_zlabel("Energy (J)")
    ax_surf.set_xlim(0, Nsize)
    ax_surf.set_ylim(0, Nsize)
    ax_surf.set_zlim(0, ZMAX_FIXED)
    ax_surf.grid(False)
    ax_surf.zaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax_surf.zaxis.get_major_formatter().set_useOffset(False)

    # 2D Contour
    cont_im = ax_cont.imshow(Z0, origin='upper', cmap=ENERGY_CMAP,
                             vmin=0, vmax=ZMAX_FIXED)
    ax_cont.set_title("Local Energy Contour (Step=0)", fontsize=12)
    ax_cont.axis('off')
    ax_cont.contour(Z0, levels=10, colors='k', alpha=0.7)
    cbar = fig_4.colorbar(cont_im, ax=ax_cont, shrink=0.7, pad=0.02)
    cbar.set_label("Local Energy (J)", fontsize=LABEL_FONT_SIZE)

    # Marker Heatmap
    ax_marker.set_title("Case Marker", fontsize=12)
    ax_marker.axis('off')
    hm_marker_main = ax_marker.imshow(spins, cmap=BFS_CMAP, vmin=0, vmax=q-1)
    highlight_overlay = np.zeros((Nsize, Nsize, 4), dtype=float)
    hm_marker_overlay = ax_marker.imshow(highlight_overlay, interpolation='nearest')

    # 화살표 annotation 보관용
    arrow_nb1 = []
    arrow_nb2 = []

    # NEB #1 diff
    ax_neb1_diff.set_title("NEB#1 Diff (M→T, meV)", fontsize=12)
    ax_neb1_diff.set_xlabel("Reaction Coord", fontsize=LABEL_FONT_SIZE)
    ax_neb1_diff.set_ylabel("ΔE (meV)", fontsize=LABEL_FONT_SIZE)
    ax_neb1_diff.set_xlim(-0.1,1.1)
    ax_neb1_diff.set_ylim(0,1000)
    ax_neb1_diff.grid(True, linestyle=':')

    labels_nb1 = ["M4T0","M3T1","M2T2","M1T3","M0T4"]
    colors_nb1 = ['blue','cyan','dodgerblue','navy','purple']
    line_nb1_diff=[]
    for i in range(5):
        ln,= ax_neb1_diff.plot([],[],'-o', color=colors_nb1[i], label=labels_nb1[i])
        line_nb1_diff.append(ln)
    hd1, ld1 = ax_neb1_diff.get_legend_handles_labels()
    ax_neb1_diff.legend(hd1, ld1, loc='best', fontsize=9)

    # NEB #2 diff
    ax_neb2_diff.set_title("NEB#2 Diff (M→O, meV)", fontsize=12)
    ax_neb2_diff.set_xlabel("Reaction Coord", fontsize=LABEL_FONT_SIZE)
    ax_neb2_diff.set_ylabel("ΔE (meV)", fontsize=LABEL_FONT_SIZE)
    ax_neb2_diff.set_xlim(-0.1,1.1)
    ax_neb2_diff.set_ylim(0,1000)
    ax_neb2_diff.grid(True, linestyle=':')

    labels_nb2 = ["M4O0","M3O1","M2O2","M1O3","M0O4"]
    colors_nb2 = ['red','orange','magenta','brown','green']
    line_nb2_diff=[]
    for i in range(5):
        ln2,= ax_neb2_diff.plot([],[],'-o', color=colors_nb2[i], label=labels_nb2[i])
        line_nb2_diff.append(ln2)
    hd2, ld2 = ax_neb2_diff.get_legend_handles_labels()
    ax_neb2_diff.legend(hd2, ld2, loc='best', fontsize=9)

    # NEB #1 abs
    ax_neb1_abs.set_title("NEB#1 Abs (M→T, J)", fontsize=12)
    ax_neb1_abs.set_xlabel("Reaction Coord", fontsize=LABEL_FONT_SIZE)
    ax_neb1_abs.set_ylabel("Energy (J)", fontsize=LABEL_FONT_SIZE)
    ax_neb1_abs.set_xlim(-0.1,1.1)
    ax_neb1_abs.set_ylim(0, ZMAX_FIXED*1.2)
    ax_neb1_abs.grid(True, linestyle=':')
    line_nb1_abs=[]
    for i in range(5):
        la,= ax_neb1_abs.plot([],[],'-o',color=colors_nb1[i], label=labels_nb1[i])
        line_nb1_abs.append(la)
    ha1, la1 = ax_neb1_abs.get_legend_handles_labels()
    ax_neb1_abs.legend(ha1, la1, loc='best', fontsize=9)

    # NEB #2 abs
    ax_neb2_abs.set_title("NEB#2 Abs (M→O, J)", fontsize=12)
    ax_neb2_abs.set_xlabel("Reaction Coord", fontsize=LABEL_FONT_SIZE)
    ax_neb2_abs.set_ylabel("Energy (J)", fontsize=LABEL_FONT_SIZE)
    ax_neb2_abs.set_xlim(-0.1,1.1)
    ax_neb2_abs.set_ylim(0, ZMAX_FIXED*1.2)
    ax_neb2_abs.grid(True, linestyle=':')
    line_nb2_abs=[]
    for i in range(5):
        la2,= ax_neb2_abs.plot([],[],'-o',color=colors_nb2[i], label=labels_nb2[i])
        line_nb2_abs.append(la2)
    ha2, la2 = ax_neb2_abs.get_legend_handles_labels()
    ax_neb2_abs.legend(ha2, la2, loc='best', fontsize=9)

    def init_anim():
        return (im_heat, surf, cont_im, hm_marker_main, hm_marker_overlay)

    def update_anim(frame):
        # 매 프레임마다 화살표 지우기
        for ann in arrow_nb1:
            ann.remove()
        arrow_nb1.clear()
        for ann2 in arrow_nb2:
            ann2.remove()
        arrow_nb2.clear()

        if frame> total_steps:
            anim_4.event_source.stop()
            return (im_heat, surf, cont_im, hm_marker_main, hm_marker_overlay)

        T_now = T_func_profile(segs, frame)

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
        nonlocal current_time
        current_time += (accepted_time_sum * TIME_SCALE)

        step_list.append(frame)
        temp_list.append(T_now)

        # Heatmap
        im_heat.set_data(spins)
        ax_heat.set_title(f"Heatmap (Step={frame}, T={T_now:.1f}K)", fontsize=12)

        # Local energy mat
        matE = np.zeros((Nsize,Nsize))
        for xx in range(Nsize):
            for yy in range(Nsize):
                matE[xx,yy] = local_energy(spins, xx, yy, Nsize, T_now, J)
        energy_mats.append(matE.copy())
        spin_snapshots.append(spins.copy())

        # 3D Surf
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
        surf_new= ax_surf.plot_surface(Xg, Yg, matE, cmap=ENERGY_CMAP,
                                       linewidth=0, antialiased=False)

        # 2D Contour
        ax_cont.clear()
        ax_cont.axis('off')
        ax_cont.set_title(f"Local Energy Contour (Step={frame}, T={T_now:.1f}K)", fontsize=12)
        cont_im_new= ax_cont.imshow(matE, origin='upper', cmap=ENERGY_CMAP,
                                    vmin=0, vmax=ZMAX_FIXED)
        ax_cont.contour(matE, levels=10, colors='k', alpha=0.7)
        cont_im_new.set_clim(0, ZMAX_FIXED)
        cbar.update_normal(cont_im_new)

        # Marker Heatmap
        hm_marker_main.set_data(spins)
        new_highlight= np.zeros((Nsize,Nsize,4), dtype=float)

        # NEB #1 diff / abs
        all_nb1_diff_vals=[]
        all_nb1_abs_vals=[]
        for i in range(5):
            T_need = i
            M_need = 4 - i
            site = find_m_site_with_Tneighbors(spins, T_need, M_need)
            if site is not None:
                (mx,my)= site
                # M 상태(From)
                sp_copy= spins.copy()
                sp_copy[mx,my] = 2  # M
                b0M, hM, sM, fM, pM, iM, eM = total_energy_breakdown_extended(sp_copy,Nsize,T_now,J)

                # T 상태(To)
                sp_copy[mx,my] = 3  # T
                b0T, hT, sT, fT, pT, iT, eT = total_energy_breakdown_extended(sp_copy,Nsize,T_now,J)

                dE_J= eT - eM
                dE_meV= dE_J * J_to_meV

                # diff(meV)
                line_nb1_diff[i].set_data([0,1],[0,dE_meV])
                all_nb1_diff_vals.extend([0,dE_meV])

                # abs(J)
                line_nb1_abs[i].set_data([0,1],[eM,eT])
                all_nb1_abs_vals.extend([eM,eT])

                # CSV 기록(M -> T)
                bulk_f = bulk_energy_components(2, T_now, h)
                field_f = field_energy(2, J)
                potts_f, int_f_d = interaction_energy_details(spins, mx, my, 2, Nsize, J)

                bulk_t = bulk_energy_components(3, T_now, h)
                field_t = field_energy(3, J)
                potts_t, int_t_d = interaction_energy_details(spins, mx, my, 3, Nsize, J)

                log_neb_energy_details(
                    step=frame, i=i, x=mx, y=my, T_val=T_now,
                    from_label="M", b0_f=b0M, h_f=hM, s_f=sM, f_f=fM,
                    p_f=pM, int_f=iM, E_f=eM,
                    to_label="T",   b0_t=b0T, h_t=hT, s_t=sT, f_t=fT,
                    p_t=pT, int_t=iT, E_t=eT,
                    bulk_term_f=bulk_f, field_term_f=field_f,
                    potts_terms_f=potts_f, int_terms_f=int_f_d,
                    bulk_term_t=bulk_t, field_term_t=field_t,
                    potts_terms_t=potts_t, int_terms_t=int_t_d
                )

                # Marker color
                import matplotlib.colors as mcolors
                rgba1= mcolors.to_rgba(colors_nb1[i], alpha=0.7)
                new_highlight[mx,my] = rgba1

                # 화살표
                ann= ax_marker.annotate(
                    labels_nb1[i], xy=(my,mx),
                    xytext=(my+0.5,mx+0.5),
                    arrowprops=dict(arrowstyle='->', color=colors_nb1[i], lw=1.5),
                    color=colors_nb1[i],
                    fontsize=9
                )
                arrow_nb1.append(ann)
            else:
                line_nb1_diff[i].set_data([],[])
                line_nb1_abs[i].set_data([],[])

        # NEB #1 diff y-limit
        if all_nb1_diff_vals:
            mn_d1= min(all_nb1_diff_vals)
            mx_d1= max(all_nb1_diff_vals)
            if abs(mx_d1-mn_d1)<1e-30:
                ax_neb1_diff.set_ylim(mn_d1-1e-3,mx_d1+1e-3)
            else:
                pd1=0.05*(mx_d1 - mn_d1)
                ax_neb1_diff.set_ylim(mn_d1-pd1,mx_d1+pd1)
        else:
            ax_neb1_diff.set_ylim(0,1)

        # NEB #1 abs y-limit
        if all_nb1_abs_vals:
            mn_a1= min(all_nb1_abs_vals)
            mx_a1= max(all_nb1_abs_vals)
            if abs(mx_a1-mn_a1)<1e-30:
                ax_neb1_abs.set_ylim(mn_a1-1e-25,mx_a1+1e-25)
            else:
                pa1=0.05*(mx_a1 - mn_a1)
                ax_neb1_abs.set_ylim(mn_a1-pa1,mx_a1+pa1)
        else:
            ax_neb1_abs.set_ylim(0,1)

        # NEB #2 diff / abs
        all_nb2_diff_vals=[]
        all_nb2_abs_vals=[]
        for i in range(5):
            O_need = i
            M_need = 4 - i
            site= find_m_site_with_Oneighbors(spins,O_need,M_need)
            if site is not None:
                (mx,my)= site
                sp_copy= spins.copy()

                # M 상태(From)
                sp_copy[mx,my] = 2
                b0M2, hM2, sM2, fM2, pM2, iM2, eM2 = total_energy_breakdown_extended(sp_copy,Nsize,T_now,J)

                # O 상태(To, Up=0)
                sp_copy[mx,my] = 0
                b0O2, hO2, sO2, fO2, pO2, iO2, eO2 = total_energy_breakdown_extended(sp_copy,Nsize,T_now,J)

                dE_J2= eO2 - eM2
                dE_meV2= dE_J2 * J_to_meV

                # diff
                line_nb2_diff[i].set_data([0,1],[0,dE_meV2])
                all_nb2_diff_vals.extend([0,dE_meV2])

                # abs
                line_nb2_abs[i].set_data([0,1],[eM2,eO2])
                all_nb2_abs_vals.extend([eM2,eO2])

                # CSV 기록(M -> O)
                bulk_f2 = bulk_energy_components(2, T_now, h)
                field_f2 = field_energy(2, J)
                potts_f2, int_f2_d = interaction_energy_details(spins, mx, my, 2, Nsize, J)

                bulk_t2 = bulk_energy_components(0, T_now, h)
                field_t2 = field_energy(0, J)
                potts_t2, int_t2_d = interaction_energy_details(spins, mx, my, 0, Nsize, J)

                log_neb_energy_details(
                    step=frame, i=i, x=mx, y=my, T_val=T_now,
                    from_label="M", b0_f=b0M2, h_f=hM2, s_f=sM2, f_f=fM2,
                    p_f=pM2, int_f=iM2, E_f=eM2,
                    to_label="O",   b0_t=b0O2, h_t=hO2, s_t=sO2, f_t=fO2,
                    p_t=pO2, int_t=iO2, E_t=eO2,
                    bulk_term_f=bulk_f2, field_term_f=field_f2,
                    potts_terms_f=potts_f2, int_terms_f=int_f2_d,
                    bulk_term_t=bulk_t2, field_term_t=field_t2,
                    potts_terms_t=potts_t2, int_terms_t=int_t2_d
                )

                # Marker
                import matplotlib.colors as mcolors
                rgba2= mcolors.to_rgba(colors_nb2[i], alpha=0.7)
                new_highlight[mx,my] = rgba2

                ann2= ax_marker.annotate(
                    labels_nb2[i], xy=(my,mx),
                    xytext=(my+0.5,mx+0.5),
                    arrowprops=dict(arrowstyle='->', color=colors_nb2[i], lw=1.5),
                    color=colors_nb2[i],
                    fontsize=9
                )
                arrow_nb2.append(ann2)
            else:
                line_nb2_diff[i].set_data([],[])
                line_nb2_abs[i].set_data([],[])

        # NEB #2 diff y-limit
        if all_nb2_diff_vals:
            mn_d2= min(all_nb2_diff_vals)
            mx_d2= max(all_nb2_diff_vals)
            if abs(mx_d2-mn_d2)<1e-30:
                ax_neb2_diff.set_ylim(mn_d2-1e-3,mx_d2+1e-3)
            else:
                p2=0.05*(mx_d2 - mn_d2)
                ax_neb2_diff.set_ylim(mn_d2-p2,mx_d2+p2)
        else:
            ax_neb2_diff.set_ylim(0,1)

        # NEB #2 abs y-limit
        if all_nb2_abs_vals:
            mn_a2= min(all_nb2_abs_vals)
            mx_a2= max(all_nb2_abs_vals)
            if abs(mx_a2-mn_a2)<1e-30:
                ax_neb2_abs.set_ylim(mn_a2-1e-25,mx_a2+1e-25)
            else:
                pa2=0.05*(mx_a2 - mn_a2)
                ax_neb2_abs.set_ylim(mn_a2-pa2,mx_a2+pa2)
        else:
            ax_neb2_abs.set_ylim(0,1)

        # Marker overlay
        hm_marker_overlay.set_data(new_highlight)
        hm_marker_overlay.set_alpha(1.0)

        return (im_heat, surf_new, cont_im_new, hm_marker_main, hm_marker_overlay)

    anim_4 = FuncAnimation(
        fig_4, update_anim, init_func=init_anim,
        frames= total_steps+1, interval=300, blit=False, repeat=False
    )

    plt.show()

    # 시뮬레이션 종료 후 PDF 저장
    print("\n시뮬레이션 완료. PDF로 저장할 Step 번호(예: 30 60):")
    steps_in = input("Steps: ").strip()
    if not steps_in:
        return
    try:
        step_vals = [int(s) for s in steps_in.split()]
        pdf_out = "Realtime_3D_energy_profile.pdf"
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(pdf_out) as pdf:
            Xg_save, Yg_save = np.meshgrid(np.arange(Nsize), np.arange(Nsize))
            for stv in step_vals:
                if 0<=stv<len(spin_snapshots):
                    snap = spin_snapshots[stv]
                    matE = energy_mats[stv]
                    T_here= temp_list[stv] if stv<len(temp_list) else 0.0

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
                    ax_s.plot_surface(Xg_save, Yg_save, matE, cmap=ENERGY_CMAP,
                                      linewidth=0, antialiased=False)
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
                    c_im = ax_c.imshow(matE, origin='upper', cmap=ENERGY_CMAP,
                                       vmin=0, vmax=ZMAX_FIXED)
                    ax_c.contour(matE, levels=10, colors='k', alpha=0.7)
                    cbar_c = plt.colorbar(c_im, ax=ax_c, shrink=0.8)
                    cbar_c.set_label("Local Energy (J)", fontsize=PDF_LABEL_FONT_SIZE)
                    pdf.savefig(fig_c, bbox_inches='tight')
                    plt.close(fig_c)

            print(f"PDF 저장 완료: {pdf_out}")
    except:
        pass

################################################################################
# [K] 메인
################################################################################
def main():
    # barrier csv 초기화
    initialize_barrier_file()
    # NEB 상세 CSV 초기화
    initialize_neb_detail_file('neb_energy_details.csv')

    # 물질 선택(도핑 등)
    select_material()

    # NEB 모드 실행
    mode_4_realtime_energy_profile()

    # 모든 시뮬레이션 끝난 후 barrier_summary.csv 생성
    postprocess_barrier_data('barrier_values.csv','barrier_summary.csv')

    print("All done.")
    logging.info("Simulation ended.")

if __name__=="__main__":
    main()
