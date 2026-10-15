import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import fluidsynth
import time
from pathlib import Path

# ==========================================
# 1. 人耳心理声学核函数 R(f1, f2) - Plomp-Levelt 模型
# ==========================================
def plomp_levelt_raw(f1, f2):
    """
    计算两个纯音 (f1, f2) 之间的粗糙度 (Roughness/Dissonance)
    """
    f_min = np.minimum(f1, f2)
    f_diff = np.abs(f1 - f2)
    
    # 临界频带 (Critical Bandwidth) 的近似表达
    cbw = 24.7 * (4.37 * f_min / 1000.0 + 1.0)
    
    # 相对频率差 x
    x = f_diff / cbw
    
    # Plomp-Levelt 拟合曲线: a * exp(-b*x) - a * exp(-c*x)
    # 当 x ≈ 0.25 时达到刺耳峰值
    a, b, c = 3.5, 3.5, 5.75
    r = a * (np.exp(-b * x) - np.exp(-c * x))
    r[x < 0] = 0
    return np.maximum(r, 0)


# ==========================================
# 2. 从 SF2 文件渲染音符并提取频谱 S(f_base, f)
# ==========================================
class SF2SpectrumExtractor:
    def __init__(self, sf2_path, sample_rate=44100):
        self.sf2_path = sf2_path
        self.sr = sample_rate
        self.fs = fluidsynth.Synth(samplerate=int(sample_rate))
        self.sfid = self.fs.sfload(self.sf2_path)
        self.fs.program_select(0, self.sfid, 0, 0)

    def get_spectrum(self, midi_note, num_samples=22050):
        """
        触发特定 MIDI 音符，提取稳态频谱
        """
        self.fs.noteon(0, midi_note, 100)
        # 预热一小段时间避开初始瞬态噪声
        _ = self.fs.get_samples(4410) 
        # 提取稳态波形
        samples = self.fs.get_samples(num_samples)
        self.fs.noteoff(0, midi_note)
        
        # 声道合并 (Stereo -> Mono)
        samples_mono = samples[0::2].astype(np.float64) / 32768.0
        
        # 加窗并进行 FFT
        window = np.hanning(len(samples_mono))
        fft_data = np.abs(np.fft.rfft(samples_mono * window))
        freqs = np.fft.rfftfreq(len(samples_mono), 1.0 / self.sr)
        
        # 归一化幅度谱
        max_val = np.max(fft_data)
        if max_val > 0:
            fft_data /= max_val
            
        return freqs, fft_data


# ==========================================
# 3. 计算双音不和谐度 D(fA, fB)
# ==========================================
def calculate_dissonance_matrix(extractor, midi_range=range(36, 85), freq_max=10000):
    """
    遍历 MIDI 音符范围，计算二维不和谐度矩阵 D[A, B]
    midi_range: 默认 C2 (36) 到 C6 (84)
    """
    notes = list(midi_range)
    N = len(notes)
    
    # 预先提取所有音符的频谱，避免重复计算
    print("正在从 SF2 文件提取各音高频谱...")
    spectra = []
    for note in notes:
        freqs, spec = extractor.get_spectrum(note)
        # 截断高频，减少不必要的计算
        valid_idx = freqs <= freq_max
        # 降采样/阈值化去噪，保留有能量的频点
        f_active = freqs[valid_idx]
        s_active = spec[valid_idx]
        mask = s_active > 0.01 # 忽略弱于 -40dB 的杂音
        
        spectra.append((f_active[mask], s_active[mask]))

    D = np.zeros((N, N))
    print("正在计算双音不和谐度矩阵 D(f_A, f_B)...")
    start_time = time.time()
    
    # 双重循环计算两音之间的离散互干涉 (连续积分的离散采样)
    for i in range(N):
        fA_vec, sA_vec = spectra[i]
        for j in range(i, N):
            fB_vec, sB_vec = spectra[j]
            
            # 构建两两频点的网格
            f1_mesh, f2_mesh = np.meshgrid(fA_vec, fB_vec, indexing='ij')
            s1_mesh, s2_mesh = np.meshgrid(sA_vec, sB_vec, indexing='ij')
            
            # 计算有效能量: min(S1, S2)
            energy_mesh = np.minimum(s1_mesh, s2_mesh)
            
            # 计算粗糙核 R(f1, f2)
            r_mesh = plomp_levelt_raw(f1_mesh, f2_mesh)
            
            # 全频段积分求和
            dissonance = np.sum(energy_mesh * r_mesh)
            
            D[i, j] = dissonance
            D[j, i] = dissonance # 对称矩阵

    print(f"计算完成，耗时: {time.time() - start_time:.2f} 秒")
    return notes, D


# ==========================================
# 4. 绘制热力图 (Heatmap)
# ==========================================
def plot_dissonance_heatmap(notes, D):
    plt.figure(figsize=(10, 8))
    
    # 将 MIDI 转换为音名 (例如 60 -> C4)
    note_names = []
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for n in notes:
        name = f"{names[n % 12]}{n // 12 - 1}"
        note_names.append(name)
        
    # 绘制热力图
    # 使用 imshow，Origin 放在左下角
    im = plt.imshow(D, cmap='magma', origin='lower', aspect='auto')
    
    plt.colorbar(im, label='Sensory Dissonance D(f_A, f_B)')
    
    # 设置坐标轴标签 (每隔几个音标出一个)
    step = 3 if len(notes) > 30 else 1
    plt.xticks(ticks=np.arange(0, len(notes), step), labels=note_names[::step], rotation=45)
    plt.yticks(ticks=np.arange(0, len(notes), step), labels=note_names[::step])
    
    plt.xlabel('Base Note A ($f_A$)')
    plt.ylabel('Base Note B ($f_B$)')
    plt.title('Piano SF2 Sensory Dissonance Heatmap $D(f_A, f_B)$')
    
    plt.tight_layout()
    output_path = Path(__file__).with_name("dissonance_heatmap.png")
    plt.savefig(output_path, dpi=150)

    # Headless environments use the non-interactive Agg backend.
    if plt.get_backend().lower() == "agg":
        print(f"热力图已保存至: {output_path}")
    else:
        plt.show()

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 替换为你实际的 sf2 文件路径
    SF2_FILE = "soundfonts/SRX_Acoustic_Bass.sf2" 
    
    try:
        extractor = SF2SpectrumExtractor(SF2_FILE)
        # 计算 C2 (36) 到 C6 (84) 之间的音高不和谐度
        notes, D_matrix = calculate_dissonance_matrix(extractor, midi_range=range(36, 85))
        plot_dissonance_heatmap(notes, D_matrix)
    except Exception as e:
        print(f"运行出错 (可能是缺少 SF2 文件或 FluidSynth 环境): {e}")