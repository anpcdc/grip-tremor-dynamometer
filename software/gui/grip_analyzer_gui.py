import sys
import serial
import numpy as np
from PySide6.QtWidgets import *
from PySide6.QtCore import QTimer, Qt, QThread, Signal
import pyqtgraph as pg
from scipy.signal import butter, filtfilt, welch, savgol_filter
from scipy.fft import fft, fftfreq


# ================= PROTOCOL TIMING =================
# DEMO_MODE = True shortens the rest periods to 5 s (used for live demonstrations).
# Set DEMO_MODE = False for the real protocol: 20 s rest after each MVC,
# 60 s rest after each submaximal hold.
DEMO_MODE = True
REST_AFTER_MVC = 5 if DEMO_MODE else 20
REST_AFTER_SUB = 5 if DEMO_MODE else 60


# ================= NORMATIVE DATA (Dodds et al., 2014) =================
# (age_min, age_max, mean_kgf, std_kgf) — requires calibrated force in kgf

NORMS = {
    "M": [
        (5, 9, 7.7, 2.9),
        (10, 14, 17.2, 4.1),
        (15, 19, 29.6, 5.6),
        (20, 24, 41.5, 7.3),
        (25, 29, 48.8, 8.7),
        (30, 34, 51.6, 9.6),
        (35, 39, 51.6, 10.1),
        (40, 44, 50.3, 10.3),
        (45, 49, 48.8, 10.3),
        (50, 54, 47.6, 10.1),
        (55, 59, 46.2, 9.8),
        (60, 64, 44.6, 9.2),
        (65, 69, 42.3, 8.6),
        (70, 74, 39.1, 8.1),
        (75, 79, 35.6, 7.6),
        (80, 84, 32.2, 7.3),
        (85, 89, 28.5, 7.0),
        (90, 999, 24.7, 6.8),
    ],
    "F": [
        (5, 9, 8.0, 3.1),
        (10, 14, 16.7, 3.8),
        (15, 19, 23.9, 4.5),
        (20, 24, 28.4, 5.1),
        (25, 29, 30.6, 5.6),
        (30, 34, 31.4, 6.0),
        (35, 39, 31.3, 6.2),
        (40, 44, 30.7, 6.3),
        (45, 49, 29.9, 6.4),
        (50, 54, 28.7, 6.4),
        (55, 59, 27.5, 6.4),
        (60, 64, 26.5, 6.2),
        (65, 69, 25.3, 6.0),
        (70, 74, 23.5, 5.7),
        (75, 79, 21.4, 5.4),
        (80, 84, 19.1, 5.1),
        (85, 89, 16.6, 4.7),
        (90, 999, 14.2, 4.4),
    ],
}


def compute_zscore(fmax_kgf, age, sex):
    """Z-score of HGS relative to normative data (Dodds et al., 2014).
    NOTE: fmax_kgf must be a calibrated value in kgf.
    """
    sex_key = "M" if sex.upper().startswith("M") else "F"
    for a_min, a_max, mu, sigma in NORMS[sex_key]:
        if a_min <= age <= a_max:
            return (fmax_kgf - mu) / sigma
    return None


def most_stable_window_dedup(force_unique, time_unique, window_sec=10):
    """Find the most stable 10-second window using only unique force readings.

    force_unique : 1-D array of deduplicated force values (one per real HX711 reading)
    time_unique  : corresponding timestamps (seconds)
    window_sec   : window length in seconds (default 10)

    Returns the start and end *timestamps* of the best window so the caller can
    slice both the deduplicated force array and the full-resolution acc array by time.
    """
    if len(force_unique) < 2:
        return time_unique[0], time_unique[-1]

    # Build index windows whose time span is closest to window_sec
    best_std = np.inf
    best_t0, best_t1 = time_unique[0], time_unique[-1]

    for i in range(len(time_unique)):
        t0 = time_unique[i]
        t1 = t0 + window_sec
        if t1 > time_unique[-1]:
            break
        mask = (time_unique >= t0) & (time_unique < t1)
        seg = force_unique[mask]
        if len(seg) < 2:
            continue
        s = float(np.std(seg))
        if s < best_std:
            best_std = s
            best_t0, best_t1 = t0, t1

    return best_t0, best_t1


def compute_tfr(
    f_stable_dedup,
    t_stable_dedup,
    a_stable_full,
    t_stable_full,
    band_low=4.0,
    band_high=12.0,
    window_sec=1.0,
):
    """Tremor-Force Ratio per 1-second non-overlapping windows.

    Uses deduplicated force (for Fmax per window) and full-resolution acc (for FFT).
    TFR_w = peak FFT amplitude of acc (band_low–band_high Hz) / max force in window.
    Returns (mean_tfr, std_tfr).
    """
    tfr_vals = []
    t_start = t_stable_full[0] if len(t_stable_full) > 0 else 0.0
    t_end = t_stable_full[-1] if len(t_stable_full) > 0 else 0.0

    t_win = t_start
    while t_win + window_sec <= t_end + 1e-9:
        t_win_end = t_win + window_sec

        # Force: deduplicated readings inside this 1-second window
        mask_f = (t_stable_dedup >= t_win) & (t_stable_dedup < t_win_end)
        f_win = f_stable_dedup[mask_f]
        if len(f_win) == 0:
            t_win += window_sec
            continue
        F_max_w = float(np.max(np.abs(f_win)))
        if F_max_w == 0:
            t_win += window_sec
            continue

        # Acc: full-resolution samples inside this 1-second window
        mask_a = (t_stable_full >= t_win) & (t_stable_full < t_win_end)
        a_win = a_stable_full[mask_a]
        if len(a_win) < 4:
            t_win += window_sec
            continue

        fs_win = len(a_win) / window_sec  # actual acc fs in this window
        N = len(a_win)
        window = np.hamming(N)
        yf = np.abs(fft(a_win * window))[: N // 2]
        xf = fftfreq(N, 1.0 / fs_win)[: N // 2]
        band = (xf >= band_low) & (xf <= band_high)
        if not np.any(band):
            t_win += window_sec
            continue
        a_peak_w = float(np.max(yf[band]))
        tfr_vals.append(a_peak_w / F_max_w)
        t_win += window_sec

    if tfr_vals:
        return float(np.mean(tfr_vals)), float(np.std(tfr_vals))
    return 0.0, 0.0


# ================= FAKE SERIAL =================


class FakeSerialThread(QThread):
    new_data = Signal(float, float, float, float, float)

    def __init__(self):
        super().__init__()
        self.running = True
        self.t = 0

    def run(self):
        dt = 20  # ms → 50 Hz
        phase = "idle"
        phase_time = 0
        max_force = 400  # arbitrary units (raw ADC proxy)

        while self.running:
            self.t += dt
            phase_time += dt / 1000.0

            if phase == "idle" and phase_time > 3:
                phase = "max"
                phase_time = 0
            elif phase == "max" and phase_time > 5:
                phase = "rest"
                phase_time = 0
            elif phase == "rest" and phase_time > 5:
                phase = "sub"
                phase_time = 0
            elif phase == "sub" and phase_time > 30:
                phase = "idle"
                phase_time = 0

            if phase == "max":
                force = max_force * (1 - np.exp(-phase_time * 2))
                force += 8 * np.random.normal(0, 1)
            elif phase == "sub":
                target = 0.3 * max_force
                force = target + 5 * np.sin(2 * np.pi * 0.3 * phase_time)
                force += 3 * np.random.normal(0, 1)
            else:
                force = 5 + np.random.normal(0, 1)

            # Accelerometer: tremor at ~10 Hz + noise
            t_s = self.t / 1000.0
            tremor = 0.5 * np.sin(2 * np.pi * 10 * t_s)
            ax = tremor + np.random.normal(0, 0.05)
            ay = tremor * 0.8 + np.random.normal(0, 0.05)
            az = tremor * 0.3 + np.random.normal(0, 0.05)

            self.new_data.emit(self.t, force, ax, ay, az)
            self.msleep(dt)

    def stop(self):
        self.running = False


# ---------------- SERIAL THREAD ----------------


class SerialThread(QThread):
    new_data = Signal(float, float, float, float, float)

    def __init__(self):
        super().__init__()
        self.ser = serial.Serial("COM7", 115200, timeout=1)
        self.running = True

    def run(self):
        while self.running:
            try:
                line = self.ser.readline().decode().strip()
                parts = line.split(",")
                if len(parts) != 5:
                    continue
                t, f, ax, ay, az = map(float, parts)
                self.new_data.emit(t, f, ax, ay, az)
            except Exception:
                continue

    def stop(self):
        self.running = False
        self.ser.close()


# ---------------- MAIN WINDOW ----------------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grip Analyzer")
        self.setGeometry(100, 100, 1000, 720)

        self.setStyleSheet(
            """
QWidget {
    background-color: #121212;
    color: white;
    font-family: Arial;
}
QPushButton {
    background-color: #00AEEF;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 15px;
    min-width: 120px;
    max-width: 200px;
}
QPushButton:hover { background-color: #0095cc; }
QLineEdit, QComboBox {
    background-color: #1E1E1E;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 6px;
}
"""
        )

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.user_data = {}
        self.time, self.force = [], []
        self.ax, self.ay, self.az = [], [], []
        # Indices where the load cell value actually changed (new unique reading).
        # At 10 Hz load cell / 100 Hz acc, 1 in every 10 samples is a real new reading.
        self.force_unique_idx: list[int] = []
        self._last_force_val: float | None = None

        FAKE_DATA = False

        self.serial = FakeSerialThread() if FAKE_DATA else SerialThread()
        self.serial.new_data.connect(self.store_data)
        self.serial.start()

        self.start_page = StartPage(self)
        self.info_page = InfoPage(self)
        self.protocol_page = ProtocolPage(self)
        self.test_page = TestPage(self)
        self.result_page = ResultPage(self)

        for page in [
            self.start_page,
            self.info_page,
            self.protocol_page,
            self.test_page,
            self.result_page,
        ]:
            self.stack.addWidget(page)

    def store_data(self, t, f, ax, ay, az):
        idx = len(self.time)  # index before appending
        self.time.append(t)
        self.force.append(f)
        self.ax.append(ax)
        self.ay.append(ay)
        self.az.append(az)
        # Track indices where load cell produced a genuinely new reading
        if f != self._last_force_val:
            self.force_unique_idx.append(idx)
            self._last_force_val = f


# ---------------- START PAGE ----------------


class StartPage(QWidget):
    def __init__(self, main):
        super().__init__()
        layout = QVBoxLayout()

        title = QLabel("Hand Grip Test")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 40px;")

        btn = QPushButton("Começar")
        btn.clicked.connect(lambda: main.stack.setCurrentWidget(main.info_page))
        btn.setStyleSheet("font-size: 20px;")

        layout.addStretch()
        layout.addWidget(title)
        layout.addSpacing(80)
        layout.addWidget(btn, alignment=Qt.AlignCenter)
        layout.addStretch()
        self.setLayout(layout)


# ---------------- INFO PAGE ----------------


class InfoPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(60, 40, 60, 40)

        self.inputs = {}
        for label_text, key in [
            ("Altura (cm)", "height"),
            ("Peso (kg)", "weight"),
            ("Idade", "age"),
        ]:
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-size: 17px;")
            edit = QLineEdit()
            self.inputs[key] = edit
            layout.addWidget(lbl)
            layout.addWidget(edit)

        lbl_sex = QLabel("Sexo")
        lbl_sex.setStyleSheet("font-size: 17px;")
        self.sex_combo = QComboBox()
        self.sex_combo.addItems(["Masculino", "Feminino"])
        layout.addWidget(lbl_sex)
        layout.addWidget(self.sex_combo)

        btn = QPushButton("Prosseguir")
        btn.setStyleSheet("font-size: 20px;")
        btn.clicked.connect(self.save)
        layout.addSpacing(16)
        layout.addWidget(btn, alignment=Qt.AlignCenter)
        self.setLayout(layout)

    def save(self):
        self.main.user_data = {
            "height": self.inputs["height"].text(),
            "weight": self.inputs["weight"].text(),
            "age": self.inputs["age"].text(),
            "sex": "M" if self.sex_combo.currentIndex() == 0 else "F",
        }
        self.main.stack.setCurrentWidget(self.main.protocol_page)


# ---------------- PROTOCOL PAGE ----------------


class ProtocolPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(30, 20, 30, 20)

        text_protocolo = QLabel("Protocolo de Hand Grip Test\n\n")
        text_protocolo.setAlignment(Qt.AlignCenter)
        text_protocolo.setStyleSheet("font-size: 30px;")
        text_protocolo.setContentsMargins(0, 20, 0, 10)

        # Step 1
        text_primeiro_passo = QLabel("1º Passo:\n")
        text_primeiro_passo.setStyleSheet("font-size: 22px;")
        text_primeiro_passo.setContentsMargins(10, 0, 0, 0)

        text_mvc = QLabel(
            "• Realização de Contração Máxima durante 5s\n\n"
            f"• Descanso: {REST_AFTER_MVC}s\n\n"
            "• Repetição do 1º Passo mais 2 vezes\n"
        )
        text_mvc.setAlignment(Qt.AlignCenter)
        text_mvc.setStyleSheet("font-size: 20px;")

        # Step 2
        text_segundo_passo = QLabel("2º Passo:\n")
        text_segundo_passo.setStyleSheet("font-size: 22px;")
        text_segundo_passo.setContentsMargins(10, 0, 0, 0)

        text_manter = QLabel(
            "• Manter 30% da Contração Máxima durante 20s\n\n"
            f"• Descanso: {REST_AFTER_SUB}s\n\n"
            "• Repetição do 2º Passo mais 2 vezes\n"
        )
        text_manter.setAlignment(Qt.AlignCenter)
        text_manter.setStyleSheet("font-size: 20px;")

        btn = QPushButton("Iniciar Teste")
        btn.clicked.connect(self.start)
        btn.setStyleSheet("font-size: 20px; padding: 10px;")

        def make_box(label_widget, text_widget):
            box = QWidget()
            bl = QVBoxLayout()
            bl.setSpacing(10)
            bl.addWidget(label_widget)
            bl.addWidget(text_widget)
            box.setLayout(bl)
            box.setStyleSheet(
                """
                QWidget {
                    border: 2px solid #555;
                    border-radius: 10px;
                    padding: 10px;
                    background-color: #1e1e1e;
                }
            """
            )
            return box

        layout.addWidget(text_protocolo)
        layout.addSpacing(-20)
        layout.addWidget(make_box(text_primeiro_passo, text_mvc))
        layout.addWidget(make_box(text_segundo_passo, text_manter))
        layout.addStretch()
        layout.addWidget(btn, alignment=Qt.AlignCenter)
        self.setLayout(layout)

    def start(self):
        self.main.stack.setCurrentWidget(self.main.test_page)
        self.main.test_page.start_sequence()


# ---------------- TEST PAGE ----------------


class TestPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.max_segments = []
        self.sub_segments = []

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(40, 20, 40, 20)

        self.top_container = QWidget()
        self.top_layout = QVBoxLayout()
        self.top_layout.setSpacing(5)

        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 35px;")

        self.timer_label = QLabel("")
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 40px; color: white;")

        self.top_layout.addWidget(self.label)
        self.top_layout.addWidget(self.timer_label)
        self.top_container.setLayout(self.top_layout)

        self.plot = pg.PlotWidget()
        self.plot.hide()
        self.plot.setMinimumHeight(250)
        self.plot.setBackground("#1E1E1E")

        self.curve = self.plot.plot(pen="c")
        self.target_line = self.plot.addLine(y=0, pen="r")

        self.main_layout.addWidget(self.top_container)
        self.main_layout.addWidget(self.plot)
        self.setLayout(self.main_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.phase_timer = None  # Timer para fases (MVC e Sub)

        self.phase = 0
        self.counter = 0
        self.start_idx = 0
        self.max_force = 0.0
        self.target = 0.0

    def switch_to_vertical(self):
        new_layout = QVBoxLayout()
        new_layout.setSpacing(5)
        new_layout.addSpacing(50)
        new_layout.addWidget(self.label)
        new_layout.addSpacing(-200)
        new_layout.addWidget(self.timer_label)
        self._replace_layout(new_layout)

    def switch_to_horizontal(self):
        new_layout = QHBoxLayout()
        new_layout.setSpacing(20)
        new_layout.addWidget(self.label)
        new_layout.addWidget(self.timer_label)
        self._replace_layout(new_layout)

    def _replace_layout(self, new_layout):
        QWidget().setLayout(self.top_layout)
        self.top_layout = new_layout
        self.top_container.setLayout(self.top_layout)

    def start_sequence(self):
        self.phase = 0
        self.next_phase()

    def next_phase(self):
        # MVCs (phases 0, 2, 4)
        if self.phase in (0, 2, 4):
            rep = {0: 1, 2: 2, 4: 3}[self.phase]
            self.switch_to_vertical()
            self.plot.hide()
            self.countdown(f"MVC {rep}/3 — Contração Máxima em...", 3, self.start_max)

        # Rest after MVCs (phases 1, 3)
        elif self.phase in (1, 3):
            self.start_rest(REST_AFTER_MVC, "Descanso")

        # Rest after last MVC (phase 5)
        elif self.phase == 5:
            self.start_rest(REST_AFTER_MVC, "Descanso")

        # Calculate max_force from MVC segments
        elif self.phase == 6:
            forces = self.main.force
            max_vals = [
                float(np.max(np.array(forces[s:e])))
                for s, e in self.max_segments
                if e > s and len(forces[s:e]) > 0
            ]
            self.max_force = max(max_vals) if max_vals else 1.0
            self.target = 0.3 * self.max_force
            self.phase = 7
            self.next_phase()

        # Submaximals (phases 7, 9, 11)
        elif self.phase in (7, 9, 11):
            rep = {7: 1, 9: 2, 11: 3}[self.phase]
            self.switch_to_horizontal()
            self.plot.show()
            self.countdown(
                f"Sub {rep}/3 — Mantenha {self.target / 1000:.1f} kg em...",
                3,
                self.start_sub,
            )

        # Rest after submaximals (phases 8, 10)
        elif self.phase in (8, 10):
            self.start_rest(REST_AFTER_SUB, "Descanso")

        # End of test (phase 12)
        else:
            self.timer.stop()
            self.main.stack.setCurrentWidget(self.main.result_page)
            self.main.result_page.compute()

    def countdown(self, text, sec, callback):
        self.counter = sec
        self.label.setText(text)

        # Para o timer anterior se ainda estiver ativo
        if self.phase_timer is not None:
            self.phase_timer.stop()

        self.phase_timer = QTimer()

        def update():
            if self.counter > 0:
                self.timer_label.setText(str(self.counter))
                self.counter -= 1
            else:
                self.phase_timer.stop()
                callback()

        self.phase_timer.timeout.connect(update)
        self.phase_timer.start(1000)

    def start_rest(self, duration, text="Descanso"):
        self.timer.stop()
        self.switch_to_vertical()
        self.plot.hide()
        self.label.setText(text)
        self.counter = duration

        # Para o timer anterior se ainda estiver ativo
        if self.phase_timer is not None:
            self.phase_timer.stop()

        self.phase_timer = QTimer()

        def update():
            if self.counter > 0:
                self.timer_label.setText(f"{self.counter}s")
                self.counter -= 1
            else:
                self.phase_timer.stop()
                self.phase += 1
                self.next_phase()

        self.phase_timer.timeout.connect(update)
        self.phase_timer.start(1000)

    def start_max(self):
        rep = {0: 1, 2: 2, 4: 3}.get(self.phase, "?")
        self.label.setText(f"MVC {rep}/3 — Realize a Contração Máxima")
        self.plot.hide()
        self.start_idx = len(self.main.force)
        self.run_phase(5, segment_type="max")

    def start_sub(self):
        rep = {7: 1, 9: 2, 11: 3}.get(self.phase, "?")
        self.label.setText(f"Sub {rep}/3 — Mantenha {self.target / 1000:.1f} kg")
        self.plot.show()
        self.start_idx = len(self.main.time)

        target_kg = self.target / 1000.0
        self.plot.setYRange(0, 2.5 * target_kg)
        self.plot.setXRange(0, 20)
        self.plot.enableAutoRange(False)
        self.plot.setLabel("left", "Força (kg)")
        self.plot.setLabel("bottom", "Tempo (s)")
        self.target_line.setValue(target_kg)

        self.timer.start(50)
        self.run_phase(20, segment_type="sub")

    def run_phase(self, duration, segment_type=None):
        self.counter = duration

        # Para o timer anterior se ainda estiver ativo
        if self.phase_timer is not None:
            self.phase_timer.stop()

        # Cria um novo timer e guarda como atributo para evitar garbage collection
        self.phase_timer = QTimer()

        def update():
            if self.counter > 0:
                self.timer_label.setText(f"{self.counter}s")
                self.counter -= 1
            else:
                self.phase_timer.stop()
                end_idx = len(self.main.force)
                if segment_type == "max":
                    self.max_segments.append((self.start_idx, end_idx))
                elif segment_type == "sub":
                    self.sub_segments.append((self.start_idx, end_idx))
                self.phase += 1
                self.next_phase()

        self.phase_timer.timeout.connect(update)
        self.phase_timer.start(1000)

    def update_plot(self):
        t = np.array(self.main.time[self.start_idx :])
        f = np.array(self.main.force[self.start_idx :]) / 1000.0  # Converter para kg
        if len(t) < 2 or len(f) < 2:
            return
        t = (t - t[0]) / 1000.0
        self.curve.setData(t, f)


# ---------------- RESULT PAGE ----------------


class ResultPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.results = {}

        layout = QVBoxLayout()
        layout.setContentsMargins(25, 15, 25, 15)
        layout.setSpacing(10)

        self.title = QLabel("Resultados")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 26px; font-weight: bold;")

        # Quatro caixas de métricas em linha
        self.text_layout = QHBoxLayout()
        self.force_label = QLabel()
        self.control_label = QLabel()
        self.tremor_force_label = QLabel()
        self.tremor_acc_label = QLabel()

        for lbl in [
            self.force_label,
            self.control_label,
            self.tremor_force_label,
            self.tremor_acc_label,
        ]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(
                "font-size: 12px; border: 1px solid #444; "
                "border-radius: 8px; padding: 8px;"
            )

            self.text_layout.addWidget(lbl)

        # Três gráficos
        self.plot_force = pg.PlotWidget(title="Força ao longo do teste")
        self.plot_psd = pg.PlotWidget(title="PSD: Força (0-5 Hz)")
        self.plot_time = pg.PlotWidget(title="PSD: Tremor da mão (4-12 Hz)")

        for p in [self.plot_force, self.plot_psd, self.plot_time]:
            p.setBackground("#1E1E1E")
            p.setMinimumHeight(160)

        layout.addWidget(self.title)
        layout.addLayout(self.text_layout)
        layout.addWidget(self.plot_force)
        layout.addWidget(self.plot_psd)
        layout.addWidget(self.plot_time)
        layout.addStretch()

        # Botões alinhados à direita
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.save_btn = QPushButton("Guardar CSV")
        self.save_btn.setMaximumWidth(150)
        self.save_btn.clicked.connect(self.save_csv)
        button_layout.addWidget(self.save_btn)
        self.finish_btn = QPushButton("Concluir")
        self.finish_btn.setMaximumWidth(150)
        self.finish_btn.clicked.connect(self.on_finish_clicked)
        button_layout.addWidget(self.finish_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    # ------------------------------------------------------------------ #
    #  ON FINISH CLICKED                                                   #
    # ------------------------------------------------------------------ #

    def on_finish_clicked(self):
        """Fecha a aplicação quando o botão Concluir é clicado"""
        QApplication.quit()

    # ------------------------------------------------------------------ #
    #  COMPUTE                                                             #
    # ------------------------------------------------------------------ #

    def compute(self):
        force = np.array(self.main.force) / 1000.0  # Converter ADU para kg
        ax_arr = np.array(self.main.ax)
        ay_arr = np.array(self.main.ay)
        az_arr = np.array(self.main.az)
        acc = np.sqrt(ax_arr**2 + ay_arr**2 + az_arr**2)
        time = np.array(self.main.time) / 1000.0  # ms → s

        # ==============================================================
        # PHASE 1 — MVC: Fmax, Time to Peak
        # ==============================================================
        max_forces, t_peaks = [], []
        mvc_duration = 5.0  # MVC expected duration in seconds

        for start, end in self.main.test_page.max_segments:
            f_seg = force[start:end]
            t_seg = time[start:end]
            if len(f_seg) < 5:
                continue

            peak_idx = int(np.argmax(f_seg))
            max_forces.append(f_seg[peak_idx])

            # Onset detection: first sample > baseline + 5% of segment peak
            baseline = float(np.mean(f_seg[:3])) if len(f_seg) >= 3 else 0.0
            onset_threshold = baseline + 0.05 * f_seg[peak_idx]
            above_threshold = np.where(f_seg >= onset_threshold)[0]

            if len(above_threshold) > 0:
                onset_idx = int(above_threshold[0])
                ttp = float(t_seg[peak_idx] - t_seg[onset_idx])
            else:
                ttp = float(t_seg[peak_idx] - t_seg[0])  # fallback

            t_peaks.append(ttp)

        if not max_forces:
            self.force_label.setText("Erro: sem dados MVC")
            return

        Fmax = float(np.max(max_forces))
        TTP_mean = float(np.mean(t_peaks)) if t_peaks else 0.0

        # ---- Z-score HGS (Dodds et al., 2014) ----
        # NOTE: Z-score assumes Fmax is in kgf (calibrated units).
        # With raw ADC units the Z-score is for reference only.
        Z = None
        try:
            age = int(self.main.user_data.get("age", 0))
            sex = self.main.user_data.get("sex", "M")
            Z = compute_zscore(Fmax, age, sex)
        except Exception:
            Z = None

        # ==============================================================
        # PHASE 2 — Sub-MVC: CV, PSD Force 0–4 Hz, PSD Acc 8–12 Hz
        # ==============================================================
        sub_segments = self.main.test_page.sub_segments
        if not sub_segments:
            self.force_label.setText("Erro: sem segmento sub-MVC")
            return

        cvs, power_forces, power_accs = [], [], []
        tfr_means, tfr_stds = [], []

        # Para os espectros (guardados com suas grelhas nativas, sem interpolação prévia)
        psd_force_list = []
        psd_acc_list = []
        fft_force_list = []
        fft_acc_list = []
        time_domain_force_list = []
        time_domain_acc_list = []
        time_domain_time_force_list = []
        time_domain_time_acc_list = []
        fft_freq_grid = None
        psd_freq_grid = None

        for start, end in sub_segments:
            # --- Full-resolution arrays (100 Hz, shared timestamps) ---
            f_full = force[start:end]  # force with duplicates
            a_full = acc[start:end]  # acc at full 100 Hz
            t_full = time[start:end]  # timestamps in seconds

            if len(f_full) < 20:
                continue

            # --- Deduplicated force (real HX711 readings only, ~10 Hz) ---
            unique_idx_global = np.array(self.main.force_unique_idx)
            mask_seg = (unique_idx_global >= start) & (unique_idx_global < end)
            seg_unique_idx = unique_idx_global[mask_seg]  # global indices into force[]
            f_dedup = force[seg_unique_idx]
            t_dedup = time[seg_unique_idx]  # timestamps of real readings

            if len(f_dedup) < 4:
                continue

            fs_acc = (
                len(f_full) / (t_full[-1] - t_full[0])
                if t_full[-1] > t_full[0]
                else 100.0
            )
            fs_force_real = (
                len(f_dedup) / (t_dedup[-1] - t_dedup[0])
                if t_dedup[-1] > t_dedup[0]
                else 10.0
            )

            # --- Trim to after 90% of target reached (use full-res force for detection) ---
            target_threshold = 0.9 * 0.3 * Fmax
            above = np.where(f_full >= target_threshold)[0]
            idx_entry = above[0] if len(above) > 0 else 0
            t_entry = t_full[idx_entry]

            f_full_trim = f_full[idx_entry:]
            a_full_trim = a_full[idx_entry:]
            t_full_trim = t_full[idx_entry:]
            mask_dedup_trim = t_dedup >= t_entry
            f_dedup_trim = f_dedup[mask_dedup_trim]
            t_dedup_trim = t_dedup[mask_dedup_trim]

            if len(f_dedup_trim) < 4:
                continue

            # --- Most stable 10-second window (decided on deduplicated force) ---
            t_win_start, t_win_end = most_stable_window_dedup(
                f_dedup_trim, t_dedup_trim, window_sec=10
            )

            # Slice deduplicated force in that window
            mask_stable_dedup = (t_dedup_trim >= t_win_start) & (
                t_dedup_trim < t_win_end
            )
            f_stable_dedup = f_dedup_trim[mask_stable_dedup]
            t_stable_dedup = t_dedup_trim[mask_stable_dedup]

            # Slice full-res acc in the same time window
            mask_stable_full = (t_full_trim >= t_win_start) & (t_full_trim < t_win_end)
            a_stable_full_win = a_full_trim[mask_stable_full]
            t_stable_full_win = t_full_trim[mask_stable_full]

            if len(f_stable_dedup) < 2 or len(a_stable_full_win) < 4:
                continue

            # ---- CV (%) — deduplicated force, stable window ----
            mu = float(np.mean(f_stable_dedup))
            if mu != 0:
                cvs.append(float(np.std(f_stable_dedup)) / mu * 100.0)

            # ---- PSD Force 0–4 Hz — deduplicated force, Hamming ----
            try:
                nyq_f = fs_force_real / 2.0
                hi_f = min(4.0 / nyq_f, 0.999)
                lo_f = max(0.1 / nyq_f, 0.001)
                b, a_coef = butter(4, [lo_f, hi_f], btype="band")
                f_filt = filtfilt(b, a_coef, f_stable_dedup)
                nperseg_f = min(len(f_filt), 64)  # at ~10 Hz / 10 s → ~100 samples
                f_pf, Pxx_f = welch(
                    f_filt,
                    fs_force_real,
                    window="hamming",
                    nperseg=nperseg_f,
                    noverlap=nperseg_f // 2,
                )
                f_pf, Pxx_f = np.asarray(f_pf), np.asarray(Pxx_f)
                band_f = (f_pf >= 0.1) & (f_pf <= 4.0)
                power_forces.append(float(np.trapezoid(Pxx_f[band_f], f_pf[band_f])))
            except Exception:
                Pxx_f = np.zeros(2)
                f_pf = np.array([0.0, 1.0])

            # ---- PSD Acc 2–15 Hz — full-res acc, stable window, Hamming ----
            # Filter 2–15 Hz to capture the full tremor classification spectrum:
            #   2–4 Hz  → cerebellar / parkinsonian rest tremor
            #   4–8 Hz  → essential tremor / mixed
            #   8–12 Hz → physiological tremor
            # Peak frequency and FWHM are extracted over the full 2–15 Hz band.
            # Power is still integrated over 8–12 Hz only for the quantitative metric.
            try:
                nyq_a = fs_acc / 2.0
                lo_a = max(2.0 / nyq_a, 0.001)
                hi_a = min(15.0 / nyq_a, 0.999)
                b_a, a_a_coef = butter(4, [lo_a, hi_a], btype="band")
                a_filt = filtfilt(b_a, a_a_coef, a_stable_full_win)
                nperseg_a = min(len(a_filt), 256)
                f_pa, Pxx_a = welch(
                    a_filt,
                    fs_acc,
                    window="hamming",
                    nperseg=nperseg_a,
                    noverlap=nperseg_a // 2,
                )
                f_pa, Pxx_a = np.asarray(f_pa), np.asarray(Pxx_a)
                # Quantitative metric: integrate only physiological tremor band
                band_physio = (f_pa >= 8.0) & (f_pa <= 12.0)
                power_accs.append(
                    float(np.trapezoid(Pxx_a[band_physio], f_pa[band_physio]))
                )
            except Exception:
                Pxx_a = np.zeros(2)
                f_pa = np.array([0.0, 1.0])

            # ---- TFR — 1-second windows, dedup force + full-res acc ----
            tfr_mean_seg, tfr_std_seg = compute_tfr(
                f_stable_dedup,
                t_stable_dedup,
                a_stable_full_win,
                t_stable_full_win,
            )
            if tfr_mean_seg > 0:
                tfr_means.append(tfr_mean_seg)
                tfr_stds.append(tfr_std_seg)

            # ---- Store spectra WITHOUT interpolation (preserve native frequency grids) ----
            psd_force_list.append(
                (f_pf, Pxx_f)
            )  # Store (frequency, power) tuple for force
            psd_acc_list.append(
                (f_pa, Pxx_a)
            )  # Store (frequency, power) tuple for acceleration

            # ---- Time-domain for visualisation (full trimmed segment) ----
            def butter_band(sig, low, high, fs):
                try:
                    nyq = fs / 2
                    b, a = butter(
                        4, [max(low / nyq, 0.001), min(high / nyq, 0.999)], btype="band"
                    )
                    return filtfilt(b, a, sig)
                except Exception:
                    return sig

            time_domain_force_list.append(
                butter_band(f_full_trim, 0.1, 4.0, fs_force_real)
            )
            time_domain_acc_list.append(butter_band(a_full_trim, 4.0, 12.0, fs_acc))
            time_domain_time_force_list.append(t_full_trim)
            time_domain_time_acc_list.append(t_full_trim)

        # ---- Aggregate across repetitions ----
        cv = float(np.mean(cvs)) if cvs else 0.0
        psd_force_04 = float(np.mean(power_forces)) if power_forces else 0.0
        # PSD acc 8-12 Hz: mean of per-repetition integrals (consistent with force PSD method)
        psd_acc_812 = float(np.mean(power_accs)) if power_accs else 0.0

        # Tremor-Force Ratio
        tfr_mean = float(np.mean(tfr_means)) if tfr_means else 0.0
        tfr_std = float(np.mean(tfr_stds)) if tfr_stds else 0.0

        # ---- Averaged PSD spectra for plotting only ----
        mean_psd_force = None
        mean_psd_acc = None
        psd_freq_force = None
        psd_freq_acc = None
        peak_freq_force = 0.0
        peak_freq_acc = 0.0
        freq_spread = 0.0  # FWHM of dominant tremor peak

        if psd_force_list:
            psd_freq_force = np.linspace(0, 5, 256)
            psd_avg_force = np.zeros_like(psd_freq_force)
            for f_array, pxx_array in psd_force_list:
                psd_avg_force += np.interp(
                    psd_freq_force, f_array, pxx_array, left=0, right=0
                )
            mean_psd_force = psd_avg_force / len(psd_force_list)

            # Peak frequency of dominant force fluctuation component (0.1–4 Hz)
            band_force_plot = (psd_freq_force >= 0.1) & (psd_freq_force <= 4.0)
            if np.any(band_force_plot):
                peak_idx_force = int(np.argmax(mean_psd_force[band_force_plot]))
                peak_freq_force = float(psd_freq_force[band_force_plot][peak_idx_force])
            else:
                peak_freq_force = 0.0

        if psd_acc_list:
            # Average over full 2–15 Hz classification band
            psd_freq_acc = np.linspace(2, 15, 512)
            psd_avg_acc = np.zeros_like(psd_freq_acc)
            for f_array, pxx_array in psd_acc_list:
                psd_avg_acc += np.interp(
                    psd_freq_acc, f_array, pxx_array, left=0, right=0
                )
            mean_psd_acc = psd_avg_acc / len(psd_acc_list)

            # Peak frequency over the full classification band (2–15 Hz)
            peak_idx_acc = int(np.argmax(mean_psd_acc))
            peak_freq_acc = float(psd_freq_acc[peak_idx_acc])

            # FWHM of the dominant peak
            half_max = mean_psd_acc[peak_idx_acc] / 2.0
            above_half = mean_psd_acc >= half_max
            if np.any(above_half):
                idxs = np.where(above_half)[0]
                freq_spread = float(psd_freq_acc[idxs[-1]] - psd_freq_acc[idxs[0]])

            # ---- Tremor classification flag based on peak frequency ----
            # Apenas Essencial ou Fisiológico
            if peak_freq_acc < 8.0:
                tremor_class = "Essencial/Misto (2–8 Hz)"
            else:
                tremor_class = "Fisiológico (8–12 Hz)"

        # ---- Tremor RMS — from stable window acc (consistent with other metrics) ----
        tremor_rms = 0.0
        if time_domain_acc_list:
            # Each entry in time_domain_acc_list is filtered acc over the full trimmed segment.
            # For RMS we use only the stable 10-s portion, approximated as the central 10 s
            # (the stable window was already selected per-segment during computation above).
            # Re-use the per-segment a_stable_full_win that was last computed — instead,
            # compute RMS from the per-segment filtered stable acc collected during the loop.
            pass  # RMS computed below from stored stable segments

        # Collect stable acc segments for RMS (done in loop above via time_domain_acc_list;
        # here we use mean_psd_acc as proxy since direct stable acc isn't separately stored)
        # Simplest consistent approach: RMS from averaged PSD via Parseval's theorem
        if mean_psd_acc is not None and psd_freq_acc is not None:
            df = psd_freq_acc[1] - psd_freq_acc[0]
            tremor_rms = float(np.sqrt(np.sum(mean_psd_acc) * df))

        # ==============================================================
        # STORE FOR CSV
        # ==============================================================
        z_str = f"{Z:.2f}" if Z is not None else "N/A"
        self.results = {
            "Fmax": round(Fmax, 3),
            "TTP_s": round(TTP_mean, 3),
            "Z_HGS": z_str,
            "CV_pct": round(cv, 3),
            "PSD_Force_0_4Hz": round(psd_force_04, 6),
            "Peak_Freq_Force_Hz": round(peak_freq_force, 2),
            "PSD_Acc_8_12Hz": round(psd_acc_812, 6),
            "Peak_Freq_Tremor_Hz": round(peak_freq_acc, 2),
            "Tremor_FWHM_Hz": round(freq_spread, 2),
            "Tremor_RMS_g": round(tremor_rms, 6),
            "TFR_mean": round(tfr_mean, 6),
            "TFR_std": round(tfr_std, 6),
        }

        # ==============================================================
        # UPDATE UI
        # ==============================================================
        z_str_display = f"{Z:.2f}" if Z is not None else "N/A"
        self.force_label.setText(
            f"🔴  FORÇA\n"
            f"Fmax: {Fmax:.1f} kg\n"
            f"Time to Peak: {TTP_mean:.2f} s\n"
            f"Z-score HGS: {z_str_display}"
        )
        self.control_label.setText(
            f"🟢  CONTROLO\n" f"CV: {cv:.1f} %\n" f"TFR: {tfr_mean:.4f} ± {tfr_std:.4f}"
        )
        self.tremor_force_label.setText(
            f"🟣  TREMOR FORÇA\n"
            f"PSD 0.1–4 Hz:\n"
            f"{psd_force_04:.4f} kg²/Hz\n"
            f"Peak: {peak_freq_force:.2f} Hz"
        )
        psd_acc_note = f"{psd_acc_812:.2e} g²/Hz" if psd_acc_812 > 0 else "N/A"
        self.tremor_acc_label.setText(
            f"🟡  TREMOR MÃO\n"
            f"PSD 8–12 Hz: {psd_acc_note}\n"
            f"Peak: {peak_freq_acc:.1f} Hz | FWHM: {freq_spread:.2f} Hz\n"
        )

        # ---- Plots ----
        self.plot_force.clear()
        time_s = time
        self.plot_force.plot(time_s, force, pen=pg.mkPen("#555", width=1))
        self.plot_force.setLabel("left", "Força (Kg)")
        self.plot_force.setLabel("bottom", "Tempo (s)")

        # Sombrear MVC
        for s, e in self.main.test_page.max_segments:
            if e <= s or e > len(time):
                continue
            e = min(e, len(time) - 1)
            region = pg.LinearRegionItem(
                [time_s[s], time_s[e]], brush=(50, 100, 255, 50)
            )
            region.setMovable(False)
            self.plot_force.addItem(region)

        # Sombrear Sub-MVC
        for s, e in self.main.test_page.sub_segments:
            if e <= s or e > len(time):
                continue
            e = min(e, len(time) - 1)
            region = pg.LinearRegionItem(
                [time_s[s], time_s[e]], brush=(50, 200, 100, 50)
            )
            region.setMovable(False)
            self.plot_force.addItem(region)

        # Plot da força por cima
        self.plot_force.plot(time_s, force, pen=pg.mkPen("c", width=1))

        # --- PSD Força (0-5 Hz) ---
        if mean_psd_force is not None and psd_freq_force is not None:
            self.plot_psd.clear()
            self.plot_psd.plot(
                psd_freq_force, mean_psd_force, pen="r", name="Força (0.1-4 Hz)"
            )
            self.plot_psd.setXRange(0, 5)
            self.plot_psd.setLabel("left", "Potência (kg²/Hz)")
            self.plot_psd.setLabel("bottom", "Frequência (Hz)")
            self.plot_psd.addLegend()

        # --- PSD Tremor (2-15 Hz — full classification band) ---
        if mean_psd_acc is not None and psd_freq_acc is not None:
            self.plot_time.clear()
            self.plot_time.plot(
                psd_freq_acc, mean_psd_acc, pen="y", name="Tremor (2-15 Hz)"
            )
            # Add vertical reference line for tremor classification band (8 Hz)
            self.plot_time.addLine(x=8.0, pen=pg.mkPen("r", style=Qt.DashLine))
            self.plot_time.setXRange(2, 15)
            self.plot_time.setLabel("left", "Potência (g²/Hz)")
            self.plot_time.setLabel("bottom", "Frequência (Hz)")
            self.plot_time.setTitle("PSD: Tremor da mão (2–15 Hz)")

    # ------------------------------------------------------------------ #
    #  CSV SAVE                                                            #
    # ------------------------------------------------------------------ #

    def save_csv(self):
        import csv, os

        file = "results.csv"
        exists = os.path.isfile(file)

        with open(file, "a", newline="") as fh:
            writer = csv.writer(fh)
            if not exists:
                writer.writerow(
                    [
                        "age",
                        "sex",
                        "height",
                        "weight",
                        "Fmax",
                        "TTP_s",
                        "Z_HGS",
                        "CV_pct",
                        "PSD_Force_0_4Hz",
                        "Peak_Freq_Force_Hz",
                        "PSD_Acc_8_12Hz",
                        "Peak_Freq_Tremor_Hz",
                        "Tremor_FWHM_Hz",
                        "Tremor_RMS_g",
                        "TFR_mean",
                        "TFR_std",
                    ]
                )
            writer.writerow(
                [
                    self.main.user_data.get("age"),
                    self.main.user_data.get("sex"),
                    self.main.user_data.get("height"),
                    self.main.user_data.get("weight"),
                    self.results.get("Fmax"),
                    self.results.get("TTP_s"),
                    self.results.get("Z_HGS"),
                    self.results.get("CV_pct"),
                    self.results.get("PSD_Force_0_4Hz"),
                    self.results.get("Peak_Freq_Force_Hz"),
                    self.results.get("PSD_Acc_8_12Hz"),
                    self.results.get("Peak_Freq_Tremor_Hz"),
                    self.results.get("Tremor_FWHM_Hz"),
                    self.results.get("Tremor_RMS_g"),
                    self.results.get("TFR_mean"),
                    self.results.get("TFR_std"),
                ]
            )


# ---------------- RUN ----------------

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
