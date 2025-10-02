"""
ChemPyLab - Kimya Öğrencileri için Kapsamlı Python Kütüphanesi
================================================================
Temel kimya hesaplamalarından ML/DL uygulamalarına kadar geniş bir yelpazede araçlar sunar.
"""

import numpy as np
import pandas as pd
from scipy import constants, optimize, integrate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
import warnings
from collections import defaultdict

# ==================== TEMEL SABİTLER ====================
class Constants:
    """Kimyasal sabitler"""
    AVOGADRO = 6.022e23  # mol^-1
    R_GAS = 8.314  # J/(mol·K)
    FARADAY = 96485  # C/mol
    PLANCK = 6.626e-34  # J·s
    LIGHT_SPEED = 2.998e8  # m/s
    BOLTZMANN = 1.381e-23  # J/K
    ELECTRON_CHARGE = 1.602e-19  # C
    
# ==================== PERİYODİK TABLO ====================
class PeriodicTable:
    """Periyodik tablo ve element özellikleri"""
    
    def __init__(self):
        self.elements = {
            'H': {'name': 'Hidrojen', 'mass': 1.008, 'number': 1, 'electronegativity': 2.20, 'radius': 37},
            'He': {'name': 'Helyum', 'mass': 4.003, 'number': 2, 'electronegativity': None, 'radius': 32},
            'Li': {'name': 'Lityum', 'mass': 6.941, 'number': 3, 'electronegativity': 0.98, 'radius': 134},
            'Be': {'name': 'Berilyum', 'mass': 9.012, 'number': 4, 'electronegativity': 1.57, 'radius': 90},
            'B': {'name': 'Bor', 'mass': 10.81, 'number': 5, 'electronegativity': 2.04, 'radius': 82},
            'C': {'name': 'Karbon', 'mass': 12.01, 'number': 6, 'electronegativity': 2.55, 'radius': 77},
            'N': {'name': 'Azot', 'mass': 14.01, 'number': 7, 'electronegativity': 3.04, 'radius': 75},
            'O': {'name': 'Oksijen', 'mass': 16.00, 'number': 8, 'electronegativity': 3.44, 'radius': 73},
            'F': {'name': 'Flor', 'mass': 19.00, 'number': 9, 'electronegativity': 3.98, 'radius': 71},
            'Ne': {'name': 'Neon', 'mass': 20.18, 'number': 10, 'electronegativity': None, 'radius': 69},
            'Na': {'name': 'Sodyum', 'mass': 22.99, 'number': 11, 'electronegativity': 0.93, 'radius': 154},
            'Mg': {'name': 'Magnezyum', 'mass': 24.31, 'number': 12, 'electronegativity': 1.31, 'radius': 130},
            'Al': {'name': 'Alüminyum', 'mass': 26.98, 'number': 13, 'electronegativity': 1.61, 'radius': 118},
            'Si': {'name': 'Silisyum', 'mass': 28.09, 'number': 14, 'electronegativity': 1.90, 'radius': 111},
            'P': {'name': 'Fosfor', 'mass': 30.97, 'number': 15, 'electronegativity': 2.19, 'radius': 106},
            'S': {'name': 'Kükürt', 'mass': 32.07, 'number': 16, 'electronegativity': 2.58, 'radius': 102},
            'Cl': {'name': 'Klor', 'mass': 35.45, 'number': 17, 'electronegativity': 3.16, 'radius': 99},
            'Ar': {'name': 'Argon', 'mass': 39.95, 'number': 18, 'electronegativity': None, 'radius': 97},
            'K': {'name': 'Potasyum', 'mass': 39.10, 'number': 19, 'electronegativity': 0.82, 'radius': 196},
            'Ca': {'name': 'Kalsiyum', 'mass': 40.08, 'number': 20, 'electronegativity': 1.00, 'radius': 174},
            'Fe': {'name': 'Demir', 'mass': 55.85, 'number': 26, 'electronegativity': 1.83, 'radius': 125},
            'Cu': {'name': 'Bakır', 'mass': 63.55, 'number': 29, 'electronegativity': 1.90, 'radius': 128},
            'Zn': {'name': 'Çinko', 'mass': 65.39, 'number': 30, 'electronegativity': 1.65, 'radius': 134},
            'Br': {'name': 'Brom', 'mass': 79.90, 'number': 35, 'electronegativity': 2.96, 'radius': 114},
            'Ag': {'name': 'Gümüş', 'mass': 107.87, 'number': 47, 'electronegativity': 1.93, 'radius': 144},
            'I': {'name': 'İyot', 'mass': 126.90, 'number': 53, 'electronegativity': 2.66, 'radius': 133},
            'Au': {'name': 'Altın', 'mass': 196.97, 'number': 79, 'electronegativity': 2.54, 'radius': 144},
        }
    
    def get_element(self, symbol: str) -> Dict:
        """Element bilgilerini getir"""
        return self.elements.get(symbol, None)
    
    def get_mass(self, symbol: str) -> float:
        """Atomik kütle"""
        elem = self.get_element(symbol)
        return elem['mass'] if elem else None

# ==================== MOLEKÜL SINIFI ====================
@dataclass
class Molecule:
    """Molekül sınıfı"""
    formula: str
    name: Optional[str] = None
    smiles: Optional[str] = None
    
    def __post_init__(self):
        self.composition = self._parse_formula()
        self.molecular_weight = self._calculate_mw()
    
    def _parse_formula(self) -> Dict[str, int]:
        """Molekül formülünü parse et"""
        import re
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, self.formula)
        composition = {}
        for element, count in matches:
            count = int(count) if count else 1
            composition[element] = composition.get(element, 0) + count
        return composition
    
    def _calculate_mw(self) -> float:
        """Molekül ağırlığını hesapla"""
        pt = PeriodicTable()
        mw = 0
        for element, count in self.composition.items():
            mass = pt.get_mass(element)
            if mass:
                mw += mass * count
        return mw
    
    def empirical_formula(self) -> str:
        """Empirik formül"""
        from math import gcd
        from functools import reduce
        
        counts = list(self.composition.values())
        common_divisor = reduce(gcd, counts)
        
        empirical = {}
        for element, count in self.composition.items():
            empirical[element] = count // common_divisor
        
        formula = ""
        for elem, count in empirical.items():
            if count == 1:
                formula += elem
            else:
                formula += f"{elem}{count}"
        return formula

# ==================== TEMEL KİMYA HESAPLAMALARI ====================
class BasicChemistry:
    """Temel kimya hesaplamaları"""
    
    @staticmethod
    def molarity(moles: float, volume_L: float) -> float:
        """Molarite hesapla (mol/L)"""
        return moles / volume_L
    
    @staticmethod
    def molality(moles_solute: float, kg_solvent: float) -> float:
        """Molalite hesapla (mol/kg)"""
        return moles_solute / kg_solvent
    
    @staticmethod
    def dilution(C1: float, V1: float, V2: float) -> float:
        """Seyreltme hesabı: C1V1 = C2V2"""
        return (C1 * V1) / V2
    
    @staticmethod
    def pH(H_concentration: float = None, OH_concentration: float = None) -> float:
        """pH hesapla"""
        if H_concentration:
            return -np.log10(H_concentration)
        elif OH_concentration:
            pOH = -np.log10(OH_concentration)
            return 14 - pOH
        else:
            raise ValueError("H+ veya OH- konsantrasyonu gerekli")
    
    @staticmethod
    def pKa_to_Ka(pKa: float) -> float:
        """pKa'dan Ka hesapla"""
        return 10**(-pKa)
    
    @staticmethod
    def henderson_hasselbalch(pKa: float, ratio_base_acid: float) -> float:
        """Henderson-Hasselbalch denklemi"""
        return pKa + np.log10(ratio_base_acid)
    
    @staticmethod
    def ideal_gas_law(P: float = None, V: float = None, n: float = None, T: float = None) -> Dict:
        """İdeal gaz yasası: PV = nRT"""
        R = Constants.R_GAS
        
        if P and V and n:  # T'yi bul
            T = (P * V) / (n * R)
            return {'T': T}
        elif P and V and T:  # n'yi bul
            n = (P * V) / (R * T)
            return {'n': n}
        elif P and n and T:  # V'yi bul
            V = (n * R * T) / P
            return {'V': V}
        elif V and n and T:  # P'yi bul
            P = (n * R * T) / V
            return {'P': P}
        else:
            raise ValueError("3 değişken verilmeli")

# ==================== TERMODİNAMİK ====================
class Thermodynamics:
    """Termodinamik hesaplamalar"""
    
    @staticmethod
    def gibbs_free_energy(dH: float, T: float, dS: float) -> float:
        """Gibbs serbest enerjisi: ΔG = ΔH - TΔS"""
        return dH - T * dS
    
    @staticmethod
    def equilibrium_constant(dG: float, T: float) -> float:
        """Denge sabiti: ΔG = -RT ln(K)"""
        R = Constants.R_GAS
        return np.exp(-dG / (R * T))
    
    @staticmethod
    def vant_hoff(K1: float, K2: float, T1: float, T2: float) -> float:
        """Van't Hoff denklemi ile ΔH hesapla"""
        R = Constants.R_GAS
        dH = -R * np.log(K2/K1) / (1/T2 - 1/T1)
        return dH
    
    @staticmethod
    def arrhenius(A: float = None, Ea: float = None, T: float = None, k: float = None) -> Dict:
        """Arrhenius denklemi: k = A * exp(-Ea/RT)"""
        R = Constants.R_GAS
        
        if A and Ea and T:  # k'yı bul
            k = A * np.exp(-Ea / (R * T))
            return {'k': k}
        elif k and Ea and T:  # A'yı bul
            A = k / np.exp(-Ea / (R * T))
            return {'A': A}
        elif k and A and T:  # Ea'yı bul
            Ea = -R * T * np.log(k / A)
            return {'Ea': Ea}
        else:
            raise ValueError("Uygun parametreler verilmeli")
    
    @staticmethod
    def clausius_clapeyron(P1: float, P2: float, T1: float, T2: float) -> float:
        """Clausius-Clapeyron denklemi ile ΔHvap hesapla"""
        R = Constants.R_GAS
        dHvap = -R * np.log(P2/P1) / (1/T2 - 1/T1)
        return dHvap

# ==================== KİNETİK ====================
class Kinetics:
    """Kimyasal kinetik hesaplamalar"""
    
    @staticmethod
    def rate_law(k: float, concentrations: List[float], orders: List[float]) -> float:
        """Hız yasası: rate = k[A]^m[B]^n..."""
        rate = k
        for conc, order in zip(concentrations, orders):
            rate *= conc**order
        return rate
    
    @staticmethod
    def half_life(k: float, order: int, initial_conc: float = None) -> float:
        """Yarı ömür hesaplama"""
        if order == 0:
            if initial_conc is None:
                raise ValueError("0. derece için başlangıç konsantrasyonu gerekli")
            return initial_conc / (2 * k)
        elif order == 1:
            return np.log(2) / k
        elif order == 2:
            if initial_conc is None:
                raise ValueError("2. derece için başlangıç konsantrasyonu gerekli")
            return 1 / (k * initial_conc)
        else:
            raise ValueError("Desteklenmeyen reaksiyon derecesi")
    
    @staticmethod
    def integrated_rate_law(order: int, k: float, t: float, C0: float) -> float:
        """Entegre hız yasası"""
        if order == 0:
            return C0 - k * t
        elif order == 1:
            return C0 * np.exp(-k * t)
        elif order == 2:
            return 1 / (1/C0 + k * t)
        else:
            raise ValueError("Desteklenmeyen reaksiyon derecesi")

# ==================== KUANTUM KİMYA ====================
class QuantumChemistry:
    """Kuantum kimya hesaplamaları"""
    
    @staticmethod
    def energy_photon(wavelength: float = None, frequency: float = None) -> float:
        """Foton enerjisi: E = hν = hc/λ"""
        h = Constants.PLANCK
        c = Constants.LIGHT_SPEED
        
        if frequency:
            return h * frequency
        elif wavelength:
            return h * c / wavelength
        else:
            raise ValueError("Dalga boyu veya frekans gerekli")
    
    @staticmethod
    def de_broglie_wavelength(mass: float, velocity: float) -> float:
        """De Broglie dalga boyu: λ = h/mv"""
        h = Constants.PLANCK
        return h / (mass * velocity)
    
    @staticmethod
    def rydberg_equation(n1: int, n2: int) -> float:
        """Rydberg denklemi (Hidrojen için dalga boyu)"""
        R_H = 1.097e7  # m^-1
        if n2 <= n1:
            raise ValueError("n2 > n1 olmalı")
        
        wavelength_inv = R_H * (1/n1**2 - 1/n2**2)
        return 1 / wavelength_inv
    
    @staticmethod
    def heisenberg_uncertainty(dx: float = None, dp: float = None) -> float:
        """Heisenberg belirsizlik ilkesi: ΔxΔp ≥ ℏ/2"""
        hbar = Constants.PLANCK / (2 * np.pi)
        
        if dx:
            return hbar / (2 * dx)  # Minimum dp
        elif dp:
            return hbar / (2 * dp)  # Minimum dx
        else:
            raise ValueError("dx veya dp gerekli")

# ==================== SPEKTROSKOPI ====================
class Spectroscopy:
    """Spektroskopi veri analizi"""
    
    @staticmethod
    def beer_lambert_law(A: float = None, epsilon: float = None, 
                        b: float = None, c: float = None) -> Dict:
        """Beer-Lambert Yasası: A = εbc"""
        if A is None and all(v is not None for v in [epsilon, b, c]):
            A = epsilon * b * c
            return {'A': A}
        elif epsilon is None and all(v is not None for v in [A, b, c]):
            epsilon = A / (b * c)
            return {'epsilon': epsilon}
        elif c is None and all(v is not None for v in [A, epsilon, b]):
            c = A / (epsilon * b)
            return {'c': c}
        else:
            raise ValueError("3 parametre verilmeli")
    
    @staticmethod
    def process_spectrum(wavelengths: np.ndarray, intensities: np.ndarray) -> Dict:
        """Spektrum verisi işleme"""
        # Normalize et
        norm_intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        
        # Pikleri bul
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(norm_intensities, height=0.1, distance=5)
        
        return {
            'normalized': norm_intensities,
            'peaks': peaks,
            'peak_wavelengths': wavelengths[peaks],
            'peak_intensities': norm_intensities[peaks]
        }
    
    @staticmethod
    def baseline_correction(spectrum: np.ndarray, method: str = 'polynomial', degree: int = 3) -> np.ndarray:
        """Baseline düzeltme"""
        from scipy import signal
        
        if method == 'polynomial':
            # Polynomial fitting
            x = np.arange(len(spectrum))
            coeffs = np.polyfit(x, spectrum, degree)
            baseline = np.polyval(coeffs, x)
            return spectrum - baseline
        elif method == 'median':
            # Median filter
            baseline = signal.medfilt(spectrum, kernel_size=21)
            return spectrum - baseline
        else:
            raise ValueError("Desteklenmeyen metod")

# ==================== ELEKTROKİMYA ====================
class Electrochemistry:
    """Elektrokimya hesaplamaları"""
    
    @staticmethod
    def nernst_equation(E0: float, n: float, Q: float, T: float = 298.15) -> float:
        """Nernst denklemi: E = E° - (RT/nF)lnQ"""
        R = Constants.R_GAS
        F = Constants.FARADAY
        
        E = E0 - (R * T / (n * F)) * np.log(Q)
        return E
    
    @staticmethod
    def faraday_law(charge: float = None, moles: float = None, n: float = None) -> Dict:
        """Faraday yasası: Q = nFm"""
        F = Constants.FARADAY
        
        if charge is None and moles and n:
            charge = n * F * moles
            return {'charge': charge}
        elif moles is None and charge and n:
            moles = charge / (n * F)
            return {'moles': moles}
        else:
            raise ValueError("Uygun parametreler verilmeli")
    
    @staticmethod
    def cell_potential(E_cathode: float, E_anode: float) -> float:
        """Hücre potansiyeli"""
        return E_cathode - E_anode

# ==================== MAKİNE ÖĞRENMESİ İÇİN VERİ İŞLEME ====================
class ChemMLPreprocessor:
    """Kimyasal veri için ML ön işleme"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def molecular_descriptors(self, molecule: Molecule) -> Dict:
        """Basit moleküler tanımlayıcılar"""
        pt = PeriodicTable()
        
        descriptors = {
            'molecular_weight': molecule.molecular_weight,
            'num_atoms': sum(molecule.composition.values()),
            'num_elements': len(molecule.composition),
        }
        
        # Elektronegatiflik ortalaması
        electroneg = []
        for elem, count in molecule.composition.items():
            elem_data = pt.get_element(elem)
            if elem_data and elem_data['electronegativity']:
                electroneg.extend([elem_data['electronegativity']] * count)
        
        if electroneg:
            descriptors['mean_electronegativity'] = np.mean(electroneg)
            descriptors['std_electronegativity'] = np.std(electroneg)
        
        # H-bağı donor/akseptör sayısı (basit tahmin)
        descriptors['h_bond_donors'] = molecule.composition.get('N', 0) + molecule.composition.get('O', 0)
        descriptors['h_bond_acceptors'] = (molecule.composition.get('N', 0) + 
                                          molecule.composition.get('O', 0) + 
                                          molecule.composition.get('F', 0))
        
        return descriptors
    
    def prepare_dataset(self, molecules: List[Molecule], targets: np.ndarray = None,
                       test_size: float = 0.2, random_state: int = 42) -> Dict:
        """ML için veri seti hazırla"""
        # Özellikleri çıkar
        features_list = []
        for mol in molecules:
            desc = self.molecular_descriptors(mol)
            features_list.append(list(desc.values()))
            if not self.feature_names:
                self.feature_names = list(desc.keys())
        
        X = np.array(features_list)
        
        # Normalize et
        X_scaled = self.scaler.fit_transform(X)
        
        if targets is not None:
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, targets, test_size=test_size, random_state=random_state
            )
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': self.feature_names,
                'scaler': self.scaler
            }
        else:
            return {
                'X': X_scaled,
                'feature_names': self.feature_names,
                'scaler': self.scaler
            }

# ==================== MAKİNE ÖĞRENMESİ MODELLERİ ====================
class ChemML:
    """Kimya için ML modelleri"""
    
    @staticmethod
    def property_prediction_pipeline():
        """Özellik tahmin pipeline'ı"""
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestRegressor
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        return pipeline
    
    @staticmethod
    def classification_pipeline():
        """Sınıflandırma pipeline'ı"""
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        return pipeline
    
    @staticmethod
    def train_model(X_train, y_train, model_type='regression'):
        """Model eğit"""
        if model_type == 'regression':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        return model
    
    @staticmethod
    def evaluate_model(model, X_test, y_test, model_type='regression'):
        """Model değerlendirme"""
        predictions = model.predict(X_test)
        
        if model_type == 'regression':
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': predictions
            }
        else:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': predictions
            }

# ==================== DERİN ÖĞRENME İÇİN HAZIRLIK ====================
class ChemDL:
    """Kimya için derin öğrenme araçları"""
    
    @staticmethod
    def create_molecular_fingerprint(molecule: Molecule, size: int = 1024) -> np.ndarray:
        """Basit moleküler parmak izi (Morgan benzeri)"""
        np.random.seed(hash(molecule.formula) % 2**32)
        
        # Basit bir parmak izi simülasyonu
        fingerprint = np.zeros(size)
        
        # Atomlara ve bağlara dayalı bit ayarlama
        for elem, count in molecule.composition.items():
            # Her element için benzersiz bitler ayarla
            for i in range(count):
                bit_idx = hash(f"{elem}_{i}") % size
                fingerprint[bit_idx] = 1
        
        # Element çiftleri için bitler (basit bağ simülasyonu)
        elements = list(molecule.composition.keys())
        for i in range(len(elements)):
            for j in range(i+1, len(elements)):
                bit_idx = hash(f"{elements[i]}_{elements[j]}") % size
                fingerprint[bit_idx] = 1
        
        return fingerprint
    
    @staticmethod
    def prepare_sequences_for_rnn(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """RNN için sekans hazırlama"""
        sequences = []
        targets = []
        
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
            targets.append(data[i+seq_length])
        
        return np.array(sequences), np.array(targets)
    
    @staticmethod
    def create_simple_nn_architecture(input_dim: int, output_dim: int = 1, 
                                     task: str = 'regression') -> Dict:
        """Basit sinir ağı mimarisi önerisi"""
        architecture = {
            'input_layer': {
                'units': input_dim,
                'activation': None
            },
            'hidden_layers': [
                {'units': 128, 'activation': 'relu', 'dropout': 0.2},
                {'units': 64, 'activation': 'relu', 'dropout': 0.2},
                {'units': 32, 'activation': 'relu', 'dropout': 0.1}
            ],
            'output_layer': {
                'units': output_dim,
                'activation': 'linear' if task == 'regression' else 'softmax'
            },
            'optimizer': 'adam',
            'loss': 'mse' if task == 'regression' else 'categorical_crossentropy',
            'metrics': ['mae'] if task == 'regression' else ['accuracy']
        }
        
        return architecture
    
    @staticmethod
    def augment_molecular_data(molecules: List[Molecule], augmentation_factor: int = 2) -> List[Molecule]:
        """Moleküler veri artırma (basit)"""
        augmented = []
        
        for mol in molecules:
            augmented.append(mol)
            
            # İzomer simülasyonu (aynı formül, farklı isim)
            for i in range(augmentation_factor - 1):
                new_mol = Molecule(
                    formula=mol.formula,
                    name=f"{mol.name}_isomer_{i}" if mol.name else f"isomer_{i}"
                )
                augmented.append(new_mol)
        
        return augmented

# ==================== VİZÜALİZASYON ====================
class ChemVisualization:
    """Kimyasal veri görselleştirme"""
    
    @staticmethod
    def plot_spectrum(wavelengths: np.ndarray, intensities: np.ndarray, 
                     title: str = "Spektrum", peaks: np.ndarray = None):
        """Spektrum grafiği"""
        plt.figure(figsize=(10, 6))
        plt.plot(wavelengths, intensities, 'b-', linewidth=1.5)
        
        if peaks is not None:
            plt.plot(wavelengths[peaks], intensities[peaks], 'ro', markersize=8)
            for peak in peaks:
                plt.annotate(f'{wavelengths[peak]:.1f}', 
                           xy=(wavelengths[peak], intensities[peak]),
                           xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Dalga Boyu (nm)')
        plt.ylabel('Yoğunluk')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_reaction_profile(reaction_coordinate: np.ndarray, energy: np.ndarray,
                             labels: List[str] = None):
        """Reaksiyon profili"""
        plt.figure(figsize=(10, 6))
        plt.plot(reaction_coordinate, energy, 'b-', linewidth=2)
        plt.plot(reaction_coordinate, energy, 'ro', markersize=8)
        
        if labels:
            for i, label in enumerate(labels):
                plt.annotate(label, xy=(reaction_coordinate[i], energy[i]),
                           xytext=(0, 10), textcoords='offset points', ha='center')
        
        plt.xlabel('Reaksiyon Koordinatı')
        plt.ylabel('Enerji (kJ/mol)')
        plt.title('Reaksiyon Enerji Profili')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_titration_curve(volume: np.ndarray, pH: np.ndarray):
        """Titrasyon eğrisi"""
        plt.figure(figsize=(10, 6))
        plt.plot(volume, pH, 'b-', linewidth=2)
        
        # Eşdeğerlik noktasını bul (en dik nokta)
        dpH = np.gradient(pH)
        eq_point_idx = np.argmax(np.abs(dpH))
        
        plt.plot(volume[eq_point_idx], pH[eq_point_idx], 'ro', markersize=10)
        plt.annotate(f'Eşdeğerlik Noktası\nV={volume[eq_point_idx]:.1f} mL\npH={pH[eq_point_idx]:.2f}',
                    xy=(volume[eq_point_idx], pH[eq_point_idx]),
                    xytext=(10, 10), textcoords='offset points')
        
        plt.xlabel('Eklenen Hacim (mL)')
        plt.ylabel('pH')
        plt.title('Titrasyon Eğrisi')
        plt.grid(True, alpha=0.3)
        plt.show()

# ==================== DENEYSEL VERİ ANALİZİ ====================
class ExperimentalAnalysis:
    """Deneysel veri analizi"""
    
    @staticmethod
    def linear_regression(x: np.ndarray, y: np.ndarray) -> Dict:
        """Lineer regresyon analizi"""
        from scipy import stats
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Tahmin değerleri
        y_pred = slope * x + intercept
        
        # Artıklar
        residuals = y - y_pred
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_error': std_err,
            'equation': f"y = {slope:.4f}x + {intercept:.4f}",
            'predictions': y_pred,
            'residuals': residuals
        }
    
    @staticmethod
    def error_propagation(values: List[float], errors: List[float], 
                         operation: str) -> Tuple[float, float]:
        """Hata yayılımı hesaplama"""
        values = np.array(values)
        errors = np.array(errors)
        
        if operation == 'sum':
            result = np.sum(values)
            error = np.sqrt(np.sum(errors**2))
        elif operation == 'difference':
            result = values[0] - values[1]
            error = np.sqrt(errors[0]**2 + errors[1]**2)
        elif operation == 'product':
            result = np.prod(values)
            relative_errors = errors / values
            error = result * np.sqrt(np.sum(relative_errors**2))
        elif operation == 'division':
            result = values[0] / values[1]
            relative_errors = errors / values
            error = result * np.sqrt(relative_errors[0]**2 + relative_errors[1]**2)
        else:
            raise ValueError("Desteklenmeyen işlem")
        
        return result, error
    
    @staticmethod
    def statistical_analysis(data: np.ndarray) -> Dict:
        """İstatistiksel analiz"""
        from scipy import stats
        
        return {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data, ddof=1),
            'variance': np.var(data, ddof=1),
            'sem': stats.sem(data),
            'min': np.min(data),
            'max': np.max(data),
            'q1': np.percentile(data, 25),
            'q3': np.percentile(data, 75),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'cv': np.std(data, ddof=1) / np.mean(data) * 100  # Coefficient of variation
        }

# ==================== ÇÖZELTILER ====================
class Solutions:
    """Çözelti hesaplamaları"""
    
    @staticmethod
    def colligative_properties(molality: float, i: float = 1) -> Dict:
        """Kolligatif özellikler (i = van't Hoff faktörü)"""
        # Sabitler
        Kf_water = 1.86  # °C·kg/mol (suyun donma noktası alçalma sabiti)
        Kb_water = 0.512  # °C·kg/mol (suyun kaynama noktası yükselme sabiti)
        
        # Donma noktası alçalması
        delta_Tf = i * Kf_water * molality
        
        # Kaynama noktası yükselmesi
        delta_Tb = i * Kb_water * molality
        
        # Ozmotik basınç (25°C için)
        R = 0.08206  # L·atm/(mol·K)
        T = 298.15  # K
        osmotic_pressure = i * molality * R * T
        
        return {
            'freezing_point_depression': delta_Tf,
            'boiling_point_elevation': delta_Tb,
            'osmotic_pressure': osmotic_pressure,
            'new_freezing_point': 0 - delta_Tf,
            'new_boiling_point': 100 + delta_Tb
        }
    
    @staticmethod
    def buffer_capacity(buffer_conc: float, volume: float) -> float:
        """Tampon kapasitesi"""
        return buffer_conc * volume
    
    @staticmethod
    def ionic_strength(concentrations: List[float], charges: List[float]) -> float:
        """İyonik güç: I = 0.5 * Σ(ci * zi^2)"""
        I = 0.5 * sum(c * z**2 for c, z in zip(concentrations, charges))
        return I

# ==================== KİMYASAL DENGE ====================
class ChemicalEquilibrium:
    """Kimyasal denge hesaplamaları"""
    
    @staticmethod
    def ice_table(initial: Dict, change: Dict, equilibrium: Dict = None) -> pd.DataFrame:
        """ICE tablosu oluştur"""
        species = list(initial.keys())
        
        if equilibrium is None:
            equilibrium = {s: initial[s] + change[s] for s in species}
        
        df = pd.DataFrame({
            'Initial': [initial[s] for s in species],
            'Change': [change[s] for s in species],
            'Equilibrium': [equilibrium[s] for s in species]
        }, index=species)
        
        return df
    
    @staticmethod
    def quadratic_equilibrium(K: float, initial_conc: float) -> float:
        """Kuadratik denge problemi çözücü"""
        # Örnek: A ⇌ B + C için
        # K = x^2 / (initial_conc - x)
        
        a = 1
        b = K
        c = -K * initial_conc
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            raise ValueError("Negatif diskriminant")
        
        x1 = (-b + np.sqrt(discriminant)) / (2*a)
        x2 = (-b - np.sqrt(discriminant)) / (2*a)
        
        # Fiziksel olarak anlamlı çözümü seç
        if 0 <= x1 <= initial_conc:
            return x1
        elif 0 <= x2 <= initial_conc:
            return x2
        else:
            raise ValueError("Geçerli çözüm bulunamadı")
    
    @staticmethod
    def le_chatelier(K_initial: float, stress: str, magnitude: float = None) -> str:
        """Le Chatelier prensibi"""
        predictions = {
            'temperature_increase': "Endotermik reaksiyon: İleri yön\nEkzotermik reaksiyon: Geri yön",
            'temperature_decrease': "Endotermik reaksiyon: Geri yön\nEkzotermik reaksiyon: İleri yön",
            'pressure_increase': "Daha az gaz mol sayısına doğru kayar",
            'pressure_decrease': "Daha fazla gaz mol sayısına doğru kayar",
            'concentration_increase': "Eklenen maddeyi tüketme yönünde",
            'concentration_decrease': "Eksilen maddeyi üretme yönünde"
        }
        
        return predictions.get(stress, "Belirtilen stres tanımlanmadı")

# ==================== YARDIMCI FONKSİYONLAR ====================
class Utils:
    """Yardımcı fonksiyonlar"""
    
    @staticmethod
    def significant_figures(value: float, sig_figs: int) -> float:
        """Anlamlı basamaklar"""
        from math import log10, floor
        
        if value == 0:
            return 0
        
        return round(value, -int(floor(log10(abs(value)))) + (sig_figs - 1))
    
    @staticmethod
    def scientific_notation(value: float) -> str:
        """Bilimsel notasyon"""
        return f"{value:.2e}"
    
    @staticmethod
    def unit_converter(value: float, from_unit: str, to_unit: str) -> float:
        """Birim dönüştürücü"""
        # Basit dönüşümler
        conversions = {
            ('g', 'kg'): 0.001,
            ('kg', 'g'): 1000,
            ('mL', 'L'): 0.001,
            ('L', 'mL'): 1000,
            ('atm', 'Pa'): 101325,
            ('Pa', 'atm'): 1/101325,
            ('C', 'K'): lambda x: x + 273.15,
            ('K', 'C'): lambda x: x - 273.15,
            ('cal', 'J'): 4.184,
            ('J', 'cal'): 1/4.184,
        }
        
        key = (from_unit, to_unit)
        if key in conversions:
            if callable(conversions[key]):
                return conversions[key](value)
            else:
                return value * conversions[key]
        else:
            raise ValueError(f"Dönüşüm desteklenmiyor: {from_unit} -> {to_unit}")

# ==================== ANA SINIF ====================
class ChemPyLab:
    """Ana kütüphane sınıfı"""
    
    def __init__(self):
        self.periodic_table = PeriodicTable()
        self.basic_chem = BasicChemistry()
        self.thermo = Thermodynamics()
        self.kinetics = Kinetics()
        self.quantum = QuantumChemistry()
        self.spectro = Spectroscopy()
        self.electro = Electrochemistry()
        self.ml_preprocessor = ChemMLPreprocessor()
        self.ml = ChemML()
        self.dl = ChemDL()
        self.viz = ChemVisualization()
        self.exp_analysis = ExperimentalAnalysis()
        self.solutions = Solutions()
        self.equilibrium = ChemicalEquilibrium()
        self.utils = Utils()
        self.constants = Constants()
    
    def quick_analysis(self, formula: str) -> Dict:
        """Hızlı molekül analizi"""
        mol = Molecule(formula)
        descriptors = self.ml_preprocessor.molecular_descriptors(mol)
        
        return {
            'formula': formula,
            'molecular_weight': mol.molecular_weight,
            'composition': mol.composition,
            'empirical_formula': mol.empirical_formula(),
            'descriptors': descriptors
        }
    
    def demo(self):
        """Kütüphane demo"""
        print("=== ChemPyLab Demo ===\n")
        
        # 1. Molekül analizi
        print("1. Molekül Analizi:")
        mol = Molecule("C6H12O6", name="Glukoz")
        print(f"   Formül: {mol.formula}")
        print(f"   Molekül Ağırlığı: {mol.molecular_weight:.2f} g/mol")
        print(f"   Empirik Formül: {mol.empirical_formula()}\n")
        
        # 2. pH hesaplama
        print("2. pH Hesaplama:")
        pH_value = self.basic_chem.pH(H_concentration=1e-5)
        print(f"   [H+] = 1e-5 M için pH = {pH_value:.2f}\n")
        
        # 3. İdeal gaz yasası
        print("3. İdeal Gaz Yasası:")
        gas_result = self.basic_chem.ideal_gas_law(P=1, V=22.4, T=273.15)
        print(f"   P=1 atm, V=22.4 L, T=273.15 K için n = {gas_result['n']:.2f} mol\n")
        
        # 4. Gibbs enerjisi
        print("4. Termodinamik:")
        dG = self.thermo.gibbs_free_energy(dH=-285.8, T=298, dS=-0.1636)
        print(f"   ΔG = {dG:.2f} kJ/mol\n")
        
        # 5. Kinetik
        print("5. Kinetik:")
        t_half = self.kinetics.half_life(k=0.693, order=1)
        print(f"   k=0.693 s⁻¹ için yarı ömür = {t_half:.2f} s\n")
        
        # 6. ML descriptor
        print("6. ML Tanımlayıcılar:")
        desc = self.ml_preprocessor.molecular_descriptors(mol)
        print(f"   Atom sayısı: {desc['num_atoms']}")
        print(f"   H-bağı donörleri: {desc['h_bond_donors']}\n")
        
        print("Demo tamamlandı! Detaylı kullanım için dokümantasyona bakın.")

# Kullanım örneği
if __name__ == "__main__":
    # Kütüphaneyi başlat
    chem = ChemPyLab()
    
    # Demo çalıştır
    chem.demo()
    
    print("\n=== Örnek Kullanımlar ===\n")
    
    # Örnek 1: Çözelti hazırlama
    print("Örnek 1: 0.1 M 500 mL NaCl çözeltisi hazırlama")
    mol_weight = chem.periodic_table.get_mass('Na') + chem.periodic_table.get_mass('Cl')
    moles_needed = chem.basic_chem.molarity(0.1, 0.5) * 0.5
    mass_needed = moles_needed * mol_weight
    print(f"Gerekli NaCl miktarı: {mass_needed:.2f} g\n")
    
    # Örnek 2: Spektrum analizi
    print("Örnek 2: Spektrum verisi işleme")
    wavelengths = np.linspace(200, 800, 100)
    intensities = np.exp(-(wavelengths - 400)**2 / 5000) + 0.5 * np.exp(-(wavelengths - 550)**2 / 3000)
    spectrum_data = chem.spectro.process_spectrum(wavelengths, intensities)
    print(f"Bulunan pik sayısı: {len(spectrum_data['peaks'])}")
    print(f"Pik dalga boyları: {spectrum_data['peak_wavelengths'][:3]}... nm\n")
    
    # Örnek 3: ML için veri hazırlama
    print("Örnek 3: ML için moleküler veri hazırlama")
    molecules = [
        Molecule("CH4", name="Metan"),
        Molecule("C2H6", name="Etan"),
        Molecule("C3H8", name="Propan"),
        Molecule("C4H10", name="Bütan")
    ]
    
    # Kaynama noktaları (°C) - örnek hedef değişken
    boiling_points = np.array([-161.5, -88.6, -42.1, -0.5])
    
    dataset = chem.ml_preprocessor.prepare_dataset(molecules, boiling_points, test_size=0.25)
    print(f"Eğitim seti boyutu: {dataset['X_train'].shape}")
    print(f"Özellik sayısı: {len(dataset['feature_names'])}")
    print(f"Özellikler: {dataset['feature_names']}") 