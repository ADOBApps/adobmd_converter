"""
PDB file parser for ADOBMD Converter
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np

class PDBAtom:
    """Represents an atom in PDB format"""
    
    def __init__(self):
        self.serial = 0
        self.name = ""
        self.alt_loc = ""
        self.res_name = ""
        self.chain_id = ""
        self.res_seq = 0
        self.i_code = ""
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.occupancy = 1.0
        self.temp_factor = 0.0
        self.element = ""
        self.charge = ""
        self.mass = 0.0

class PDBBond:
    """Represents a bond between atoms"""
    
    def __init__(self, atom1: int, atom2: int, bond_type: int = 1):
        self.atom1 = atom1
        self.atom2 = atom2
        self.bond_type = bond_type  # 1=single, 2=double, 3=triple

class PDBParser:
    """Parser for Protein Data Bank (PDB) files"""
    
    # Atomic masses (amu)
    ATOMIC_MASSES = {
        'H': 1.008, 'HE': 4.0026, 'LI': 6.94, 'BE': 9.0122, 'B': 10.81,
        'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'NE': 20.180,
        'NA': 22.990, 'MG': 24.305, 'AL': 26.982, 'SI': 28.085, 'P': 30.974,
        'S': 32.06, 'CL': 35.45, 'AR': 39.95, 'K': 39.098, 'CA': 40.078,
        'SC': 44.956, 'TI': 47.867, 'V': 50.942, 'CR': 51.996, 'MN': 54.938,
        'FE': 55.845, 'CO': 58.933, 'NI': 58.693, 'CU': 63.546, 'ZN': 65.38,
        'GA': 69.723, 'GE': 72.630, 'AS': 74.922, 'SE': 78.96, 'BR': 79.904,
        'KR': 83.798, 'RB': 85.468, 'SR': 87.62, 'Y': 88.906, 'ZR': 91.224,
        'NB': 92.906, 'MO': 95.95, 'TC': 98.0, 'RU': 101.07, 'RH': 102.91,
        'PD': 106.42, 'AG': 107.87, 'CD': 112.41, 'IN': 114.82, 'SN': 118.71,
        'SB': 121.76, 'TE': 127.60, 'I': 126.90, 'XE': 131.29, 'CS': 132.91,
        'BA': 137.33, 'LA': 138.91, 'CE': 140.12, 'PR': 140.91, 'ND': 144.24,
        'PM': 145.0, 'SM': 150.36, 'EU': 151.96, 'GD': 157.25, 'TB': 158.93,
        'DY': 162.50, 'HO': 164.93, 'ER': 167.26, 'TM': 168.93, 'YB': 173.05,
        'LU': 174.97, 'HF': 178.49, 'TA': 180.95, 'W': 183.84, 'RE': 186.21,
        'OS': 190.23, 'IR': 192.22, 'PT': 195.08, 'AU': 196.97, 'HG': 200.59,
        'TL': 204.38, 'PB': 207.2, 'BI': 208.98, 'PO': 209.0, 'AT': 210.0,
        'RN': 222.0, 'FR': 223.0, 'RA': 226.0, 'AC': 227.0, 'TH': 232.04,
        'PA': 231.04, 'U': 238.03, 'NP': 237.0, 'PU': 244.0, 'AM': 243.0,
        'CM': 247.0, 'BK': 247.0, 'CF': 251.0, 'ES': 252.0, 'FM': 257.0,
        'MD': 258.0, 'NO': 259.0, 'LR': 262.0
    }
    
    # Covalent radii (Angstroms) for bond detection
    COVALENT_RADII = {
        'H': 0.31, 'HE': 0.28, 'LI': 1.28, 'BE': 0.96, 'B': 0.84,
        'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'NE': 0.58,
        'NA': 1.66, 'MG': 1.41, 'AL': 1.21, 'SI': 1.11, 'P': 1.07,
        'S': 1.05, 'CL': 1.02, 'AR': 1.06, 'K': 2.03, 'CA': 1.76,
        'SC': 1.70, 'TI': 1.60, 'V': 1.53, 'CR': 1.39, 'MN': 1.39,
        'FE': 1.32, 'CO': 1.26, 'NI': 1.24, 'CU': 1.32, 'ZN': 1.22,
        'GA': 1.22, 'GE': 1.20, 'AS': 1.19, 'SE': 1.20, 'BR': 1.20,
        'KR': 1.16, 'RB': 2.20, 'SR': 1.95, 'Y': 1.90, 'ZR': 1.75,
        'NB': 1.64, 'MO': 1.54, 'TC': 1.47, 'RU': 1.46, 'RH': 1.42,
        'PD': 1.39, 'AG': 1.45, 'CD': 1.44, 'IN': 1.42, 'SN': 1.39,
        'SB': 1.39, 'TE': 1.38, 'I': 1.39, 'XE': 1.40, 'CS': 2.44,
        'BA': 2.15, 'LA': 2.07, 'CE': 2.04, 'PR': 2.03, 'ND': 2.01,
        'PM': 1.99, 'SM': 1.98, 'EU': 1.98, 'GD': 1.96, 'TB': 1.94,
        'DY': 1.92, 'HO': 1.92, 'ER': 1.89, 'TM': 1.90, 'YB': 1.87,
        'LU': 1.87, 'HF': 1.75, 'TA': 1.70, 'W': 1.62, 'RE': 1.51,
        'OS': 1.44, 'IR': 1.41, 'PT': 1.36, 'AU': 1.36, 'HG': 1.32,
        'TL': 1.45, 'PB': 1.46, 'BI': 1.48, 'PO': 1.40, 'AT': 1.50,
        'RN': 1.50, 'FR': 2.60, 'RA': 2.21, 'AC': 2.15, 'TH': 2.06,
        'PA': 2.00, 'U': 1.96, 'NP': 1.90, 'PU': 1.87, 'AM': 1.80,
        'CM': 1.69
    }
    
    def __init__(self, bond_cutoff_factor: float = 1.2):
        self.atoms: List[PDBAtom] = []
        self.bonds: List[PDBBond] = []
        self.title = ""
        self.bond_cutoff_factor = bond_cutoff_factor
    
    def parse(self, filename: str) -> bool:
        """Parse PDB file"""
        try:
            with open(filename, 'r') as f:
                for line in f:
                    record = line[0:6].strip()
                    
                    if record == 'TITLE':
                        self.title = line[10:].strip()
                    
                    elif record == 'ATOM' or record == 'HETATM':
                        atom = self._parse_atom_line(line)
                        if atom:
                            self.atoms.append(atom)
                    
                    elif record == 'CONECT':
                        self._parse_conect_line(line)
                    
                    elif record == 'END':
                        break
            
            # Auto-detect bonds if not present in file
            if not self.bonds:
                self._detect_bonds()
            
            return True
            
        except Exception as e:
            print(f"Error parsing PDB file: {e}")
            return False
    
    def _parse_atom_line(self, line: str) -> Optional[PDBAtom]:
        """Parse ATOM/HETATM line"""
        try:
            atom = PDBAtom()
            
            # Basic fields (fixed format)
            atom.serial = int(line[6:11])
            atom.name = line[12:16].strip()
            atom.alt_loc = line[16:17].strip()
            atom.res_name = line[17:20].strip()
            atom.chain_id = line[21:22].strip()
            atom.res_seq = int(line[22:26])
            atom.i_code = line[26:27].strip()
            atom.x = float(line[30:38])
            atom.y = float(line[38:46])
            atom.z = float(line[46:54])
            atom.occupancy = float(line[54:60]) if line[54:60].strip() else 1.0
            atom.temp_factor = float(line[60:66]) if line[60:66].strip() else 0.0
            
            # Element symbol (from columns 76-78 or from atom name)
            if len(line) >= 78:
                atom.element = line[76:78].strip().upper()
            else:
                # Guess from atom name
                atom.element = re.sub(r'[0-9]', '', atom.name)[:2].upper()
            
            # Set mass
            atom.mass = self.ATOMIC_MASSES.get(atom.element, 12.01)
            
            return atom
            
        except Exception as e:
            print(f"Error parsing atom line: {e}")
            return None
    
    def _parse_conect_line(self, line: str):
        """Parse CONECT bond records"""
        try:
            parts = line.split()
            if len(parts) < 3:
                return
            
            atom1 = int(parts[1])
            for i in range(2, len(parts)):
                atom2 = int(parts[i])
                self.bonds.append(PDBBond(atom1, atom2))
                
        except Exception as e:
            print(f"Error parsing CONECT line: {e}")
    
    def _detect_bonds(self):
        """Auto-detect bonds based on covalent radii"""
        n_atoms = len(self.atoms)
        
        for i in range(n_atoms):
            atom1 = self.atoms[i]
            r1 = self.COVALENT_RADII.get(atom1.element, 1.5)
            
            for j in range(i + 1, n_atoms):
                atom2 = self.atoms[j]
                r2 = self.COVALENT_RADII.get(atom2.element, 1.5)
                
                # Calculate distance
                dx = atom1.x - atom2.x
                dy = atom1.y - atom2.y
                dz = atom1.z - atom2.z
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                # Bond if distance < sum of covalent radii * factor
                cutoff = (r1 + r2) * self.bond_cutoff_factor
                if dist < cutoff:
                    self.bonds.append(PDBBond(i + 1, j + 1))  # 1-based indices
    
    def get_coordinates(self) -> np.ndarray:
        """Get coordinates as numpy array (Angstroms)"""
        coords = np.zeros((len(self.atoms), 3))
        for i, atom in enumerate(self.atoms):
            coords[i, 0] = atom.x
            coords[i, 1] = atom.y
            coords[i, 2] = atom.z
        return coords
    
    def get_elements(self) -> List[str]:
        """Get element symbols"""
        return [atom.element for atom in self.atoms]
    
    def get_masses(self) -> List[float]:
        """Get atomic masses"""
        return [atom.mass for atom in self.atoms]
    
    def get_residues(self) -> List[int]:
        """Get residue numbers for each atom"""
        return [atom.res_seq for atom in self.atoms]
    
    def get_box_size(self) -> Tuple[float, float, float]:
        """Calculate bounding box"""
        coords = self.get_coordinates()
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        size = max_coords - min_coords
        return size[0], size[1], size[2]