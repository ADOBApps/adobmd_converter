"""
SDF (Structure Data File) parser for ADOBMD Converter
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np

class SDFAtom:
    """Represents an atom in SDF format"""
    
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.element = ""
        self.mass_diff = 0
        self.charge = 0
        self.stereo_parity = 0
        self.hydrogen_count = 0
        self.stereo_care = 0
        self.valence = 0
        self.mass = 0.0

class SDFBond:
    """Represents a bond in SDF format"""
    
    def __init__(self, atom1: int, atom2: int, bond_type: int = 1):
        self.atom1 = atom1
        self.atom2 = atom2
        self.bond_type = bond_type  # 1=single, 2=double, 3=triple, 4=aromatic
        self.stereo = 0

class SDFProperty:
    """Represents a property in SDF format"""
    
    def __init__(self, name: str, value: str):
        self.name = name
        self.value = value

class SDFParser:
    """Parser for Structure Data File (SDF) format"""
    
    # Atomic masses (same as PDB parser)
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
    
    def __init__(self):
        self.molecules: List[Dict] = []
        self.current_mol = None
    
    def parse(self, filename: str) -> bool:
        """Parse SDF file (may contain multiple molecules)"""
        try:
            with open(filename, 'r') as f:
                content = f.readlines()
            
            i = 0
            while i < len(content):
                mol_data = self._parse_molecule(content, i)
                if mol_data:
                    self.molecules.append(mol_data)
                    i = mol_data['end_line']
                else:
                    i += 1
            
            return len(self.molecules) > 0
            
        except Exception as e:
            print(f"Error parsing SDF file: {e}")
            return False
    
    def _parse_molecule(self, lines: List[str], start: int) -> Optional[Dict]:
        """Parse a single molecule from the file"""
        try:
            if start >= len(lines):
                return None
            
            # Header lines
            if len(lines[start].strip()) == 0:
                return None
            
            mol = {
                'name': lines[start].strip(),
                'program': lines[start + 1].strip() if start + 1 < len(lines) else '',
                'comment': lines[start + 2].strip() if start + 2 < len(lines) else '',
                'atoms': [],
                'bonds': [],
                'properties': []
            }
            
            # Counts line
            if start + 3 >= len(lines):
                return None
            
            counts_line = lines[start + 3]
            n_atoms = int(counts_line[0:3])
            n_bonds = int(counts_line[3:6])
            
            # Parse atoms
            atom_start = start + 4
            for i in range(n_atoms):
                line_idx = atom_start + i
                if line_idx >= len(lines):
                    break
                
                atom = self._parse_atom_line(lines[line_idx])
                if atom:
                    mol['atoms'].append(atom)
            
            # Parse bonds
            bond_start = atom_start + n_atoms
            for i in range(n_bonds):
                line_idx = bond_start + i
                if line_idx >= len(lines):
                    break
                
                bond = self._parse_bond_line(lines[line_idx])
                if bond:
                    mol['bonds'].append(bond)
            
            # Find end of molecule ($$$$)
            current = bond_start + n_bonds
            while current < len(lines):
                if lines[current].strip() == '$$$$':
                    mol['end_line'] = current + 1
                    break
                
                # Parse properties (M lines)
                if lines[current].startswith('> '):
                    prop = self._parse_property(lines, current)
                    if prop:
                        mol['properties'].append(prop)
                        current = prop['end_line']
                    else:
                        current += 1
                else:
                    current += 1
            
            return mol
            
        except Exception as e:
            print(f"Error parsing molecule: {e}")
            return None
    
    def _parse_atom_line(self, line: str) -> Optional[SDFAtom]:
        """Parse atom line in SDF format"""
        try:
            atom = SDFAtom()
            
            # Fixed format: 10.4, 10.4, 10.4, then element symbol, then optional fields
            atom.x = float(line[0:10])
            atom.y = float(line[10:20])
            atom.z = float(line[20:30])
            atom.element = line[31:34].strip().capitalize()
            
            # Optional fields (if present)
            if len(line) >= 39:
                atom.mass_diff = int(line[34:36]) if line[34:36].strip() else 0
                atom.charge = int(line[36:39]) if line[36:39].strip() else 0
            if len(line) >= 42:
                atom.stereo_parity = int(line[39:42]) if line[39:42].strip() else 0
            if len(line) >= 45:
                atom.hydrogen_count = int(line[42:45]) if line[42:45].strip() else 0
            if len(line) >= 48:
                atom.stereo_care = int(line[45:48]) if line[45:48].strip() else 0
            if len(line) >= 51:
                atom.valence = int(line[48:51]) if line[48:51].strip() else 0
            
            # Set mass
            atom.mass = self.ATOMIC_MASSES.get(atom.element, 12.01)
            
            return atom
            
        except Exception as e:
            print(f"Error parsing atom line: {e}")
            return None
    
    def _parse_bond_line(self, line: str) -> Optional[SDFBond]:
        """Parse bond line in SDF format"""
        try:
            bond = SDFBond(
                atom1=int(line[0:3]),
                atom2=int(line[3:6]),
                bond_type=int(line[6:9])
            )
            
            if len(line) >= 12:
                bond.stereo = int(line[9:12]) if line[9:12].strip() else 0
            
            return bond
            
        except Exception as e:
            print(f"Error parsing bond line: {e}")
            return None
    
    def _parse_property(self, lines: List[str], start: int) -> Optional[Dict]:
        """Parse property lines starting with '> '"""
        try:
            header = lines[start].strip()
            
            # Extract property name
            name_match = re.search(r'<(.+)>', header)
            name = name_match.group(1) if name_match else 'unknown'
            
            # Read value lines until empty line
            value_lines = []
            i = start + 1
            while i < len(lines) and lines[i].strip():
                value_lines.append(lines[i].strip())
                i += 1
            
            value = '\n'.join(value_lines)
            
            return {
                'name': name,
                'value': value,
                'end_line': i + 1  # Skip the empty line
            }
            
        except Exception as e:
            print(f"Error parsing property: {e}")
            return None
    
    def get_first_molecule(self) -> Optional[Dict]:
        """Get the first molecule in the file"""
        return self.molecules[0] if self.molecules else None
    
    def get_all_molecules(self) -> List[Dict]:
        """Get all molecules"""
        return self.molecules