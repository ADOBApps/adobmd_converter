"""
XYZ file parser for ADOBMD Converter

This file is part of QAH.
QAH is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

QAH is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Copyright (C) [2026] Acxel David Orozco Baldomero
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np

class XYZAtom:
    """Represents an atom in XYZ format"""
    
    def __init__(self):
        self.element = ""
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.mass = 0.0
        self.index = 0

class XYZParser:
    """Parser for XYZ (Extended XYZ) format"""
    
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
    
    # Covalent radii for bond detection (Angstroms)
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
        self.atoms: List[XYZAtom] = []
        self.bonds: List[Tuple[int, int, int]] = []  # (atom1, atom2, bond_type)
        self.comment = ""
        self.title = ""
        self.bond_cutoff_factor = bond_cutoff_factor
        self.frames: List[Dict] = []  # For multi-frame XYZ files
        self.has_multiple_frames = False
    
    def parse(self, filename: str, frame: int = 0) -> bool:
        """
        Parse XYZ file
        frame: which frame to load (0 for first, -1 for last, or specific index)
        """
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Remove empty lines at start
            while lines and not lines[0].strip():
                lines.pop(0)
            
            if not lines:
                return False
            
            # Check if multi-frame
            i = 0
            frames = []
            while i < len(lines):
                frame_data = self._parse_frame(lines, i)
                if frame_data:
                    frames.append(frame_data)
                    i = frame_data['end_line']
                else:
                    i += 1
            
            self.frames = frames
            self.has_multiple_frames = len(frames) > 1
            
            # Select frame
            if frames:
                if frame == -1:
                    selected_frame = frames[-1]
                else:
                    selected_frame = frames[min(frame, len(frames)-1)]
                
                self.atoms = selected_frame['atoms']
                self.comment = selected_frame['comment']
                self.title = f"Frame {selected_frame['index']}"
                
                # Auto-detect bonds
                self._detect_bonds()
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Error parsing XYZ file: {e}")
            return False
    
    def _parse_frame(self, lines: List[str], start: int) -> Optional[Dict]:
        """Parse a single frame from XYZ file"""
        try:
            if start >= len(lines):
                return None
            
            # Skip empty lines
            while start < len(lines) and not lines[start].strip():
                start += 1
            
            if start >= len(lines):
                return None
            
            # First line: number of atoms
            try:
                natoms = int(lines[start].strip())
            except ValueError:
                return None
            
            # Second line: comment
            comment_line = start + 1
            if comment_line >= len(lines):
                return None
            
            comment = lines[comment_line].strip()
            
            # Parse atoms
            atoms = []
            atom_start = comment_line + 1
            
            for i in range(natoms):
                line_idx = atom_start + i
                if line_idx >= len(lines):
                    break
                
                atom = self._parse_atom_line(lines[line_idx], i)
                if atom:
                    atoms.append(atom)
            
            # Find end of frame
            end_line = atom_start + natoms
            
            return {
                'atoms': atoms,
                'comment': comment,
                'index': len(self.frames),
                'end_line': end_line
            }
            
        except Exception as e:
            print(f"Error parsing frame: {e}")
            return None
    
    def _parse_atom_line(self, line: str, index: int) -> Optional[XYZAtom]:
        """Parse atom line in XYZ format"""
        try:
            parts = line.strip().split()
            if len(parts) < 4:
                return None
            
            atom = XYZAtom()
            atom.index = index + 1  # 1-based index
            
            # Standard XYZ: element x y z
            atom.element = parts[0].capitalize()
            
            # Handle 2-letter elements
            if len(atom.element) > 2:
                atom.element = atom.element[:2]
            
            atom.x = float(parts[1])
            atom.y = float(parts[2])
            atom.z = float(parts[3])
            
            # Optional fields (extended XYZ format)
            if len(parts) >= 5:
                # Could be mass, charge, etc.
                try:
                    atom.mass = float(parts[4])
                except ValueError:
                    pass
            
            # Set mass from database if not provided
            if atom.mass == 0.0:
                atom.mass = self.ATOMIC_MASSES.get(atom.element, 12.01)
            
            return atom
            
        except Exception as e:
            print(f"Error parsing atom line: {e}")
            return None
    
    def _detect_bonds(self):
        """Auto-detect bonds based on covalent radii"""
        n_atoms = len(self.atoms)
        self.bonds = []
        
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
                    self.bonds.append((i + 1, j + 1, 1))  # 1-based indices, single bond
    
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
    
    def get_box_size(self, padding: float = 5.0) -> Tuple[float, float, float]:
        """Calculate bounding box with padding"""
        coords = self.get_coordinates()
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        size = max_coords - min_coords
        return size[0] + 2*padding, size[1] + 2*padding, size[2] + 2*padding
    
    def get_frame_count(self) -> int:
        """Get number of frames in multi-frame XYZ"""
        return len(self.frames)
    
    def load_frame(self, frame_index: int) -> bool:
        """Load a specific frame from multi-frame XYZ"""
        if 0 <= frame_index < len(self.frames):
            frame = self.frames[frame_index]
            self.atoms = frame['atoms']
            self.comment = frame['comment']
            self.title = f"Frame {frame_index}"
            self._detect_bonds()
            return True
        return False