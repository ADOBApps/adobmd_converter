"""
CIF (Crystallographic Information File) parser for ADOBMD Converter

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

class CIFAtom:
    """Represents an atom in CIF format"""
    
    def __init__(self):
        self.element = ""
        self.x = 0.0  # Fractional coordinates
        self.y = 0.0
        self.z = 0.0
        self.site_label = ""
        self.occupancy = 1.0
        self.u_iso = 0.0
        self.fract_x = 0.0
        self.fract_y = 0.0
        self.fract_z = 0.0
        self.cart_x = 0.0  # Cartesian coordinates (converted)
        self.cart_y = 0.0
        self.cart_z = 0.0
        self.mass = 0.0

class CIFFrame:
    """Represents crystal frame in CIF"""
    
    def __init__(self):
        self.a = 0.0  # Unit cell lengths
        self.b = 0.0
        self.c = 0.0
        self.alpha = 90.0  # Unit cell angles (degrees)
        self.beta = 90.0
        self.gamma = 90.0
        self.space_group = "P1"
        self.volume = 0.0
        self.cell_formula = ""
        self.cell_formula_units = 1  # Z value

class CIFParser:
    """Parser for Crystallographic Information File (CIF) format"""
    
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
    
    def __init__(self, supercell: Tuple[int, int, int] = (1, 1, 1)):
        self.atoms: List[CIFAtom] = []
        self.bonds: List[Tuple[int, int, int]] = []
        self.frame = CIFFrame()
        self.title = ""
        self.supercell = supercell  # Create supercell (n1, n2, n3)
        self.transformation_matrix = None  # For converting fractional to Cartesian
        self.inverse_matrix = None
        self.symmetry_ops = []  # Store symmetry operations
    
    def parse(self, filename: str, apply_symmetry: bool = True) -> bool:
        """Parse CIF file
        apply_symmetry: If True, apply symmetry operations to generate full unit cell
        """
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Remove comments (start with #)
            lines = [line.split('#')[0] for line in lines]
            
            # Parse data blocks
            data_blocks = self._split_data_blocks(lines)
            
            if not data_blocks:
                return False
            
            # Parse first data block
            self._parse_data_block(data_blocks[0])
            
            # Apply symmetry to generate full unit cell
            if apply_symmetry and self.symmetry_ops:
                self.apply_symmetry(generate_all=True)
            
            # Convert fractional to Cartesian coordinates
            self._fractional_to_cartesian()
            
            # Apply supercell expansion
            if self.supercell != (1, 1, 1):
                self._expand_supercell()
            
            # Auto-detect bonds
            self._detect_bonds()
            
            return True
            
        except Exception as e:
            print(f"Error parsing CIF file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _split_data_blocks(self, lines: List[str]) -> List[List[str]]:
        """Split CIF into data blocks (data_XXX)"""
        blocks = []
        current_block = []
        in_block = False
        
        for line in lines:
            if line.strip().startswith('data_'):
                if current_block:
                    blocks.append(current_block)
                current_block = [line]
                in_block = True
            elif in_block:
                current_block.append(line)
        
        if current_block:
            blocks.append(current_block)
        
        return blocks
    
    def _parse_data_block(self, lines: List[str]):
        """Parse a single data block"""
        # Extract data names and values
        data_dict = {}
        loop_data = []
        in_loop = False
        loop_headers = []
        current_loop_lines = []
        
        # Store raw symmetry operations
        self._raw_symmetry_ops = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Handle loop_ keyword
            if line.startswith('loop_'):
                if in_loop and loop_headers and current_loop_lines:
                    # Process previous loop before starting new one
                    self._process_loop_data(loop_headers, current_loop_lines, data_dict)
                
                in_loop = True
                loop_headers = []
                current_loop_lines = []
                i += 1
                continue
            
            if in_loop:
                if line.startswith('_'):
                    # This is a header line
                    loop_headers.append(line)
                    i += 1
                else:
                    # This is data - collect all data lines until next loop or empty line
                    while i < len(lines) and lines[i].strip() and not lines[i].startswith('loop_'):
                        current_loop_lines.append(lines[i].strip())
                        i += 1
                    
                    # Process the collected loop data
                    self._process_loop_data(loop_headers, current_loop_lines, data_dict)
                    
                    in_loop = False
            else:
                # Regular data_* line
                if line.startswith('_'):
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0]
                        value = ' '.join(parts[1:])
                        data_dict[key] = self._parse_cif_value(value)
                i += 1
        
        # Process any remaining loop
        if in_loop and loop_headers and current_loop_lines:
            self._process_loop_data(loop_headers, current_loop_lines, data_dict)
        
        # Extract cell parameters
        self._extract_cell_parameters(data_dict)
        
        # Extract atoms
        self._extract_atoms(data_dict)
    
    def _process_loop_data(self, headers: List[str], lines: List[str], data_dict: Dict):
        """Process loop data and add to data_dict with indices"""
        if not headers or not lines:
            return
        
        n_cols = len(headers)
        all_values = []
        for data_line in lines:
            # Handle quoted strings by simple split
            all_values.extend(data_line.split())
        
        n_rows = len(all_values) // n_cols
        
        # Check if this is symmetry operations
        is_symmetry = any('_symmetry_equiv_pos' in h for h in headers)
        
        for row in range(n_rows):
            for col, header in enumerate(headers):
                idx = row * n_cols + col
                if idx < len(all_values):
                    key = f"{header}_{row}"
                    value = all_values[idx]
                    
                    # Store in data_dict
                    data_dict[key] = self._parse_cif_value(value)
                    
                    # If this is symmetry operation, also store separately
                    if is_symmetry and '_symmetry_equiv_pos_as_xyz' in header:
                        self._raw_symmetry_ops.append(value)
    
    def _parse_cif_value(self, value: str):
        """Parse CIF value (handles quoted strings, numbers, etc.)"""
        value = value.strip()
        
        # Remove surrounding quotes
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        
        # Try to convert to number
        try:
            return float(value)
        except ValueError:
            try:
                return int(value)
            except ValueError:
                return value
    
    def _extract_cell_parameters(self, data: Dict):
        """Extract unit cell parameters from CIF data"""
        
        # Look for cell length tags
        for tag in ['_cell_length_a', '_a']:
            if tag in data:
                self.frame.a = float(data[tag])
                break
        
        for tag in ['_cell_length_b', '_b']:
            if tag in data:
                self.frame.b = float(data[tag])
                break
        
        for tag in ['_cell_length_c', '_c']:
            if tag in data:
                self.frame.c = float(data[tag])
                break
        
        # Look for cell angle tags
        for tag in ['_cell_angle_alpha', '_alpha']:
            if tag in data:
                self.frame.alpha = float(data[tag])
                break
        
        for tag in ['_cell_angle_beta', '_beta']:
            if tag in data:
                self.frame.beta = float(data[tag])
                break
        
        for tag in ['_cell_angle_gamma', '_gamma']:
            if tag in data:
                self.frame.gamma = float(data[tag])
                break
        
        # Look for space group
        for tag in ['_symmetry_space_group_name_H-M', '_space_group_name_H-M_alt']:
            if tag in data:
                self.frame.space_group = str(data[tag])
                break
        
        # Look for cell formula
        for tag in ['_chemical_formula_sum', '_chemical_formula_structural']:
            if tag in data:
                self.frame.cell_formula = str(data[tag])
                break
        
        # Parse symmetry operations
        self.symmetry_ops = []
        
        # Look for symmetry operations with indices
        op_dict = {}
        for key, value in data.items():
            if '_symmetry_equiv_pos_as_xyz_' in key:
                try:
                    idx = int(key.split('_')[-1])
                    op_dict[idx] = str(value)
                except (ValueError, IndexError):
                    pass
        
        if op_dict:
            # Sort by index and add to list
            for idx in sorted(op_dict.keys()):
                self.symmetry_ops.append(op_dict[idx])
        else:
            # Try without indices
            for key, value in data.items():
                if '_symmetry_equiv_pos_as_xyz' in key and not key.endswith('_'):
                    self.symmetry_ops.append(str(value))
        
        # If still no symmetry ops, try to parse from the loop in the raw data
        if not self.symmetry_ops and hasattr(self, '_raw_symmetry_ops'):
            self.symmetry_ops = self._raw_symmetry_ops
        
        print(f"Found {len(self.symmetry_ops)} symmetry operations")
        
        # Calculate transformation matrix from fractional to Cartesian
        self._calculate_transformation_matrix()
        
        # Calculate cell volume
        self.frame.volume = self._calculate_volume()
    
    def _calculate_transformation_matrix(self):
        """Calculate transformation matrix from fractional to Cartesian coordinates"""
        a = self.frame.a
        b = self.frame.b
        c = self.frame.c
        alpha = np.radians(self.frame.alpha)
        beta = np.radians(self.frame.beta)
        gamma = np.radians(self.frame.gamma)
        
        # Standard triclinic transformation matrix
        volume = a * b * c * np.sqrt(
            1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 +
            2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
        
        self.transformation_matrix = np.array([
            [a, b * np.cos(gamma), c * np.cos(beta)],
            [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)],
            [0, 0, volume / (a * b * np.sin(gamma))]
        ])
        
        self.inverse_matrix = np.linalg.inv(self.transformation_matrix)
    
    def _calculate_volume(self) -> float:
        """Calculate unit cell volume"""
        a = self.frame.a
        b = self.frame.b
        c = self.frame.c
        alpha = np.radians(self.frame.alpha)
        beta = np.radians(self.frame.beta)
        gamma = np.radians(self.frame.gamma)
        
        volume = a * b * c * np.sqrt(
            1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 +
            2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
        
        return volume
    
    def _extract_atoms(self, data: Dict):
        """Extract atoms from CIF data"""
        # Clear existing atoms
        self.atoms = []
        
        # First, check if we have atom site data in loop format
        atom_keys = [k for k in data.keys() if '_atom_site' in k]
        
        if not atom_keys:
            return
        
        # Special handling for standard CIF loop format
        site_indices = set()
        for key in atom_keys:
            parts = key.split('_')
            if len(parts) >= 3:
                last_part = parts[-1]
                if last_part.isdigit():
                    idx = int(last_part)
                    site_indices.add(idx)
        
        if site_indices:
            # Format with numbered entries
            for idx in sorted(site_indices):  # Sort for consistency
                atom = CIFAtom()
                
                # Get element/type
                type_key = f'_atom_site_type_symbol_{idx}'
                if type_key in data:
                    atom.element = self._extract_element(str(data[type_key]))
                else:
                    # Try alternative keys
                    for key in [f'_atom_site_label_{idx}', f'_atom_site_type_{idx}']:
                        if key in data:
                            atom.element = self._extract_element(str(data[key]))
                            break
                
                if not atom.element:
                    continue
                
                # Get fractional coordinates
                atom.fract_x = float(data.get(f'_atom_site_fract_x_{idx}', 0.0))
                atom.fract_y = float(data.get(f'_atom_site_fract_y_{idx}', 0.0))
                atom.fract_z = float(data.get(f'_atom_site_fract_z_{idx}', 0.0))
                
                # Get occupancy
                atom.occupancy = float(data.get(f'_atom_site_occupancy_{idx}', 1.0))
                
                # Get thermal parameter
                atom.u_iso = float(data.get(f'_atom_site_U_iso_or_equiv_{idx}', 0.0))
                
                # Get label
                for key in [f'_atom_site_label_{idx}', f'_atom_site_type_symbol_{idx}']:
                    if key in data:
                        atom.site_label = str(data[key])
                        break
                
                # Set mass
                atom.mass = self.ATOMIC_MASSES.get(atom.element, 12.01)
                
                self.atoms.append(atom)
        
        else:
            # Try to parse as loop data (no indices in keys)
            self._parse_loop_atoms(data)
        
        # If still no atoms, try the specific format from your CIF
        if len(self.atoms) == 0:
            self._parse_specific_cif_format(data)
        
        print(f"Extracted {len(self.atoms)} atoms from CIF")
    
    def _parse_loop_atoms(self, data: Dict):
        """Parse atoms from loop data (no indices in keys)"""
        # Group keys by their base name
        atom_fields = {}
        for key in data.keys():
            if '_atom_site' in key:
                # Extract field name without index
                parts = key.split('_')
                if len(parts) >= 3:
                    field = '_'.join(parts[1:])  # Remove leading underscore
                    if field not in atom_fields:
                        atom_fields[field] = []
                    atom_fields[field].append(data[key])
        
        # Check if we have multiple values per field
        if atom_fields:
            # Get number of atoms from first field
            first_field = next(iter(atom_fields.values()))
            n_atoms = len(first_field)
            
            for i in range(n_atoms):
                atom = CIFAtom()
                
                # Get element
                for field in ['atom_site_type_symbol', 'atom_site_label', 'atom_site_type']:
                    if field in atom_fields and i < len(atom_fields[field]):
                        atom.element = self._extract_element(str(atom_fields[field][i]))
                        break
                
                if not atom.element:
                    continue
                
                # Get coordinates
                if 'atom_site_fract_x' in atom_fields and i < len(atom_fields['atom_site_fract_x']):
                    atom.fract_x = float(atom_fields['atom_site_fract_x'][i])
                if 'atom_site_fract_y' in atom_fields and i < len(atom_fields['atom_site_fract_y']):
                    atom.fract_y = float(atom_fields['atom_site_fract_y'][i])
                if 'atom_site_fract_z' in atom_fields and i < len(atom_fields['atom_site_fract_z']):
                    atom.fract_z = float(atom_fields['atom_site_fract_z'][i])
                
                # Get occupancy
                if 'atom_site_occupancy' in atom_fields and i < len(atom_fields['atom_site_occupancy']):
                    atom.occupancy = float(atom_fields['atom_site_occupancy'][i])
                
                # Get label
                if 'atom_site_label' in atom_fields and i < len(atom_fields['atom_site_label']):
                    atom.site_label = str(atom_fields['atom_site_label'][i])
                
                # Set mass
                atom.mass = self.ATOMIC_MASSES.get(atom.element, 12.01)
                
                self.atoms.append(atom)
    
    def _parse_specific_cif_format(self, data: Dict):
        """Parse the specific CIF format from your example"""
        # Look for atom site data in the specific format
        atom_type_symbols = []
        atom_labels = []
        atom_multiplicities = []
        atom_fract_x = []
        atom_fract_y = []
        atom_fract_z = []
        atom_occupancies = []
        
        # Collect all possible atom data
        for key, value in data.items():
            if key.startswith('_atom_site_type_symbol_'):
                try:
                    atom_type_symbols.append((int(key.split('_')[-1]), str(value)))
                except:
                    pass
            elif key.startswith('_atom_site_label_'):
                try:
                    atom_labels.append((int(key.split('_')[-1]), str(value)))
                except:
                    pass
            elif key.startswith('_atom_site_symmetry_multiplicity_'):
                try:
                    atom_multiplicities.append((int(key.split('_')[-1]), float(value)))
                except:
                    pass
            elif key.startswith('_atom_site_fract_x_'):
                try:
                    atom_fract_x.append((int(key.split('_')[-1]), float(value)))
                except:
                    pass
            elif key.startswith('_atom_site_fract_y_'):
                try:
                    atom_fract_y.append((int(key.split('_')[-1]), float(value)))
                except:
                    pass
            elif key.startswith('_atom_site_fract_z_'):
                try:
                    atom_fract_z.append((int(key.split('_')[-1]), float(value)))
                except:
                    pass
            elif key.startswith('_atom_site_occupancy_'):
                try:
                    atom_occupancies.append((int(key.split('_')[-1]), float(value)))
                except:
                    pass
        
        # Sort by index
        atom_type_symbols.sort(key=lambda x: x[0])
        atom_labels.sort(key=lambda x: x[0])
        atom_multiplicities.sort(key=lambda x: x[0])
        atom_fract_x.sort(key=lambda x: x[0])
        atom_fract_y.sort(key=lambda x: x[0])
        atom_fract_z.sort(key=lambda x: x[0])
        atom_occupancies.sort(key=lambda x: x[0])
        
        # Create atoms
        n_atoms = len(atom_type_symbols)
        for i in range(n_atoms):
            atom = CIFAtom()
            
            # Get element from type symbol (Ti4+ -> Ti)
            type_symbol = atom_type_symbols[i][1] if i < len(atom_type_symbols) else ""
            atom.element = self._extract_element(type_symbol)
            
            if not atom.element:
                continue
            
            # Get coordinates
            atom.fract_x = atom_fract_x[i][1] if i < len(atom_fract_x) else 0.0
            atom.fract_y = atom_fract_y[i][1] if i < len(atom_fract_y) else 0.0
            atom.fract_z = atom_fract_z[i][1] if i < len(atom_fract_z) else 0.0
            
            # Get occupancy
            atom.occupancy = atom_occupancies[i][1] if i < len(atom_occupancies) else 1.0
            
            # Get label
            atom.site_label = atom_labels[i][1] if i < len(atom_labels) else ""
            
            # Set mass
            atom.mass = self.ATOMIC_MASSES.get(atom.element, 12.01)
            
            self.atoms.append(atom)
        
        # If we still have no atoms, try one more approach
        if n_atoms == 0:
            # Direct lookup of specific keys in data
            for i in range(10):  # Try up to 10 atoms
                type_key = f'_atom_site_type_symbol_{i}'
                if type_key in data:
                    atom = CIFAtom()
                    atom.element = self._extract_element(str(data[type_key]))
                    atom.fract_x = float(data.get(f'_atom_site_fract_x_{i}', 0.0))
                    atom.fract_y = float(data.get(f'_atom_site_fract_y_{i}', 0.0))
                    atom.fract_z = float(data.get(f'_atom_site_fract_z_{i}', 0.0))
                    atom.occupancy = float(data.get(f'_atom_site_occupancy_{i}', 1.0))
                    atom.site_label = str(data.get(f'_atom_site_label_{i}', ''))
                    atom.mass = self.ATOMIC_MASSES.get(atom.element, 12.01)
                    self.atoms.append(atom)
    
    def _extract_element(self, symbol: str) -> str:
        """Extract element symbol from atom label"""
        # Remove numbers and special characters
        element = re.sub(r'[0-9\'\+\-]', '', symbol)
        
        # Take first 1-2 letters
        match = re.match(r'([A-Z][a-z]?)', element)
        if match:
            return match.group(1)
        
        return symbol[:2].capitalize()
    
    def _fractional_to_cartesian(self):
        """Convert fractional coordinates to Cartesian"""
        if self.transformation_matrix is None:
            return
        
        for atom in self.atoms:
            fract = np.array([atom.fract_x, atom.fract_y, atom.fract_z])
            cart = np.dot(self.transformation_matrix, fract)
            atom.cart_x, atom.cart_y, atom.cart_z = cart
    
    def _expand_supercell(self):
        """Expand crystal to supercell"""
        nx, ny, nz = self.supercell
        original_atoms = self.atoms.copy()
        self.atoms = []
        
        atom_id = 1
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    shift = np.dot(
                        self.transformation_matrix,
                        np.array([i, j, k])
                    )
                    
                    for orig in original_atoms:
                        atom = CIFAtom()
                        atom.element = orig.element
                        atom.fract_x = orig.fract_x + i
                        atom.fract_y = orig.fract_y + j
                        atom.fract_z = orig.fract_z + k
                        atom.cart_x = orig.cart_x + shift[0]
                        atom.cart_y = orig.cart_y + shift[1]
                        atom.cart_z = orig.cart_z + shift[2]
                        atom.occupancy = orig.occupancy
                        atom.mass = orig.mass
                        atom.site_label = f"{orig.site_label}_{i}_{j}_{k}"
                        
                        self.atoms.append(atom)
                        atom_id += 1
        
        # Update box size for supercell
        self.frame.a *= nx
        self.frame.b *= ny
        self.frame.c *= nz
        self._calculate_transformation_matrix()
    
    def apply_symmetry(self, generate_all: bool = True):
        """
        Apply symmetry operations to generate all symmetry-equivalent positions
        generate_all: If True, generate all positions in the unit cell
                    If False, keep only asymmetric unit
        """
        if not generate_all:
            return
        
        if not self.symmetry_ops:
            print("No symmetry operations found, keeping asymmetric unit")
            return
        
        # Store original atoms (asymmetric unit)
        original_atoms = self.atoms.copy()
        if not original_atoms:
            return
        
        self.atoms = []
        
        # Track unique positions to avoid duplicates
        unique_positions = set()
        tolerance = 1e-4
        
        # For each atom in asymmetric unit
        for asym_atom in original_atoms:
            # For each symmetry operation
            for op in self.symmetry_ops:
                # Apply symmetry operation to fractional coordinates
                x, y, z = self._apply_symmetry_op(asym_atom.fract_x, 
                                                asym_atom.fract_y, 
                                                asym_atom.fract_z, op)
                
                # Bring coordinates back into [0,1) range
                x = x % 1.0
                y = y % 1.0
                z = z % 1.0
                if x < 0: x += 1.0
                if y < 0: y += 1.0
                if z < 0: z += 1.0
                
                # Create a key for uniqueness check (rounded to tolerance)
                pos_key = (round(x/tolerance), round(y/tolerance), round(z/tolerance))
                
                if pos_key not in unique_positions:
                    unique_positions.add(pos_key)
                    
                    # Create new atom
                    atom = CIFAtom()
                    atom.element = asym_atom.element
                    atom.fract_x = x
                    atom.fract_y = y
                    atom.fract_z = z
                    atom.occupancy = asym_atom.occupancy
                    atom.mass = asym_atom.mass
                    atom.site_label = f"{asym_atom.element}{len(self.atoms)+1}"
                    atom.u_iso = asym_atom.u_iso
                    
                    self.atoms.append(atom)
        
        # Sort atoms by element and position for consistency
        self.atoms.sort(key=lambda a: (a.element, a.fract_x, a.fract_y, a.fract_z))
        
        print(f"Applied symmetry: {len(original_atoms)} asymmetric atoms -> {len(self.atoms)} total atoms")
    
    def _apply_symmetry_op(self, x: float, y: float, z: float, op: str) -> Tuple[float, float, float]:
        """Apply a symmetry operation string to fractional coordinates"""
        # Remove quotes and extra spaces
        op = op.strip().strip("'").strip('"')
        
        # Handle operations like 'x, y, z' or '-y, x+1/2, z+1/4'
        parts = op.split(',')
        if len(parts) != 3:
            return x, y, z
        
        def eval_expr(expr: str, x_val: float, y_val: float, z_val: float) -> float:
            """Evaluate a symmetry expression"""
            expr = expr.strip()
            
            # Handle common patterns
            expr = expr.replace('x', str(x_val))
            expr = expr.replace('y', str(y_val))
            expr = expr.replace('z', str(z_val))
            
            # Handle fractions
            expr = expr.replace('1/2', '0.5')
            expr = expr.replace('1/4', '0.25')
            expr = expr.replace('3/4', '0.75')
            
            # Handle negative signs
            expr = expr.replace('--', '+')
            expr = expr.replace('+-', '-')
            expr = expr.replace('-+', '-')
            
            # Safe evaluation
            try:
                return float(eval(expr))
            except:
                return 0.0
        
        try:
            new_x = eval_expr(parts[0].strip(), x, y, z)
            new_y = eval_expr(parts[1].strip(), x, y, z)
            new_z = eval_expr(parts[2].strip(), x, y, z)
            return new_x, new_y, new_z
        except:
            return x, y, z
    
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
                dx = atom1.cart_x - atom2.cart_x
                dy = atom1.cart_y - atom2.cart_y
                dz = atom1.cart_z - atom2.cart_z
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                # Bond if distance < sum of covalent radii * 1.2
                cutoff = (r1 + r2) * 1.2
                if 0.5 < dist < cutoff:  # Avoid self and too close
                    self.bonds.append((i + 1, j + 1, 1))
    
    def get_coordinates(self) -> np.ndarray:
        """Get Cartesian coordinates as numpy array (Angstroms)"""
        coords = np.zeros((len(self.atoms), 3))
        for i, atom in enumerate(self.atoms):
            coords[i, 0] = atom.cart_x
            coords[i, 1] = atom.cart_y
            coords[i, 2] = atom.cart_z
        return coords
    
    def get_fractional_coordinates(self) -> np.ndarray:
        """Get fractional coordinates as numpy array"""
        coords = np.zeros((len(self.atoms), 3))
        for i, atom in enumerate(self.atoms):
            coords[i, 0] = atom.fract_x
            coords[i, 1] = atom.fract_y
            coords[i, 2] = atom.fract_z
        return coords
    
    def get_elements(self) -> List[str]:
        """Get element symbols"""
        return [atom.element for atom in self.atoms]
    
    def get_masses(self) -> List[float]:
        """Get atomic masses"""
        return [atom.mass for atom in self.atoms]
    
    def get_occupancies(self) -> List[float]:
        """Get site occupancies"""
        return [atom.occupancy for atom in self.atoms]
    
    def get_box_size(self) -> Tuple[float, float, float]:
        """Get unit cell dimensions"""
        return self.frame.a, self.frame.b, self.frame.c
    
    def get_cell_angles(self) -> Tuple[float, float, float]:
        """Get unit cell angles (degrees)"""
        return self.frame.alpha, self.frame.beta, self.frame.gamma
    
    def get_space_group(self) -> str:
        """Get space group symbol"""
        return self.frame.space_group
    
    def get_cell_volume(self) -> float:
        """Get unit cell volume"""
        return self.frame.volume
    
    def write_cif(self, filename: str):
        """Write atoms back to CIF format (for testing)"""
        with open(filename, 'w') as f:
            f.write(f"data_{self.title}\n")
            f.write("\n")
            
            # Cell parameters
            f.write(f"_cell_length_a     {self.frame.a:.6f}\n")
            f.write(f"_cell_length_b     {self.frame.b:.6f}\n")
            f.write(f"_cell_length_c     {self.frame.c:.6f}\n")
            f.write(f"_cell_angle_alpha  {self.frame.alpha:.6f}\n")
            f.write(f"_cell_angle_beta   {self.frame.beta:.6f}\n")
            f.write(f"_cell_angle_gamma  {self.frame.gamma:.6f}\n")
            f.write(f"_cell_volume       {self.frame.volume:.6f}\n")
            f.write(f"_symmetry_space_group_name_H-M   '{self.frame.space_group}'\n")
            f.write("\n")
            
            # Atoms
            f.write("loop_\n")
            f.write("  _atom_site_label\n")
            f.write("  _atom_site_type_symbol\n")
            f.write("  _atom_site_fract_x\n")
            f.write("  _atom_site_fract_y\n")
            f.write("  _atom_site_fract_z\n")
            f.write("  _atom_site_occupancy\n")
            f.write("  _atom_site_U_iso_or_equiv\n")
            
            for i, atom in enumerate(self.atoms, 1):
                label = atom.site_label or f"{atom.element}{i}"
                f.write(f"  {label} {atom.element} "
                       f"{atom.fract_x:.6f} {atom.fract_y:.6f} {atom.fract_z:.6f} "
                       f"{atom.occupancy:.6f} {atom.u_iso:.6f}\n")