"""
Author: Acxel Orozco
Date: 20/01/2026 (Started)
Last modification: 01/02/2026
Main UI widget for ADOBMD Converter - Base class for analysis plugins - Quantum Analysis Helper (QAH)

QAH is a python software for plot and analyze Quantum Espresso DOS, 
PDOS and Bands structures with a friendly GUI and extensive plugins.

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


import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import numpy as np

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QFileDialog,
                               QTextEdit, QGroupBox, QRadioButton,
                               QSpinBox, QDoubleSpinBox, QCheckBox,
                               QTabWidget, QTableWidget, QTableWidgetItem,
                               QHeaderView, QMessageBox, QProgressBar,
                               QComboBox, QLineEdit)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont

from .pdb_parser import PDBParser
from .sdf_parser import SDFParser
from .adobmd_writer import ADOBMDWriter

class ConverterThread(QThread):
    """Background thread for file conversion"""
    
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(bool, str)
    
    def __init__(self, input_file: str, output_file: str, options: Dict):
        super().__init__()
        self.input_file = input_file
        self.output_file = output_file
        self.options = options
    
    def run(self):
        try:
            self.status.emit("Parsing input file...")
            self.progress.emit(20)
            
            # Parse based on file extension
            ext = Path(self.input_file).suffix.lower()
            writer = ADOBMDWriter()
            
            if ext == '.pdb':
                success = self._parse_pdb(writer)
            elif ext == '.sdf':
                success = self._parse_sdf(writer)
            else:
                self.finished.emit(False, f"Unsupported file format: {ext}")
                return
            
            if not success:
                self.finished.emit(False, "Failed to parse input file")
                return
            
            self.progress.emit(60)
            self.status.emit("Writing ADOBMD data file...")
            
            # Write output
            writer.write(self.output_file)
            
            # Write QM region file if requested
            if self.options.get('save_qm_region', False):
                qm_file = Path(self.output_file).with_suffix('.qm.txt')
                writer.write_qm_region_file(str(qm_file))
            
            self.progress.emit(100)
            self.status.emit("Conversion complete!")
            
            # Get statistics for summary
            stats = writer.get_statistics()
            summary = (f"Successfully converted {Path(self.input_file).name}\n\n"
                      f"Atoms: {stats['atoms']}\n"
                      f"Bonds: {stats['bonds']}\n"
                      f"Atom types: {stats['atom_types']}\n"
                      f"QM atoms: {stats['qm_atoms']}\n"
                      f"Box size: {stats['box'][0]:.1f} x {stats['box'][1]:.1f} x {stats['box'][2]:.1f} Å\n\n"
                      f"Elements:\n")
            
            for elem, count in stats['elements'].items():
                summary += f"  {elem}: {count}\n"
            
            self.finished.emit(True, summary)
            
        except Exception as e:
            logging.error(f"Conversion error: {e}")
            self.finished.emit(False, f"Error: {str(e)}")
    
    def _parse_pdb(self, writer: ADOBMDWriter) -> bool:
        """Parse PDB file and populate writer"""
        parser = PDBParser(bond_cutoff_factor=self.options.get('bond_cutoff', 1.2))
        
        if not parser.parse(self.input_file):
            return False
        
        # Set title
        writer.title = parser.title or Path(self.input_file).stem
        
        # Determine QM atoms
        qm_by_molecule = self.options.get('qm_by_molecule', False)
        qm_molecules = self.options.get('qm_molecules', [])
        
        # Add atoms
        for i, atom in enumerate(parser.atoms):
            is_qm = False
            
            if self.options.get('first_n_qm', 0) > 0:
                is_qm = i < self.options['first_n_qm']
            elif qm_by_molecule:
                is_qm = atom.res_seq in qm_molecules
            elif self.options.get('qm_indices'):
                is_qm = (i + 1) in self.options['qm_indices']
            
            writer.add_atom(
                element=atom.element,
                x=atom.x,
                y=atom.y,
                z=atom.z,
                molecule_id=atom.res_seq,
                charge=0.0,
                is_qm=is_qm
            )
        
        # Add bonds
        for bond in parser.bonds:
            writer.add_bond(bond.atom1, bond.atom2, bond.bond_type)
        
        # Set box
        box_size = parser.get_box_size()
        writer.set_box(box_size[0], box_size[1], box_size[2], 
                      padding=self.options.get('box_padding', 5.0))
        
        return True
    
    def _parse_sdf(self, writer: ADOBMDWriter) -> bool:
        """Parse SDF file and populate writer"""
        parser = SDFParser()
        
        if not parser.parse(self.input_file):
            return False
        
        mol = parser.get_first_molecule()
        if not mol:
            return False
        
        # Set title
        writer.title = mol.get('name', Path(self.input_file).stem)
        
        # Add atoms
        for i, atom in enumerate(mol['atoms']):
            is_qm = False
            
            if self.options.get('first_n_qm', 0) > 0:
                is_qm = i < self.options['first_n_qm']
            elif self.options.get('qm_indices'):
                is_qm = (i + 1) in self.options['qm_indices']
            
            writer.add_atom(
                element=atom.element,
                x=atom.x,
                y=atom.y,
                z=atom.z,
                molecule_id=1,
                charge=atom.charge / 10.0,  # SDF stores charge in 1/10 units
                is_qm=is_qm,
                mass=atom.mass
            )
        
        # Add bonds
        for bond in mol['bonds']:
            writer.add_bond(bond.atom1, bond.atom2, bond.bond_type)
        
        # Calculate box from coordinates
        coords = np.array([[a.x, a.y, a.z] for a in mol['atoms']])
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        box_size = max_coords - min_coords
        writer.set_box(box_size[0], box_size[1], box_size[2],
                      padding=self.options.get('box_padding', 5.0))
        
        return True

class ConverterWidget(QWidget):
    """Main widget for ADOBMD Converter"""
    
    def __init__(self, plugin_instance=None):
        super().__init__()
        self.plugin = plugin_instance
        self.converter_thread = None
        self.input_file = ""
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("ADOBMD Converter")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Input file selection
        file_group = QGroupBox("Input File")
        file_layout = QHBoxLayout()
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        
        select_btn = QPushButton("Browse...")
        select_btn.clicked.connect(self.select_input_file)
        select_btn.setFixedWidth(250)
        
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(select_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Options tabs
        tabs = QTabWidget()
        
        # General options
        general_tab = self.create_general_tab()
        tabs.addTab(general_tab, "General")
        
        # QM Region options
        qm_tab = self.create_qm_tab()
        tabs.addTab(qm_tab, "QM Region")
        
        # Bond options
        bond_tab = self.create_bond_tab()
        tabs.addTab(bond_tab, "Bonds")
        
        layout.addWidget(tabs)
        
        # Output options
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        
        # Output file
        file_out_layout = QHBoxLayout()
        self.output_label = QLabel("adobmd.data")
        self.output_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        
        output_btn = QPushButton("Output File...")
        output_btn.clicked.connect(self.select_output_file)
        output_btn.setFixedWidth(250)
        
        file_out_layout.addWidget(self.output_label, 1)
        file_out_layout.addWidget(output_btn)
        output_layout.addLayout(file_out_layout)
        
        # Additional output options
        self.save_qm_check = QCheckBox("Save separate QM region file")
        self.save_qm_check.setChecked(True)
        output_layout.addWidget(self.save_qm_check)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Convert button
        self.convert_btn = QPushButton("Convert to ADOBMD Format")
        self.convert_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.convert_btn.clicked.connect(self.start_conversion)
        self.convert_btn.setEnabled(False)
        layout.addWidget(self.convert_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Output display
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setFont(QFont("Courier New", 10))
        self.output_display.setMaximumHeight(200)
        layout.addWidget(self.output_display)
        
        # Store options
        self.output_file = "adobmd.data"
    
    def create_general_tab(self) -> QWidget:
        """Create general options tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Box padding
        pad_layout = QHBoxLayout()
        pad_layout.addWidget(QLabel("Box padding (Å):"))
        self.box_padding = QDoubleSpinBox()
        self.box_padding.setRange(0.0, 20.0)
        self.box_padding.setValue(5.0)
        self.box_padding.setSingleStep(0.5)
        pad_layout.addWidget(self.box_padding)
        pad_layout.addStretch()
        layout.addLayout(pad_layout)
        
        # Bond cutoff
        cutoff_layout = QHBoxLayout()
        cutoff_layout.addWidget(QLabel("Bond detection cutoff factor:"))
        self.bond_cutoff = QDoubleSpinBox()
        self.bond_cutoff.setRange(0.5, 2.0)
        self.bond_cutoff.setValue(1.2)
        self.bond_cutoff.setSingleStep(0.1)
        cutoff_layout.addWidget(self.bond_cutoff)
        cutoff_layout.addStretch()
        layout.addLayout(cutoff_layout)
        
        # Auto-detect bonds
        self.auto_bonds = QCheckBox("Auto-detect bonds (if not in file)")
        self.auto_bonds.setChecked(True)
        layout.addWidget(self.auto_bonds)
        
        layout.addStretch()
        return widget
    
    def create_qm_tab(self) -> QWidget:
        """Create QM region options tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # QM region selection method
        self.qm_method = QComboBox()
        self.qm_method.addItems(["None", "First N atoms", "By molecule ID", "Manual indices"])
        self.qm_method.currentTextChanged.connect(self.on_qm_method_changed)
        layout.addWidget(QLabel("QM region selection:"))
        layout.addWidget(self.qm_method)
        
        # First N atoms
        self.first_n_layout = QHBoxLayout()
        self.first_n_layout.addWidget(QLabel("First N atoms:"))
        self.first_n_atoms = QSpinBox()
        self.first_n_atoms.setRange(0, 10000)
        self.first_n_atoms.setValue(10)
        self.first_n_layout.addWidget(self.first_n_atoms)
        self.first_n_layout.addStretch()
        layout.addLayout(self.first_n_layout)
        
        # By molecule ID
        self.molecule_layout = QVBoxLayout()
        self.molecule_layout.addWidget(QLabel("Molecule IDs (comma-separated):"))
        self.molecule_ids = QLineEdit()
        self.molecule_ids.setPlaceholderText("e.g., 1,2,3")
        self.molecule_layout.addWidget(self.molecule_ids)
        layout.addLayout(self.molecule_layout)
        
        # Manual indices
        self.indices_layout = QVBoxLayout()
        self.indices_layout.addWidget(QLabel("Atom indices (comma-separated):"))
        self.atom_indices = QLineEdit()
        self.atom_indices.setPlaceholderText("e.g., 1,5,10-20")
        self.indices_layout.addWidget(self.atom_indices)
        layout.addLayout(self.indices_layout)
        
        # Initially hide all
        self.first_n_layout.itemAt(0).widget().setVisible(False)
        self.first_n_atoms.setVisible(False)
        self.molecule_ids.setVisible(False)
        self.atom_indices.setVisible(False)
        
        layout.addStretch()
        return widget
    
    def create_bond_tab(self) -> QWidget:
        """Create bond options tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Bond table
        self.bond_table = QTableWidget()
        self.bond_table.setColumnCount(3)
        self.bond_table.setHorizontalHeaderLabels(["Atom 1", "Atom 2", "Type"])
        self.bond_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(QLabel("Bonds (will be populated after loading file)"))
        layout.addWidget(self.bond_table)
        
        return widget
    
    def on_qm_method_changed(self, method: str):
        """Handle QM method change"""
        # Hide all
        self.first_n_atoms.setVisible(False)
        self.molecule_ids.setVisible(False)
        self.atom_indices.setVisible(False)
        
        # Show selected
        if method == "First N atoms":
            self.first_n_atoms.setVisible(True)
        elif method == "By molecule ID":
            self.molecule_ids.setVisible(True)
        elif method == "Manual indices":
            self.atom_indices.setVisible(True)
    
    def select_input_file(self):
        """Select input file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input File", "",
            "Supported Files (*.pdb *.sdf);;PDB Files (*.pdb);;SDF Files (*.sdf);;All Files (*)"
        )
        
        if file_path:
            self.input_file = file_path
            self.file_label.setText(Path(file_path).name)
            self.convert_btn.setEnabled(True)
            self.output_display.clear()
            self.output_display.append(f"Selected: {file_path}")
            self._preview_file(file_path)
    
    def select_output_file(self):
        """Select output file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save ADOBMD Data File", "adobmd.data",
            "ADOBMD Data Files (*.data);;All Files (*)"
        )
        
        if file_path:
            self.output_file = file_path
            self.output_label.setText(Path(file_path).name)
    
    def _preview_file(self, file_path: str):
        """Preview file contents"""
        try:
            ext = Path(file_path).suffix.lower()
            
            if ext == '.pdb':
                parser = PDBParser()
                if parser.parse(file_path):
                    self.output_display.append(f"\nPDB File Summary:")
                    self.output_display.append(f"  Atoms: {len(parser.atoms)}")
                    self.output_display.append(f"  Bonds: {len(parser.bonds)}")
                    self.output_display.append(f"  Title: {parser.title}")
                    
                    # Update bond table
                    self._update_bond_table(parser.bonds)
            
            elif ext == '.sdf':
                parser = SDFParser()
                if parser.parse(file_path):
                    mol = parser.get_first_molecule()
                    if mol:
                        self.output_display.append(f"\nSDF File Summary:")
                        self.output_display.append(f"  Name: {mol.get('name', 'Unknown')}")
                        self.output_display.append(f"  Atoms: {len(mol['atoms'])}")
                        self.output_display.append(f"  Bonds: {len(mol['bonds'])}")
                        
                        # Update bond table
                        self._update_bond_table(mol['bonds'])
            
        except Exception as e:
            self.output_display.append(f"Preview error: {e}")
    
    def _update_bond_table(self, bonds):
        """Update bond table with preview data"""
        self.bond_table.setRowCount(len(bonds))
        for i, bond in enumerate(bonds[:100]):  # Limit to first 100 bonds
            self.bond_table.setItem(i, 0, QTableWidgetItem(str(bond.atom1)))
            self.bond_table.setItem(i, 1, QTableWidgetItem(str(bond.atom2)))
            self.bond_table.setItem(i, 2, QTableWidgetItem(str(bond.bond_type)))
    
    def start_conversion(self):
        """Start conversion in background thread"""
        if not self.input_file:
            QMessageBox.warning(self, "Warning", "Please select an input file")
            return
        
        # Disable UI
        self.convert_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting conversion...")
        
        # Collect options
        options = {
            'bond_cutoff': self.bond_cutoff.value(),
            'box_padding': self.box_padding.value(),
            'auto_detect_bonds': self.auto_bonds.isChecked(),
            'save_qm_region': self.save_qm_check.isChecked()
        }
        
        # QM region options
        method = self.qm_method.currentText()
        if method == "First N atoms":
            options['first_n_qm'] = self.first_n_atoms.value()
        elif method == "By molecule ID":
            try:
                ids = [int(x.strip()) for x in self.molecule_ids.text().split(',') if x.strip()]
                options['qm_molecules'] = ids
                options['qm_by_molecule'] = True
            except:
                QMessageBox.warning(self, "Warning", "Invalid molecule IDs")
                self.convert_btn.setEnabled(True)
                return
        elif method == "Manual indices":
            try:
                indices = []
                for part in self.atom_indices.text().split(','):
                    part = part.strip()
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        indices.extend(range(start, end + 1))
                    else:
                        indices.append(int(part))
                options['qm_indices'] = indices
            except:
                QMessageBox.warning(self, "Warning", "Invalid atom indices")
                self.convert_btn.setEnabled(True)
                return
        
        # Start conversion thread
        self.converter_thread = ConverterThread(
            self.input_file, self.output_file, options
        )
        self.converter_thread.progress.connect(self.progress_bar.setValue)
        self.converter_thread.status.connect(self.status_label.setText)
        self.converter_thread.finished.connect(self.on_conversion_finished)
        self.converter_thread.start()
    
    def on_conversion_finished(self, success: bool, message: str):
        """Handle conversion completion"""
        self.convert_btn.setEnabled(True)
        
        if success:
            self.status_label.setText("Conversion completed successfully!")
            self.output_display.append("\n" + "="*50)
            self.output_display.append(message)
            self.output_display.append("="*50)
            
            QMessageBox.information(self, "Success", 
                                   f"File converted successfully!\n\n{message}")
        else:
            self.status_label.setText("Conversion failed")
            self.output_display.append(f"\nERROR: {message}")
            
            QMessageBox.critical(self, "Error", 
                               f"Conversion failed:\n{message}")
        
        self.progress_bar.setVisible(False)