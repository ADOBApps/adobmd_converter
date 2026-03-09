"""
Author: Acxel Orozco
Date: 2026-03-09
Description: ADOBMD Converter - Convert SDF/PDB to ADOBMD unified data format

This plugin converts molecular structure files (SDF, PDB) to ADOBMD's unified
data format for molecular dynamics and ONIOM QM/MM simulations.

Features:
- Parse PDB files (atoms, bonds, residues)
- Parse SDF files (atoms, bonds, properties)
- Auto-detect bonds from atomic distances
- Define QM regions by atom index or molecule
- Generate LAMMPS-style data file with QM/MM support

This file is part of Quantum Analysis Helper.
Quantum Analysis Helper is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Quantum Analysis Helper is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Copyright (C) [2026] Acxel David Orozco Baldomero
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging

from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QIcon
from PySide6.QtCore import QObject

from plugins.libs.plugin_manager import PluginInfo

class ADOBMDConverter(QObject):
    """Main plugin class for ADOBMD Converter"""
    
    def __init__(self, plugin_info: PluginInfo):
        super().__init__()
        self.plugin_info = plugin_info
        self.ui_widget = None
        self.icon = None
        
        # Load icon if available
        plugin_dir = Path(__file__).parent
        icon_path = plugin_dir / "icon.png"
        if icon_path.exists():
            self.icon = QIcon(str(icon_path))
    
    def initialize(self) -> bool:
        """Initialize plugin resources"""
        try:
            logging.info(f"Initializing plugin {self.plugin_info.name} version {self.plugin_info.version}")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize plugin: {e}")
            return False
    
    def get_widget(self) -> Optional[QWidget]:
        """Get the main UI widget for this plugin"""
        if self.ui_widget is None:
            try:
                from .converter_widget import ConverterWidget
                self.ui_widget = ConverterWidget(self)
            except Exception as e:
                logging.error(f"Failed to create widget: {e}")
                return None
        return self.ui_widget
    
    def cleanup(self):
        """Clean up plugin resources"""
        if self.ui_widget:
            self.ui_widget.deleteLater()
            self.ui_widget = None

# Plugin factory function (required by plugin_manager)
Plugin = ADOBMDConverter