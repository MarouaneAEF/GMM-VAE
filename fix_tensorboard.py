#!/usr/bin/env python
"""
Script pour contourner les problèmes de TensorBoard avec PyTorch sur Apple Silicon.
À utiliser comme alternative à torch.utils.tensorboard.
"""

import os
import datetime
import json

class SimpleTensorboardAlternative:
    """Alternative simple à SummaryWriter pour enregistrer les métriques sans dépendre de TensorBoard."""
    
    def __init__(self, log_dir='runs'):
        """Initialise le logger avec un répertoire pour stocker les journaux."""
        self.log_dir = log_dir
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.experiment_dir = os.path.join(log_dir, f'experiment_{timestamp}')
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.data = {}
        
    def add_scalar(self, tag, scalar_value, global_step=None):
        """Enregistre une valeur scalaire."""
        if tag not in self.data:
            self.data[tag] = []
        
        step = global_step if global_step is not None else len(self.data[tag])
        self.data[tag].append((step, float(scalar_value)))
        self._save_data()
        
    def _save_data(self):
        """Sauvegarde les données dans un fichier JSON."""
        output_file = os.path.join(self.experiment_dir, 'metrics.json')
        with open(output_file, 'w') as f:
            json.dump(self.data, f, indent=2)
            
    def close(self):
        """Ferme le logger."""
        self._save_data()
        
    def __del__(self):
        """S'assure que les données sont sauvegardées lors de la suppression de l'objet."""
        self._save_data()

def create_writer(log_dir='runs'):
    """Crée une instance de SimpleTensorboardAlternative."""
    return SimpleTensorboardAlternative(log_dir)

# Comment utiliser:
# from fix_tensorboard import create_writer
# writer = create_writer()
# writer.add_scalar('loss', 0.5, epoch) 