# Résolution des problèmes

## Erreur TensorFlow/TensorBoard sur Apple Silicon

Si vous rencontrez une erreur similaire à celle-ci lors de l'exécution du script d'entraînement:

```
NotFoundError: dlopen(...) Symbol not found: __ZN3tsl8internal10LogMessageC1EPKcii
```

C'est un problème connu lié à l'incompatibilité de certaines versions de TensorFlow avec Apple Silicon.

### Solutions possibles:

1. **Utiliser l'alternative simplifiée à TensorBoard**:
   - Remplacez l'importation de TensorBoard dans votre code par notre solution personnalisée:
   ```python
   # Remplacez
   # from torch.utils.tensorboard import SummaryWriter
   
   # Par
   from fix_tensorboard import create_writer
   
   # Et au lieu de
   # writer = SummaryWriter()
   
   # Utilisez
   writer = create_writer()
   ```

2. **Installez une version spécifique de TensorFlow**:
   ```bash
   pip uninstall tensorflow tensorboard
   pip install tensorflow-macos==2.9.0 tensorboard==2.9.0
   ```

3. **Désactivez TensorBoard complètement**:
   - Modifiez `train_gmvae.py` pour désactiver l'utilisation de TensorBoard
   - Commentez les lignes utilisant TensorBoard et utilisez simplement des impressions en console pour suivre l'entraînement

## Erreur de mémoire pendant l'entraînement

Si vous rencontrez des erreurs de mémoire lors de l'entraînement:

1. Réduisez la taille du batch (`--batch-size`)
2. Utilisez le script `run_lowres_training.sh` qui réduit la résolution des images pendant l'entraînement
3. Utilisez le mode distribué avec `run_distributed_training.sh` pour répartir la charge sur plusieurs GPU si disponibles 