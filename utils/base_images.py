import matplotlib.pyplot as plt
import math
from pathlib import Path
from PIL import Image

# --- CONFIGURACIÓN ---
FOLDER_PATH = Path("renders")
OBJECT_ID = ""
EXTENSION = "*.png" # Ajusta a .jpg si es necesario

def plot_26_views(folder_path, object_id):
    # Obtener todas las imágenes y sortearlas para que mantengan el orden del pipeline
    img_paths = sorted(list(folder_path.glob(EXTENSION)))
    
    # Limitamos a las primeras 26 para cumplir con tu visualización
    img_paths = img_paths[:26]
    
    if not img_paths:
        print(f"No se encontraron imágenes en: {folder_path}")
        return

    # Configuración de la grilla (7 columnas como pediste)
    cols = 7
    rows = math.ceil(len(img_paths) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle(f"Dataset Preview: {len(img_paths)} Views | Object: {object_id}", 
                 fontsize=16, fontweight='bold', y=1.02)

    for i, path in enumerate(img_paths):
        ax = axes[i // cols, i % cols]
        
        # Abrir imagen
        img = Image.open(path)
        ax.imshow(img)
        
        # El título será el nombre del archivo (donde suele venir el IND/AZ/EL)
        # Al titulo se le debe remover el .png
        titulo = path.name
        titulo = titulo.replace(".png", "")
        ax.set_title(f"{titulo}", fontsize=7, color='gray')
        ax.axis('off')

    # Limpieza: ocultar ejes vacíos si i < total de celdas
    for j in range(i + 1, rows * cols):
        axes[j // cols, j % cols].axis('off')

    plt.tight_layout()
    
    # Guardar resultado antes de mostrar
    output_preview = f"preview_{object_id}.png"
    plt.savefig(output_preview, bbox_inches='tight', dpi=150)
    print(f"Visualización guardada como: {output_preview}")
    
    plt.show()

if __name__ == "__main__":
    plot_26_views(FOLDER_PATH, OBJECT_ID)