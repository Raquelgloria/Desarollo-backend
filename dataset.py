import random

def load_dataset_from_moses(source_file, target_file):
    """
    Carga el dataset desde dos archivos Moses (uno para el idioma fuente y otro para el idioma destino).
    Si los archivos tienen un número de líneas diferente, se recortan al tamaño del archivo más corto.
    """
    with open(source_file, 'r', encoding='utf-8') as src_f, open(target_file, 'r', encoding='utf-8') as tgt_f:
        source_lines = src_f.readlines()
        target_lines = tgt_f.readlines()

    # Encontrar el número mínimo de líneas entre los dos archivos
    min_lines = min(len(source_lines), len(target_lines))

    # Recortar ambos archivos al tamaño más pequeño
    source_lines = source_lines[:min_lines]
    target_lines = target_lines[:min_lines]

    assert len(source_lines) == len(target_lines), "Las líneas de los archivos fuente y destino no coinciden."

    data = [{"source": source.strip(), "target": target.strip()} for source, target in zip(source_lines, target_lines)]
    
    # Crear traducciones inversas (es -> en)
    reversed_data = [{"source": target.strip(), "target": source.strip()} for source, target in zip(source_lines, target_lines)]
    
    # Combina ambos sets (directo + inverso)
    full_data = data + reversed_data
    
    return full_data
