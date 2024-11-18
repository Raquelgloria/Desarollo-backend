


def load_dataset_from_moses(source_file, target_file):
    """
    Carga el dataset desde dos archivos Moses (uno para el idioma fuente y otro para el idioma destino).
    Cada línea de los archivos debe contener una oración de la fuente y su traducción correspondiente en el destino.
    """
    with open(source_file, 'r', encoding='utf-8') as src_f, open(target_file, 'r', encoding='utf-8') as tgt_f:
        source_lines = src_f.readlines()
        target_lines = tgt_f.readlines()
    
    # Asegurarse de que ambos archivos tengan el mismo número de líneas
    assert len(source_lines) == len(target_lines), "Las líneas de los archivos fuente y destino no coinciden."
    
    data = [{"source": source.strip(), "target": target.strip()} for source, target in zip(source_lines, target_lines)]
    
    return data
